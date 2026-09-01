# Hybrid Retrieval & Reranking with Reproducible IR Evaluation

A multi-stage retrieval system evaluated on [BEIR / SciFact](https://github.com/beir-cellar/beir),
where **every stage is measured against the previous one**. The point is not "I trained a model" —
it is demonstrating search-engineer thinking about the **quality / latency trade-off**.

Evaluation metrics (`nDCG@10`, `Recall@100`, `MRR@10`) are implemented **by hand** in
[`src/metrics.py`](src/metrics.py), not pulled from a library.

## Results (SciFact, 300 test queries, 5.2k documents)

| System            | nDCG@10 | Recall@100 | MRR@10 | p95 latency |
|-------------------|:-------:|:----------:|:------:|:-----------:|
| BM25 (lexical)    | 0.6523  |   0.8731   | 0.6184 |   15.0 ms   |
| Dense (MiniLM)    | 0.6451  |   0.9250   | 0.6047 |   11.4 ms   |
| Hybrid (RRF)      | 0.6840  |   0.9577   | 0.6503 |   28.0 ms   |
| + Cross-encoder   | **0.6910** | **0.9577** | **0.6585** |   491 ms   |

*Latency is per-query p95 on CPU (Apple Silicon). BM25 index build and corpus encoding are
one-time and excluded.*

## Pipeline

1. **BM25 baseline** — lexical retrieval (`rank_bm25`). A regex tokenizer that strips punctuation
   alone lifted nDCG@10 from **0.5597 → 0.6523 (+0.093)** over naive `.split()` — preprocessing
   pays before any ML.
2. **Dense retrieval** — bi-encoder (`all-MiniLM-L6-v2`), embeddings normalized, FAISS `IndexFlatIP`
   (exact; the corpus is too small to need ANN). Inner product on unit vectors = cosine similarity.
3. **Hybrid fusion** — Reciprocal Rank Fusion over BM25 + dense. RRF operates on **ranks, not
   scores** (BM25 scores and cosine are not comparable), `k = 60`.
4. **Cross-encoder reranking** — `ms-marco-MiniLM-L-6-v2` scores each `(query, doc)` pair jointly
   over the top-100 hybrid candidates. Too slow to run over the full corpus; that is why it only
   reranks a shortlist.

## Key findings

- **Dense loses to BM25 on nDCG but wins on Recall@100.** On scientific text with exact terminology
  (gene names, compounds), lexical match is sharp at the top; the dense model — trained on general
  web text — is out-of-domain and blurs precise terms, but its semantics cast a **wider net**
  (Recall@100 0.925 vs 0.873). They are **complementary, not competitors** — which is exactly why
  hybrid fusion beats both on every metric.
- **The cross-encoder buys +0.007 nDCG for ~17× the latency** (28 ms → 491 ms). Not a failure — a
  cost/benefit finding. Hybrid is already near the ceiling (SciFact has ~1 relevant doc/query,
  binary relevance) and reranking cannot improve recall — it only reorders the shortlist, so the
  4.2% of relevant docs outside the top-100 are unreachable. On a dataset with more relevant docs
  per query, or weaker first-stage retrieval, the reranker would pay off more.
- **BM25 is slower than dense here (15 ms vs 11 ms).** `rank_bm25` scores all 5.2k docs in pure
  Python; FAISS flat search is optimized C++. A production BM25 over a real inverted index
  (Lucene / Pyserini) would be sub-millisecond. This motivates the optional Rust reimplementation.

## Run

```bash
pip install -r requirements.txt      # rank_bm25, numpy, sentence-transformers, faiss-cpu
python src/download_data.py          # downloads BEIR SciFact into data/scifact/

# run from the repo root (scripts read data/scifact/ relative to the working dir)
python src/bm25_baseline.py
python src/dense_retrieval.py
python src/hybrid_retrieval.py
python src/rerank.py
```

## Repo layout

```
src/
  data_loader.py       # load corpus / queries / qrels (test split only — zero-shot)
  metrics.py           # ndcg@k, recall@k, mrr@k — by hand
  bm25_baseline.py     # stage 1
  dense_retrieval.py   # stage 2
  hybrid_retrieval.py  # stage 3 (RRF)
  rerank.py            # stage 4 (cross-encoder)
data/scifact/          # BEIR SciFact (gitignored)
```

## Next

- RAG layer: citation-grounded answers over top-k + LLM-as-judge for faithfulness.
- Second BEIR dataset (NFCorpus) for a cross-dataset table.
- Rust: reimplement the BM25 scorer, measure speedup vs `rank_bm25`.

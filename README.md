# Hybrid Retrieval and Reranking with Reproducible IR Evaluation

This is a project I built to learn how modern retrieval systems are put together, one stage
at a time. It starts from a plain BM25 baseline and adds dense retrieval, hybrid fusion, and a
cross-encoder reranker on top. The idea was not to train a big model but to measure each stage
against the one before it, so I could actually see where the quality comes from and what it costs
in latency.

Everything is evaluated on BEIR / SciFact, which ships with qrels, so the nDCG numbers are real.
The evaluation metrics (`nDCG@10`, `Recall@100`, `MRR@10`) are written
by hand in [`src/metrics.py`](src/metrics.py) instead of pulled from a library.

## Results (SciFact, 300 test queries, 5.2k documents)

| System           | nDCG@10 | Recall@100 | MRR@10 | p95 latency |
|------------------|:-------:|:----------:|:------:|:-----------:|
| BM25 (lexical)   | 0.6523  |   0.8731   | 0.6184 |   15.0 ms   |
| Dense (MiniLM)   | 0.6451  |   0.9250   | 0.6047 |   11.4 ms   |
| Hybrid (RRF)     | 0.6840  |   0.9577   | 0.6503 |   28.0 ms   |
| + Cross-encoder  | 0.6910  |   0.9577   | 0.6585 |   491 ms    |

Latency is per-query p95 on CPU (Apple Silicon). Building the BM25 index and encoding the corpus
are one-time steps and are not included in the per-query numbers.

## Pipeline

1. BM25 baseline: lexical retrieval with `rank_bm25`. Switching the tokenizer from a naive
   `.split()` to a small regex that strips punctuation raised nDCG@10 from 0.5597 to 0.6523, so
   preprocessing already pays off before any machine learning.
2. Dense retrieval: a bi-encoder (`all-MiniLM-L6-v2`) with normalized embeddings and a FAISS
   `IndexFlatIP` index. The index is exact because the corpus is small enough that approximate
   search is not needed. Inner product on unit vectors is the same as cosine similarity.
3. Hybrid fusion: Reciprocal Rank Fusion over the BM25 and dense lists. RRF works on ranks, not
   scores, which matters because BM25 scores and cosine similarity are not on the same scale
   (`k = 60`).
4. Cross-encoder reranking: `ms-marco-MiniLM-L-6-v2` scores each (query, document) pair jointly
   over the top-100 hybrid candidates. It is too slow to run over the whole corpus, which is exactly
   why it only reranks a shortlist.

## What I found

Dense retrieval loses to BM25 on nDCG but wins on Recall@100. On scientific text with exact
terminology like gene names and compounds, lexical matching is sharp at the top of the list. The
dense model is trained on general web text, so it is out of domain here and blurs precise terms,
but its semantics cast a wider net (Recall@100 0.925 vs 0.873). The two are complementary rather
than competing, which is why hybrid fusion beats both of them on every metric.

The cross-encoder adds only 0.007 nDCG while making a query about 17 times slower (28 ms to
491 ms). I think that is a useful result rather than a disappointing one. Hybrid is already close
to the ceiling on this dataset (SciFact has roughly one relevant document per query, with binary
relevance), and reranking cannot improve recall because it only reorders the shortlist. The 4.2%
of relevant documents that are outside the top-100 stay unreachable. On a dataset with more
relevant documents per query, or with a weaker first stage, the reranker would pay off more.

## Running it

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
  data_loader.py       # load corpus / queries / qrels (test split only, zero-shot)
  metrics.py           # ndcg@k, recall@k, mrr@k, by hand
  bm25_baseline.py     # stage 1
  dense_retrieval.py   # stage 2
  hybrid_retrieval.py  # stage 3 (RRF)
  rerank.py            # stage 4 (cross-encoder)
data/scifact/          # BEIR SciFact (gitignored)
```

## Next steps

- RAG layer: citation-grounded answers over the top-k results, plus an LLM-as-judge for faithfulness.
- A second BEIR dataset (NFCorpus) for a cross-dataset table.

import time

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from data_loader import load_corpus, load_queries, load_qrels
from metrics import ndcg_at_k, recall_at_k, mrr_at_k

MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(corpus, model):
    """corpus: {doc_id: "title text"}
    Vrati (index, doc_ids):
      - doc_ids = fiksna lista, redosled ZAKLJUCAN (isti trik kao BM25)
      - index = FAISS IndexFlatIP nad normalizovanim vektorima
    Zasto IndexFlatIP + normalizacija:
      inner product na JEDINICNIM vektorima == kosinusna slicnost.
    """
    doc_ids = list(corpus.keys())
    texts = [corpus[did] for did in doc_ids]

    # encode: (5183, 384) float32. normalize_embeddings=True -> jedinicni vektori
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dim = emb.shape[1]                 # 384
    index = faiss.IndexFlatIP(dim)     # IP = inner product
    index.add(emb)                     # ubaci sve vektore
    return index, doc_ids


def search(index, doc_ids, query, model, top_n=100):
    """Vrati listu doc_id sortiranu po slicnosti desc, duzine top_n."""
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    scores, idxs = index.search(q_emb, top_n)   # idxs: (1, top_n) INDEKSI, ne doc_id

    return [doc_ids[i] for i in idxs[0]]


def run_dense(data_dir="data/scifact"):
    corpus  = load_corpus(f"{data_dir}/corpus.jsonl")
    queries = load_queries(f"{data_dir}/queries.jsonl")
    qrels   = load_qrels(f"{data_dir}/qrels/test.tsv")

    model = SentenceTransformer(MODEL_NAME)
    index, doc_ids = build_index(corpus, model)

    ndcgs, recalls, mrrs, latencies = [], [], [], []
    for qid in qrels:                       # samo test upiti
        relevant = qrels[qid]

        t0 = time.perf_counter()
        ranked = search(index, doc_ids, queries[qid], model, top_n=100)
        latencies.append((time.perf_counter() - t0) * 1000)   # ms (enkodovanje upita + faiss search)

        ndcgs.append(  ndcg_at_k(ranked, relevant, 10)  )
        recalls.append(recall_at_k(ranked, relevant, 100))
        mrrs.append(   mrr_at_k(ranked, relevant, 10)   )

    n = len(ndcgs)
    latencies.sort()
    p95 = latencies[int(0.95 * n)]
    print(f"upita: {n}")
    print(f"nDCG@10:    {sum(ndcgs)/n:.4f}")
    print(f"Recall@100: {sum(recalls)/n:.4f}")
    print(f"MRR@10:     {sum(mrrs)/n:.4f}")
    print(f"p95:        {p95:.2f} ms/upit")


if __name__ == "__main__":
    run_dense()

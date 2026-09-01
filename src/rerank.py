import time

from sentence_transformers import SentenceTransformer, CrossEncoder

from data_loader import load_corpus, load_queries, load_qrels
from metrics import ndcg_at_k, recall_at_k, mrr_at_k

import bm25_baseline as bm25mod
import dense_retrieval as densemod
from hybrid_retrieval import rrf_fuse

CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank(ce, query, candidates, corpus, top_n=100):
    """candidates: lista doc_id (hybrid top-100).
    Cross-encoder skoruje svaki (query, dok) par, presortira.
    Vrati listu doc_id po CE skoru desc.
    """
    pairs = [[query, corpus[did]] for did in candidates]

    scores = ce.predict(pairs)   

    ranked = sorted(zip(scores, candidates), key= lambda x: x[0], reverse=True)
    return [did for _, did in ranked[:top_n]]


def run_rerank(data_dir="data/scifact"):
    corpus  = load_corpus(f"{data_dir}/corpus.jsonl")
    queries = load_queries(f"{data_dir}/queries.jsonl")
    qrels   = load_qrels(f"{data_dir}/qrels/test.tsv")

    bm25, bm25_doc_ids = bm25mod.build_index(corpus)
    model = SentenceTransformer(densemod.MODEL_NAME)
    dense_index, dense_doc_ids = densemod.build_index(corpus, model)
    ce = CrossEncoder(CE_MODEL)

    ndcgs, recalls, mrrs, latencies = [], [], [], []
    for qid in qrels:
        q = queries[qid]
        bm25_ranked  = bm25mod.search(bm25, bm25_doc_ids, q, top_n=100)
        dense_ranked = densemod.search(dense_index, dense_doc_ids, q, model, top_n=100)
        candidates = rrf_fuse(bm25_ranked, dense_ranked, top_n=100)

        t0 = time.perf_counter()
        ranked = rerank(ce, q, candidates, corpus, top_n=100)
        latencies.append((time.perf_counter() - t0) * 1000)

        relevant = qrels[qid]
        ndcgs.append(ndcg_at_k(ranked, relevant, 10))
        recalls.append(recall_at_k(ranked, relevant, 100))
        mrrs.append(mrr_at_k(ranked, relevant, 10))

    n = len(ndcgs)
    latencies.sort()
    p95 = latencies[int(0.95 * n)]
    print(f"upita: {n}")
    print(f"nDCG@10:      {sum(ndcgs)/n:.4f}")
    print(f"Recall@100:   {sum(recalls)/n:.4f}")
    print(f"MRR@10:       {sum(mrrs)/n:.4f}")
    print(f"rerank p95:   {p95:.1f} ms/upit")


if __name__ == "__main__":
    run_rerank()

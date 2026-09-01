import time

from sentence_transformers import SentenceTransformer

from data_loader import load_corpus, load_queries, load_qrels
from metrics import ndcg_at_k, recall_at_k, mrr_at_k

import bm25_baseline as bm25mod
import dense_retrieval as densemod

K_RRF = 60


def rrf_fuse(bm25_ranked, dense_ranked, k=K_RRF, top_n=100):
    """Dve rang-liste doc_id -> jedna fuzionisana lista doc_id (desc po rrf_score).
    rrf_score(d) = sum 1/(k + rank) preko listi gde se d pojavljuje. rank 1-indeksiran.
    """
    scores = {}   # doc_id -> rrf_score
    for ranked in (bm25_ranked, dense_ranked):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1/(k+rank)
            
    ranked_fused = sorted(scores, key=lambda d: scores[d], reverse=True)
    return ranked_fused[:top_n]


def run_hybrid(data_dir="data/scifact"):
    corpus  = load_corpus(f"{data_dir}/corpus.jsonl")
    queries = load_queries(f"{data_dir}/queries.jsonl")
    qrels   = load_qrels(f"{data_dir}/qrels/test.tsv")

    # BM25 indeks
    bm25, bm25_doc_ids = bm25mod.build_index(corpus)

    # Dense indeks
    model = SentenceTransformer(densemod.MODEL_NAME)
    dense_index, dense_doc_ids = densemod.build_index(corpus, model)

    ndcgs, recalls, mrrs, latencies = [], [], [], []
    for qid in qrels:
        q = queries[qid]
        relevant = qrels[qid]

        t0 = time.perf_counter()
        bm25_ranked  = bm25mod.search(bm25, bm25_doc_ids, q, top_n=100)
        dense_ranked = densemod.search(dense_index, dense_doc_ids, q, model, top_n=100)
        ranked = rrf_fuse(bm25_ranked, dense_ranked, top_n=100)
        latencies.append((time.perf_counter() - t0) * 1000)   # ms (bm25 + dense + rrf)

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
    run_hybrid()

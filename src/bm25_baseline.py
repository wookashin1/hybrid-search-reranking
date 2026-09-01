import re
import time

from rank_bm25 import BM25Okapi

from data_loader import load_corpus, load_queries, load_qrels
from metrics import ndcg_at_k, recall_at_k, mrr_at_k


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_index(corpus):
    """corpus: {doc_id: "title text"}
    Vrati (bm25, doc_ids) gde:
      - doc_ids = fiksna lista ID-eva, redosled ZAKLJUCAN
      - bm25 indeksiran nad tokenima u ISTOM redosledu
    Zamka: get_scores() vraca niz poravnat po INDEKSU, ne po doc_id.
           Zato cuvamo doc_ids da mozemo indeks -> doc_id.
    """
    doc_ids = list(corpus.keys())
    corpus_tokens = [tokenize(corpus[did]) for did in doc_ids]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25, doc_ids


def search(bm25, doc_ids, query, top_n=100):
    """Vrati listu doc_id sortiranu po skoru desc, duzine top_n."""
    scores = bm25.get_scores(tokenize(query))   # niz, poravnat sa doc_ids po indeksu
    docs = sorted(zip(scores, doc_ids), key= lambda x : x[0], reverse=True)
    return [doc_id for _,doc_id in docs[:top_n]]


def run_bm25(data_dir="data/scifact"):
    corpus  = load_corpus(f"{data_dir}/corpus.jsonl")
    queries = load_queries(f"{data_dir}/queries.jsonl")
    qrels   = load_qrels(f"{data_dir}/qrels/test.tsv")

    bm25, doc_ids = build_index(corpus)

    ndcgs, recalls, mrrs, latencies = [], [], [], []
    for qid in qrels:                       # ZAMKA: samo test upiti (iz qrels), ne svih 1109
        relevant = qrels[qid]

        t0 = time.perf_counter()
        ranked = search(bm25, doc_ids, queries[qid], top_n=100)
        latencies.append((time.perf_counter() - t0) * 1000)   # ms

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
    run_bm25()

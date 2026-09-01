import math

#ranked : lista doc_id sortirana po skoru - najbolji prvi za JEDAN upit
#relevant: {doc_id: score} iz qrels za taj upit

def recall_at_k(ranked, relevant, k):

    top_k = ranked[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)

    return hits/len(relevant)

def mrr_at_k(ranked, relevant, k):

    for i, doc_id in enumerate(ranked[:k], start = 1):
        if doc_id in relevant:
            return 1/i
        
    return 0.0

def ndcg_at_k(ranked, relevant, k):

    # gains po TVOM rangiranju (0 ako doc nije u qrels)
    gains = [relevant.get(doc_id, 0) for doc_id in ranked[:k]]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))

    # idealno rangiranje: najveci relevantni prvi
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))

    return dcg / idcg if idcg > 0 else 0.0


if __name__ == "__main__":
    ranked = ["d2", "d1", "d3"]    
    relevant = {"d1": 1}
    print(ndcg_at_k(ranked, relevant, 10))   # ocekivano 0.6309
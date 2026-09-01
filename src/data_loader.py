import json

def load_corpus(path):

    ##corpus = {id : title + text}
    corpus = {}

    with open(path) as f:
        for line in f:
            doc = json.loads(line)

            doc_id = doc["_id"]
            text = doc["title"] + " " + doc["text"]

            corpus[doc_id] = text

    return corpus


def load_queries(path):

    ## queries = {query_id: text}

    queries = {}

    with open(path) as f:
        for line in f:
            query = json.loads(line)
            id = query["_id"]
            text = query["text"]

            queries[id] = text

    return queries

def load_qrels(path):
    #qrels/test.csv : {query_id: {doc_id: score}}

    qrels = {}
    doc = {}
    with open(path) as f:

        next(f)
        for line in f:
            qid, did, score = line.split("\t")
            qrels.setdefault(qid, {})[did] = int(score)

    return qrels
            

if __name__ == "__main__":
    corpus = load_corpus("/Users/vule/Desktop/hybrid-search-reranking/data/scifact/corpus.jsonl")
    queries = load_queries("/Users/vule/Desktop/hybrid-search-reranking/data/scifact/queries.jsonl")
    qrels = load_qrels("/Users/vule/Desktop/hybrid-search-reranking/data/scifact/qrels/test.tsv")
    #print(len(corpus))

    print(len(qrels))




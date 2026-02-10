import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import re

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("indexes/dense.index")
bm25 = pickle.load(open("indexes/bm25.pkl", "rb"))
corpus = pickle.load(open("indexes/corpus.pkl", "rb"))

def dense_retrieve(query, k=10):
    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, k)
    return list(zip(idxs[0], scores[0]))

def sparse_retrieve(query, k=10):
    scores = bm25.get_scores(word_tokenize(query.lower()))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(i, scores[i]) for i in top]

def rrf(dense, sparse, k=60):
    scores = {}
    for r,(i,_) in enumerate(dense):
        scores[i] = scores.get(i,0) + 1/(k+r+1)
    for r,(i,_) in enumerate(sparse):
        scores[i] = scores.get(i,0) + 1/(k+r+1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_fused_for_query(query, top_k_dense=10, top_k_sparse=10, top_n=5, min_query_token_len=4, min_overlap_matches=2):
    """Retrieve dense and sparse results, fuse them and filter by simple token overlap.

    Returns a list of (doc_index, score) for the top_n fused results. If none of the
    top fused docs contain any of the meaningful query tokens, returns an empty list.
    """
    dense = dense_retrieve(query, k=top_k_dense)
    sparse = sparse_retrieve(query, k=top_k_sparse)
    fused = rrf(dense, sparse)
    fused = fused[:top_n]

    # build query tokens (ignore short tokens)
    query_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= min_query_token_len]
    if not query_tokens:
        return fused

    # filter fused results to those that contain at least `min_overlap_matches` query tokens
    filtered = []
    for i, score in fused:
        text = corpus[i]["text"].lower()
        matches = sum(1 for tok in query_tokens if tok in text)
        if matches >= min_overlap_matches:
            filtered.append((i, score))

    # if none pass the stricter overlap test, return empty to signal no evidence
    if not filtered:
        return []

    return filtered[:top_n]

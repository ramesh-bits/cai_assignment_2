import faiss, pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("dense.index")
bm25 = pickle.load(open("bm25.pkl", "rb"))
corpus = pickle.load(open("corpus.pkl", "rb"))

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

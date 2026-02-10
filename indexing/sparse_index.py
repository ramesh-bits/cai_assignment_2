import json
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

"""Build BM25 index from the corpus chunks.

This script expects `data/corpus_chunks.json` (created by `indexing/build_corpus.py`).
"""

corpus = json.load(open("data/corpus_chunks.json"))
tokenized = [word_tokenize(c["text"].lower()) for c in corpus]

bm25 = BM25Okapi(tokenized)

pickle.dump(bm25, open("indexes/bm25.pkl", "wb"))
pickle.dump(corpus, open("indexes/corpus.pkl", "wb"))

print("BM25 index built")
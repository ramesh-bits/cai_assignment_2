import json
import pickle
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

corpus = json.load(open("corpus_chunks.json"))
tokenized = [word_tokenize(c["text"].lower()) for c in corpus]

bm25 = BM25Okapi(tokenized)

pickle.dump(bm25, open("bm25.pkl", "wb"))
pickle.dump(corpus, open("corpus.pkl", "wb"))

print("BM25 index built")
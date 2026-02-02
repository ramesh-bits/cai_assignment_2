import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

corpus = json.load(open("corpus_chunks.json"))
texts = [c["text"] for c in corpus]

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "dense.index")
pickle.dump(corpus, open("corpus.pkl", "wb"))

print("Dense index built")
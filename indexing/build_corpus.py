import wikipediaapi
import json
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

wiki = wikipediaapi.Wikipedia("en")

def chunk_text(text, chunk_size=300, overlap=50):
    tokens = word_tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i+chunk_size]
        if len(chunk) >= 200:
            chunks.append(" ".join(chunk))
    return chunks

def build_corpus(urls, out_file="corpus_chunks.json"):
    corpus = []
    chunk_id = 0

    for url in urls:
        title = url.split("/")[-1].replace("_", " ")
        page = wiki.page(title)

        if not page.exists():
            continue

        chunks = chunk_text(page.text)
        for c in chunks:
            corpus.append({
                "chunk_id": f"chunk_{chunk_id}",
                "url": page.fullurl,
                "title": page.title,
                "text": c
            })
            chunk_id += 1

    with open(out_file, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"Saved {len(corpus)} chunks")

if __name__ == "__main__":
    fixed = json.load(open("data/fixed_urls.json"))
    random_urls = json.load(open("data/random_urls.json"))
    build_corpus(fixed + random_urls)
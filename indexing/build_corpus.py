import wikipediaapi
import json
import nltk
from nltk.tokenize import sent_tokenize
import time

# Provide a proper user agent per Wikimedia policy
wiki = wikipediaapi.Wikipedia(language="en", user_agent="cai_assignment/1.0 (https://example.com)")

TIMEOUT_SECONDS = 10  # Request timeout
MAX_RETRIES = 2  # Retry failed requests


def chunk_text(text, chunk_size=250, overlap=50):
    sentences = sent_tokenize(text)
    chunks, current = [], []

    for sent in sentences:
        current.append(sent)
        if len(" ".join(current).split()) >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-overlap//10:]

    if current:
        chunks.append(" ".join(current))

    return chunks

def build_corpus(urls, out_file="data/corpus_chunks.json"):
    corpus = []
    chunk_id = 0
    failed_urls = []

    print(f"Processing {len(urls)} URLs...")

    for idx, url in enumerate(urls, 1):
        title = url.split("/")[-1].replace("_", " ")
        
        # Retry logic for timeouts/network errors
        page = None
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[{idx}/{len(urls)}] Processing: {title} (attempt {attempt+1}/{MAX_RETRIES})", end=" ")
                page = wiki.page(title)
                print("")
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"✗ ({type(e).__name__}), retrying...")
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"✗ (Skipped - {type(e).__name__}: {str(e)[:50]})")
                    failed_urls.append((url, str(e)))
                    break
        
        # If page failed to fetch, skip it
        if page is None or not page.exists():
            print(f"  Page does not exist or failed to load: {title}")
            continue

        try:
            chunks = chunk_text(page.text)
            for c in chunks:
                corpus.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "url": page.fullurl,
                    "title": page.title,
                    "text": c
                })
                chunk_id += 1
        except Exception as e:
            print(f"  Error chunking {title}: {e}")
            failed_urls.append((url, f"Chunking error: {str(e)[:50]}"))
            continue

    # ensure output directory exists
    import os
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    with open(out_file, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\n Saved {len(corpus)} chunks to {out_file}")
    if failed_urls:
        print(f"⚠ {len(failed_urls)} URLs failed (skipped gracefully):")
        for url, reason in failed_urls[:5]:
            print(f"  - {url}: {reason}")
        if len(failed_urls) > 5:
            print(f"  ... and {len(failed_urls)-5} more")

def _safe_load_list(path):
    try:
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


if __name__ == "__main__":
    # Ensure NLTK 'punkt' tokenizer is available
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download("punkt_tab")
        except Exception as e:
            print(f"Warning: NLTK download failed: {e}")
            print("Proceeding anyway - punkt may already be available")

    fixed = _safe_load_list("data/fixed_urls.json")
    random_urls = _safe_load_list("data/random_urls.json")

    urls = fixed + random_urls
    if not urls:
        print("No URLs found in data/fixed_urls.json or data/random_urls.json.")
        print("Please populate them with a JSON list of Wikipedia URLs, or use sample URLs.")
        print("\nExample:")
        print('  {"urls": ["https://en.wikipedia.org/wiki/Machine_learning", ...]}')
    else:
        print(f"Starting corpus build with {len(urls)} URLs...\n")
        build_corpus(urls)
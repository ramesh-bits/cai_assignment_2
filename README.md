## Group No. : 17

## Group Member Names:

- Poornima T           - 2024aa05600
- Ramesh Gehlot        - 2024aa05599
- Shubham Luthra       - 2024aa05596
- Sunil Prakash Inteti - 2024aa05597
- Thillaikkarasan M    - 2024aa05955

# Hybrid RAG System

This repository implements a Retrieval-Augmented Generation (RAG) system that combines dense and sparse retrieval with a stable generation pipeline and an automated evaluation harness.

## Highlights

- Hybrid retrieval using dense (FAISS + sentence-transformers) and sparse (BM25) methods fused with Reciprocal Rank Fusion (RRF).
- Grounded generation with a deterministic prompt pipeline and an extractive fallback to avoid hallucinations and runtime crashes.
- An evaluation pipeline that can auto-generate test questions, run retrieval+generation, and compute metrics (MRR, Precision@K, Recall@K, NDCG@K, ROUGE-L, F1/EM).
- Topic-restricted URL generation for focused corpora sampling.

## Project Structure

```
├── app.py                        # Streamlit web interface
├── run_pipeline.py               # Orchestrates full pipeline (indexing -> evaluation)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
|
├── indexing/                     # Data preparation & indexing
│   ├── build_corpus.py           # Fetch & chunk Wikipedia articles
│   ├── dense_index.py            # Build FAISS dense index
│   └── sparse_index.py           # Build BM25 index
│
├── rag/                          # Core RAG components
│   ├── retriever.py              # Dense, sparse retrieval + fused API
│   ├── generator.py              # Grounded generator + extractive fallback
│   └── rrf.py                    # RRF fusion logic
│
├── evaluation/                   # Evaluation pipeline & metrics
│   ├── evaluate.py               # Evaluation runner (compute/save metrics)
│   ├── generate_questions.py     # Test question generation (LLM + extractive fallback)
│   ├── evaluation_pipeline.py    # Orchestrator for running end-to-end experiments
│   └── metrics.py                # Metric utilities
│
├── indexing/                     # URL utilities (topic filtering)
│   └── generate_urls.py          # Topic-restricted/random URL generation (CLI)
|
├── data/                         # Generated data & sources (may be large)
│   ├── fixed_urls.json           # Curated Wikipedia URLs
│   ├── random_urls.json          # Random Wikipedia URLs
│   ├── corpus_chunks.json        # Processed text chunks
│   └── questions.json            # Generated test Q/A
|
└── evaluation/results/           # Evaluation outputs (CSV, JSON, reports)
```

## Quick Setup

1) Install dependencies (use a virtualenv for reproducibility):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

2) Build corpus & indices (choose fixed/random URL sources):

```bash
# Option: use curated fixed URLs + generate random ones
python indexing/generate_urls.py --count 300 --topics "machine learning,python" --out data/random_urls.json --force
python indexing/build_corpus.py
python indexing/dense_index.py
python indexing/sparse_index.py
```

Or run the complete orchestrator:

```bash
python run_pipeline.py
```

3) Run Streamlit UI:

```bash
python -m streamlit run app.py
```

Open: http://localhost:8501

## Evaluation

The repository includes an automated evaluation flow:

- `evaluation/generate_questions.py` — generates diverse question types (factual, comparative, inferential, multi-hop) using an LLM when available, with an extractive fallback that creates deterministic QA from corpus text.
- `evaluation/evaluate.py` & `evaluation/evaluation_pipeline.py` — run retrieval+generation for test questions and compute metrics (MRR, Precision@K, Recall@K, NDCG@K, ROUGE-L, F1/EM).

Run an evaluation (example):

```bash
# ensure virtualenv is active
python evaluation/evaluation_pipeline.py --questions data/questions.json --outdir evaluation/results --topk 5
```

Outputs will be saved to `evaluation/results/` (CSV, metrics.json, report)

## Topic-restricted URL generation

`indexing/generate_urls.py` supports topic filtering via `--topics` and CLI flags:

```bash
python indexing/generate_urls.py --count 200 --topics "machine learning,python" --out data/random_urls.json --force
```

This is useful to create focused corpora for targeted evaluation.

## Usage (programmatic)

```python
from rag.retriever import dense_retrieve, sparse_retrieve, rrf, corpus
from rag.generator import generate_answer

query = "What is machine learning?"
dense_results = dense_retrieve(query, k=10)
sparse_results = sparse_retrieve(query, k=10)
fused = rrf(dense_results, sparse_results)[:5]
contexts = [corpus[i]["text"] for i, _ in fused]
answer = generate_answer(query, contexts)
print(answer)
```

## Notes & Troubleshooting

- Use the project's virtualenv Python when running scripts to ensure installed packages are available (e.g., `.venv/bin/python indexing/generate_urls.py ...`).
- If the generator LLM is unavailable or fails (OOM/segfault), the system uses an extractive fallback to produce safe, grounded answers and deterministic QA for evaluation.
- If you see `FileNotFoundError: dense.index not found`, run `python run_pipeline.py` to rebuild indices.
- For SSL issues while fetching Wikipedia, set the certifi path:

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

## Dependencies

Core packages: `faiss-cpu`, `sentence-transformers`, `rank_bm25`, `transformers`, `nltk`, `requests`, `beautifulsoup4`, `lxml`, `streamlit`, `rouge-score`.
See `requirements.txt` for the full list.

## Recommended next steps

- If you want more diverse questions without using an external LLM, consider enabling the expanded extractive templates in `evaluation/generate_questions.py` (already available) and regenerating `data/questions.json` with `--force`.
- Run the topic-restricted URL generator with the virtualenv Python to produce focused corpora for evaluation.

---
If you'd like, I can now: add a few-shot prompt per qtype to `generate_questions.py` (requires LLM), or expand extractive templates and regenerate `data/questions.json` deterministically. Which do you prefer?
- Uses Google FLAN-T5 model for text2text-generation
- Takes query + context and generates natural language answers

### Evaluation (`evaluation/evaluate.py`)
- Computes ROUGE-1, ROUGE-2, ROUGE-L scores
- Evaluates answer quality against gold standard

## Configuration

### Environment Variables (Optional)
```bash
# For SSL certificate issues during Wikipedia fetches
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

### Tuning Parameters

**In `indexing/build_corpus.py`:**
- `chunk_size`: Text chunk size (default: 300 tokens)
- `overlap`: Chunk overlap (default: 50 tokens)

**In `rag/retriever.py`:**
- `k`: Number of top results per retriever (default: 10)
- `rrf_k`: RRF constant (default: 60)

**In `evaluation/evaluate.py`:**
- Metric weights and thresholds

## Troubleshooting

### SSL Certificate Errors
```bash
# Install/upgrade certifi
pip install --upgrade certifi

# Set SSL certificate environment variable
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"

# For macOS with official Python installer, run:
/Applications/Python\ 3.*/Install\ Certificates.command
```

### NLTK Data Not Found
```bash
python -c "import nltk; nltk.download('punkt')"
```

### Wikipedia API Failures
The scripts have built-in fallbacks to 10+ hardcoded Wikipedia URLs. If API requests fail, the system will automatically use these fallback URLs.

### Missing Index Files
If you get `FileNotFoundError: dense.index not found`, run:
```bash
python run_pipeline.py
```

### NLTK punkt Download SSL Issues
The `build_corpus.py` script now checks for punkt and gracefully handles SSL failures:
```bash
python indexing/build_corpus.py
```
If it fails, check the printed instructions or manually download:
```bash
# Add to~/.zshrc for persistence:
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

## Dependencies

Core packages:
- `faiss-cpu`: Dense vector indexing
- `sentence-transformers`: Semantic embeddings
- `rank-bm25`: Sparse retrieval
- `transformers`: FLAN-T5 language model
- `nltk`: Text tokenization
- `requests`: HTTP client for Wikipedia
- `beautifulsoup4`, `lxml`: HTML parsing
- `streamlit`: Web UI
- `rouge-score`: Evaluation metrics

See `requirements.txt` for full list.

## Performance Notes

- **Dense Index Build**: ~5-10 minutes for 500 articles (GPU: 2-3 min)
- **Sparse Index Build**: ~1 minute
- **Query Latency**: 2-5 seconds (dense embedding + search + FLAN-T5 generation)
- **Memory**: ~2-4 GB (FAISS + transformers models)
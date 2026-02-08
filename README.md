## Group No. : 17

## Group Member Names:

- Poornima T           - 2024aa05600
- Ramesh Gehlot        - 2024aa05599
- Shubham Luthra       - 2024aa05596
- Sunil Prakash Inteti - 2024aa05597
- Thillaikkarasan M    - 2024aa05955

# Hybrid RAG System

A Retrieval-Augmented Generation (RAG) system that combines dense and sparse retrieval methods using RRF (Reciprocal Rank Fusion) to provide accurate answers with retrieved context from Wikipedia.

## Overview

This project implements a hybrid retrieval system that:
- **Dense Retrieval**: Uses FAISS with sentence transformers (`all-MiniLM-L6-v2`) for semantic similarity
- **Sparse Retrieval**: Uses BM25 (Okapi) for keyword-based matching
- **Fusion**: Combines both results using RRF to rank final documents
- **Generation**: Uses Google FLAN-T5 to generate answers based on retrieved context
- **Evaluation**: Metrics using ROUGE scores for answer quality assessment

## Project Structure

```
├── app.py                      # Streamlit web interface
├── run_pipeline.py             # Orchestrates the entire pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── indexing/                   # Data preparation & indexing
│   ├── build_corpus.py         # Fetch & chunk Wikipedia articles
│   ├── dense_index.py          # Build FAISS dense index
│   └── sparse_index.py         # Build BM25 sparse index
│
├── rag/                        # Core RAG components
│   ├── retriever.py            # Dense, sparse, RRF retrieval
│   ├── generator.py            # FLAN-T5 answer generation
│   └── rrf.py                  # RRF fusion logic
│
├── evaluation/                 # Evaluation pipeline
│   ├── evaluate.py             # ROUGE metrics computation
│   ├── generate_questions.py   # Test question generation
│   └── metrics.py              # Metric utilities
│
├── data/                       # Generated data & sources
│   ├── fixed_urls.json         # Curated Wikipedia URLs
│   ├── random_urls.json        # Random Wikipedia URLs
│   ├── corpus_chunks.json      # Processed text chunks
│   └── questions.json          # Test questions
│
└── scripts/                    # Utility scripts
    ├── wiki.py                 # Enhanced Wikipedia corpus manager
    └── gen_random_wiki_urls.py # Generate Wikipedia URL lists
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Install all required packages
pip install --upgrade -r requirements.txt

# Download NLTK punkt tokenizer (one-time setup)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Generate Wikipedia Corpus

Choose one of the following options:

#### Option A: Quick Setup (Using Fixed URLs)
```bash
python scripts/wiki.py
```
This will:
- Fetch 200 fixed + 300 random Wikipedia articles
- Extract and chunk text
- Save to `processed_corpus.json`

#### Option B: Initialize URL Collections First
```bash
# Generate fixed URLs (curated set, ~200 URLs)
python scripts/gen_random_wiki_urls.py --mode init_fixed --fixed data/fixed_urls.json

# Generate random URLs (~300 URLs)
python scripts/gen_random_wiki_urls.py --mode new_random --random data/random_urls.json
```

Then build indices:
```bash
python indexing/build_corpus.py    # Fetch & chunk articles
python indexing/dense_index.py     # Build FAISS index
python indexing/sparse_index.py    # Build BM25 index
```

Or run the complete pipeline:
```bash
python run_pipeline.py
```

### 3. Run the Streamlit App

```bash
python -m streamlit run app.py
```

Open your browser to `http://localhost:8501` and enter a question.

## Usage

### Interactive Web Interface

```bash
python -m streamlit run app.py
```

**Features:**
- Enter natural language questions
- View retrieved chunks with scores
- Read AI-generated answers
- See response time metrics

### Using the RAG System Programmatically

```python
from rag.retriever import dense_retrieve, sparse_retrieve, rrf, corpus
from rag.generator import generate_answer

# Query
query = "What is machine learning?"

# Dense retrieval
dense_results = dense_retrieve(query, k=10)

# Sparse retrieval
sparse_results = sparse_retrieve(query, k=10)

# Fuse results
fused = rrf(dense_results, sparse_results)[:5]

# Get contexts
contexts = [corpus[i]["text"] for i, _ in fused]

# Generate answer
answer = generate_answer(query, contexts)
print(answer)
```

## Key Components

### Retriever (`rag/retriever.py`)
- **`dense_retrieve(query, k=10)`**: Uses sentence transformer embeddings + FAISS for semantic search
- **`sparse_retrieve(query, k=10)`**: Uses BM25 for keyword matching
- **`rrf(dense, sparse, k=60)`**: Reciprocal Rank Fusion combining both methods

### Generator (`rag/generator.py`)
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
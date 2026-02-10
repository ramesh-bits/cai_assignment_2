# Automated Evaluation Pipeline

Run the complete evaluation with a single command:

```bash
python evaluation_pipeline.py
```

## What It Does

The pipeline automates the entire evaluation process:

1. **Generate Questions** (100 Q&A pairs): Automatically generates diverse questions (factual, comparative, inferential, multi-hop) from the corpus
2. **Run Evaluation**: Tests the RAG system on all questions with retrieval and generation
3. **Compute Metrics**: Calculates all evaluation metrics (MRR, F1, ROUGE-L, Precision, Recall, Hit Rate, NDCG, etc.)
4. **Generate Report**: Creates comprehensive markdown report with analysis

## Output Files

All results are saved in `evaluation/results/`:

- **evaluation_detailed.csv** - Per-question results with all metrics
- **metrics.json** - All computed metric values
- **summary.txt** - Quick summary of key metrics
- **EVALUATION_REPORT.md** - Comprehensive markdown report with:
  - Executive summary
  - Mandatory MRR metric explanation
  - Custom metrics (F1 Token Overlap, ROUGE-L) with justification
  - Retrieval & generation metrics comparison
  - Error analysis by question type
  - Failed retrieval analysis

## Metrics Explained

### Mandatory Metric: Mean Reciprocal Rank (MRR)
- **What it measures**: How quickly the system finds the correct source document
- **Calculation**: Average of (1/rank) where rank is position of first correct URL
- **Range**: 0.0 (never found) to 1.0 (always first)
- **Interpretation**: Higher is better

### Custom Metric 1: F1 Token Overlap
- **What it measures**: Token-level overlap between generated and ground-truth answers
- **Why chosen**: Captures semantic content similarity, balances precision and recall
- **Calculation**: F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0
- **Interpretation**: Closer to 1.0 = better content alignment

### Custom Metric 2: ROUGE-L
- **What it measures**: Longest common subsequence overlap
- **Why chosen**: Industry-standard for NLG evaluation, more lenient than F1
- **Calculation**: ROUGE-L F1 score of longest common subsequence
- **Range**: 0.0 to 1.0
- **Interpretation**: Captures answer quality while allowing word reordering

### Additional Metrics

- **Precision@5**: Fraction of top-5 results that are relevant
- **Recall@5**: Fraction of relevant results found in top-5
- **Hit Rate@5**: Whether correct URL appears in top-5
- **NDCG@5**: Normalized discounted cumulative gain
- **Exact Match**: Exact string matching after normalization
- **Avg Response Time**: Average inference time per query

## Prerequisites

Before running, ensure:

1. **Corpus built**: `data/corpus_chunks.json` exists
   ```bash
   python indexing/build_corpus.py
   ```

2. **Indexes built**: Dense and sparse indexes exist
   ```bash
   python indexing/dense_index.py
   python indexing/sparse_index.py
   ```

3. **Dependencies installed**: 
   - sentence-transformers
   - faiss-cpu
   - rank-bm25
   - rouge-score

## Example Output

```
======================================================================
                 RAG SYSTEM AUTOMATED EVALUATION PIPELINE
======================================================================

======================================================================
STEP: Generate Evaluation Questions
======================================================================
Generating 100 Q&A pairs from corpus...
  Generated 100 questions
✅ Completed in 8.32s

======================================================================
STEP: Run Evaluation on All Questions
======================================================================
Running RAG system on all questions...
  [10/100] Evaluating...
  [20/100] Evaluating...
  ...
✅ Completed in 124.56s

======================================================================
FINAL RESULTS
======================================================================

📊 Evaluation Complete!
   Questions evaluated: 100

📈 Key Metrics:
   MRR:                  0.4523
   F1 Token Overlap:     0.6234
   ROUGE-L:              0.5891
   Avg Response Time:    1.246s

📁 Results saved to: evaluation/results/
   - evaluation_detailed.csv (detailed per-question results)
   - metrics.json (all metric values)
   - summary.txt (quick summary)
   - EVALUATION_REPORT.md (comprehensive report)
```

## Interpreting Results

### MRR of 0.45
- Correct URL is found in about 45% of cases
- When found, it appears around position 2-3 on average

### F1 Token Overlap of 0.62
- Generated answers share about 62% of tokens with ground truth
- Good semantic alignment between answers

### ROUGE-L of 0.59
- 59% longest common subsequence overlap
- Indicates reasonable answer quality

## Running Specific Steps

To run just question generation:
```bash
python evaluation/generate_questions.py
```

To run just evaluation:
```bash
python evaluation/evaluate.py
```

## Advanced Customization

Edit variables in `evaluation_pipeline.py`:
- `step_generate_questions()`: Change number of questions
- `step_run_evaluation()`: Adjust retrieval parameters
- `step_generate_report()`: Customize report format

## Troubleshooting

**"KeyError: data/questions.json not found"**
- Run question generation first or ensure data/questions.json exists

**"No relevant documents found"**
- Check that indexes are properly built
- Verify corpus_chunks.json has content

**Out of memory during generation**
- Reduce batch size in generator
- Use CPU-only mode

For more details, see the [main README](../README.md).

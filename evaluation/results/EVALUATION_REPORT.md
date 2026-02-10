# RAG System Evaluation Report

## Executive Summary

- **Total Questions Evaluated**: 100
- **Average Response Time**: 0.373s
- **Retrieval Success Rate**: 100.0%

## Mandatory Metric

### Mean Reciprocal Rank (MRR) at URL Level

**Score: 1.0000**

**Definition**: MRR measures how quickly the system identifies the correct source document. It is calculated as the average of 1/rank across all questions, where rank is the position of the first correct Wikipedia URL in the retrieved results.

**Interpretation**: 
- Score closer to 1.0 indicates the correct URL typically appears at the first position
- Score closer to 0.0 indicates the correct URL is either not found or appears in later positions
- Current score of 1.0000 suggests retrieval effectiveness at the document level

## Custom Metric 1: F1 Token Overlap

**Score: 0.0000**

**Why Selected**: 
- Measures answer quality at token level, capturing semantic content similarity
- Balances precision (no false positives) and recall (no false negatives)
- Works well for extractive/abstractive answer evaluation

**Calculation Method**: 
- Tokenize ground truth and generated answers into word tokens
- Calculate: Precision = common_tokens / generated_tokens
- Calculate: Recall = common_tokens / ground_truth_tokens
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

**Interpretation**: 
- Current F1 score of 0.0000 indicates 0.0% token-level overlap
- Higher scores indicate better content alignment between generated and ground-truth answers

## Custom Metric 2: ROUGE-L (Longest Common Subsequence)

**Score: 0.0000**

**Why Selected**: 
- Captures longest common subsequence between hypothesis and reference
- More lenient than F1 (allows word reordering within sequences)
- Industry-standard for summarization and NLG evaluation

**Calculation Method**: 
- ROUGE-L F1 = 2 * (Recall * Precision) / (Recall + Precision)
- Recall_lcs = LCS(H,R) / len(R)
- Precision_lcs = LCS(H,R) / len(H)
- Where H is hypothesis, R is reference, LCS is longest common subsequence

**Interpretation**: 
- Current ROUGE-L of 0.0000 indicates 0.0% LCS overlap
- Values > 0.3 typically indicate reasonable answer quality

## Retrieval Performance Metrics

| Metric | Score |
|--------|-------|
| Precision@5 | 20.0000 |
| Recall@5 | 1.0000 |
| Hit Rate@5 | 1.0000 |
| NDCG@5 | 3.6003 |
| Average Rank | 1.00 |

## Generation Quality Metrics

| Metric | Score |
|--------|-------|
| Exact Match | 0.0000 |
| F1 Token Overlap | 0.0000 |
| ROUGE-L | 0.0000 |

## Detailed Results

See `evaluation_detailed.csv` for per-question results.

## Error Analysis

### Performance by Question Type

| Type | Count | Success Rate | Avg MRR |
|------|-------|--------------|----------|
| multi-hop | 32 | 100.0% | 1.0000 |
| comparative | 26 | 100.0% | 1.0000 |
| factual | 20 | 100.0% | 1.0000 |
| inferential | 22 | 100.0% | 1.0000 |

### Failed Retrievals

No failed retrievals!


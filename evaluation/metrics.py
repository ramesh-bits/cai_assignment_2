"""Evaluation metrics for RAG system."""

import re
import math
from collections import defaultdict
from rouge_score import rouge_scorer


def mean_reciprocal_rank(results):
    """Mean Reciprocal Rank at URL level.
    
    MRR = (1/N) * Σ(1/rank_i) where rank is the position of correct URL
    """
    rr = []
    for r in results:
        if r["rank"] is None:
            rr.append(0)
        else:
            rr.append(1 / r["rank"])
    return sum(rr) / len(rr) if rr else 0


def precision_at_k(results, k):
    return sum(
        1 for r in results if r["rank"] is not None and r["rank"] <= k
    ) / len(results) if results else 0


def recall_at_k(results, k):
    return sum(
        1 for r in results if r["rank"] is not None and r["rank"] <= k
    ) / len(results) if results else 0


def ndcg_at_k(results, k):
    scores = []
    for r in results:
        if r["rank"] is not None and r["rank"] <= k:
            scores.append(1 / math.log2(r["rank"] + 1))
        else:
            scores.append(0)
    return sum(scores) / len(scores) if scores else 0


def rouge_l_similarity(generated_answers, ground_truths):
    """ROUGE-L F1 score: measure overlap between generated and ground truth.
    
    ROUGE-L captures longest common subsequence between hypothesis and reference.
    Returns average F1 score across all Q&A pairs.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    
    for gen, gt in zip(generated_answers, ground_truths):
        if not gen or not gt:
            scores.append(0)
            continue
        score = scorer.score(gt, gen)["rougeL"].fmeasure
        scores.append(score)
    
    return sum(scores) / len(scores) if scores else 0


def exact_match(generated_answers, ground_truths):
    """Exact Match (EM): strict string matching after normalization.
    
    EM = (# exact matches) / (total)
    Normalized by lowercasing and removing extra spaces.
    """
    matches = 0
    for gen, gt in zip(generated_answers, ground_truths):
        gen_norm = re.sub(r'\s+', ' ', gen.lower().strip())
        gt_norm = re.sub(r'\s+', ' ', gt.lower().strip())
        if gen_norm == gt_norm:
            matches += 1
    
    return matches / len(generated_answers) if generated_answers else 0


def f1_token_overlap(generated_answers, ground_truths):
    """F1 Token Overlap: token-level precision and recall.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    where Precision = common tokens / generated tokens
    and   Recall = common tokens / ground truth tokens
    """
    f1_scores = []
    
    for gen, gt in zip(generated_answers, ground_truths):
        gen_tokens = set(re.findall(r'\w+', gen.lower()))
        gt_tokens = set(re.findall(r'\w+', gt.lower()))
        
        if not gen_tokens or not gt_tokens:
            f1_scores.append(0)
            continue
        
        common = gen_tokens & gt_tokens
        precision = len(common) / len(gen_tokens)
        recall = len(common) / len(gt_tokens)
        
        if precision + recall == 0:
            f1_scores.append(0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0


def hit_rate(results, k):
    """Hit Rate@K: whether correct URL appears in top-K results.
    
    Hit@K = (# questions with correct URL in top-K) / (total questions)
    """
    hits = sum(1 for r in results if r["rank"] is not None and r["rank"] <= k)
    return hits / len(results) if results else 0


def avg_rank(results):
    """Average Rank: mean position of correct URL (if found).
    
    Only counts questions where URL was found.
    """
    ranks = [r["rank"] for r in results if r["rank"] is not None]
    return sum(ranks) / len(ranks) if ranks else float('inf')


def compute_all_metrics(evaluation_results):
    """Compute all metrics for a set of evaluation results.
    
    Args:
        evaluation_results: list of dicts with keys:
            - rank: URL rank (int or None)
            - generated_answer: generated answer text
            - ground_truth: ground truth answer text
    
    Returns:
        dict with all computed metrics
    """
    metrics = {
        "mrr": mean_reciprocal_rank(evaluation_results),
        "precision@5": precision_at_k(evaluation_results, 5),
        "recall@5": recall_at_k(evaluation_results, 5),
        "hit_rate@5": hit_rate(evaluation_results, 5),
        "avg_rank": avg_rank(evaluation_results),
        "ndcg@5": ndcg_at_k(evaluation_results, 5),
    }
    
    generated = [r.get("generated_answer", "") for r in evaluation_results]
    ground_truth = [r.get("ground_truth", "") for r in evaluation_results]
    
    metrics["rouge_l"] = rouge_l_similarity(generated, ground_truth)
    metrics["exact_match"] = exact_match(generated, ground_truth)
    metrics["f1_token_overlap"] = f1_token_overlap(generated, ground_truth)
    
    return metrics

"""Evaluation script for RAG system."""

import os
import sys
import json
import time
import csv

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.retriever import get_fused_for_query, corpus
from rag.generator import generate_answer
from evaluation.metrics import compute_all_metrics


def load_questions(filepath="data/questions.json"):
    """Load evaluation questions from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def url_rank(fused_docs, target_urls):
    """Find rank of first correct URL in retrieved documents.
    
    Args:
        fused_docs: list of (doc_id, score) tuples from retrieval
        target_urls: list of correct URL strings
    
    Returns:
        int: 1-based rank if found, None otherwise
    """
    ranked_urls = [corpus[i]["url"] for i, _ in fused_docs]
    for idx, url in enumerate(ranked_urls):
        if url in target_urls:
            return idx + 1
    return None


def run_evaluation(questions, output_dir="evaluation/results", verbose=True):
    """Run full RAG evaluation on question set.
    
    Args:
        questions: list of question dicts with keys: question, answer, source_urls
        output_dir: directory to store results
        verbose: print progress
    
    Returns:
        tuple: (evaluation_results, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    evaluation_results = []
    
    for idx, q in enumerate(questions):
        if verbose and (idx + 1) % 10 == 0:
            print(f"  [{idx + 1}/{len(questions)}] Evaluating...")
        
        query = q["question"]
        ground_truth = q["answer"]
        source_urls = q.get("source_urls", [])
        
        start_time = time.time()
        
        # Retrieve
        fused = get_fused_for_query(query, top_n=10)
        
        # Generate answer
        if fused:
            contexts = [
                f"Title: {corpus[i].get('title','')}\nURL: {corpus[i].get('url','')}\nText: {corpus[i]['text']}"
                for i, _ in fused
            ]
            contexts_for_generation = contexts[:3]
            generated_answer = generate_answer(query, contexts_for_generation)
        else:
            generated_answer = "I don't know."
        
        elapsed = time.time() - start_time
        
        # Calculate rank
        rank = url_rank(fused, source_urls)
        
        result = {
            "q_id": q.get("id", idx),
            "question": query,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "source_urls": source_urls,
            "rank": rank,
            "response_time": elapsed,
            "question_type": q.get("question_type", "unknown"),
            "num_retrieved": len(fused)
        }
        evaluation_results.append(result)
    
    # Compute metrics
    metrics = compute_all_metrics(evaluation_results)
    
    # Add derived metrics
    metrics["avg_response_time"] = sum(r["response_time"] for r in evaluation_results) / len(evaluation_results)
    metrics["retrieval_success_rate"] = sum(1 for r in evaluation_results if r["num_retrieved"] > 0) / len(evaluation_results)
    
    return evaluation_results, metrics


def save_results(evaluation_results, metrics, output_dir="evaluation/results"):
    """Save evaluation results to CSV and JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results as CSV
    csv_path = os.path.join(output_dir, "evaluation_detailed.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=evaluation_results[0].keys())
        writer.writeheader()
        writer.writerows(evaluation_results)
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RAG SYSTEM EVALUATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("MANDATORY METRIC:\n")
        f.write(f"  MRR (Mean Reciprocal Rank):  {metrics['mrr']:.4f}\n\n")
        
        f.write("CUSTOM METRICS:\n")
        f.write(f"  F1 Token Overlap:            {metrics['f1_token_overlap']:.4f}\n")
        f.write(f"  ROUGE-L Score:               {metrics['rouge_l']:.4f}\n\n")
        
        f.write("ADDITIONAL RETRIEVAL METRICS:\n")
        f.write(f"  Precision@5:                 {metrics['precision@5']:.4f}\n")
        f.write(f"  Recall@5:                    {metrics['recall@5']:.4f}\n")
        f.write(f"  Hit Rate@5:                  {metrics['hit_rate@5']:.4f}\n")
        f.write(f"  NDCG@5:                      {metrics['ndcg@5']:.4f}\n")
        f.write(f"  Average Rank:                {metrics['avg_rank']:.2f}\n\n")
        
        f.write("GENERATION METRICS:\n")
        f.write(f"  Exact Match:                 {metrics['exact_match']:.4f}\n\n")
        
        f.write("EFFICIENCY METRICS:\n")
        f.write(f"  Avg Response Time (s):       {metrics['avg_response_time']:.3f}\n")
        f.write(f"  Retrieval Success Rate:      {metrics['retrieval_success_rate']:.4f}\n")
    
    return csv_path, metrics_path, summary_path


if __name__ == "__main__":
    print("Loading questions...")
    questions = load_questions()
    print(f"Loaded {len(questions)} questions\n")
    
    print("Running evaluation...")
    results, metrics = run_evaluation(questions)
    
    print("\nSaving results...")
    csv_p, metrics_p, summary_p = save_results(results, metrics)
    
    print(f"\n✅ Results saved to:")
    print(f"  - CSV: {csv_p}")
    print(f"  - Metrics: {metrics_p}")
    print(f"  - Summary: {summary_p}")
    
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"MRR:                  {metrics['mrr']:.4f}")
    print(f"F1 Token Overlap:     {metrics['f1_token_overlap']:.4f}")
    print(f"ROUGE-L:              {metrics['rouge_l']:.4f}")
    print(f"Avg Response Time:    {metrics['avg_response_time']:.3f}s")

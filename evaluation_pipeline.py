#!/usr/bin/env python3
"""Automated RAG Evaluation Pipeline - Run with: python evaluation_pipeline.py"""

import os
import sys
import json
import time
import subprocess

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_step(name, func, *args):
    """Run a step and measure time."""
    print(f"\n{'='*70}")
    print(f"STEP: {name}")
    print('='*70)
    start = time.time()
    result = func(*args)
    elapsed = time.time() - start
    print(f"✅ Completed in {elapsed:.2f}s")
    return result


def step_generate_questions():
    """Step 1: Generate questions from corpus."""
    print("Generating 100 Q&A pairs from corpus...")
    
    import json
    import random
    from evaluation.generate_questions import generate_question, QUESTION_TYPES
    
    corpus = json.load(open("data/corpus_chunks.json"))
    
    questions = []
    used_chunks = set()
    
    while len(questions) < 100:
        chunk = random.choice(corpus)
        
        if chunk["chunk_id"] in used_chunks:
            continue
        
        used_chunks.add(chunk["chunk_id"])
        qtype = random.choice(QUESTION_TYPES)
        
        try:
            question, answer = generate_question(chunk["text"], qtype)
            if not question or not answer:
                continue
            
            questions.append({
                "id": len(questions),
                "question": question,
                "answer": answer,
                "source_urls": [chunk["url"]],
                "question_type": qtype
            })
        except Exception as e:
            continue
    
    # Save questions
    os.makedirs("data", exist_ok=True)
    with open("data/questions.json", "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"  Generated {len(questions)} questions")
    return len(questions)


def step_run_evaluation():
    """Step 2: Run evaluation on all questions."""
    print("Running RAG system on all questions...")
    
    from evaluation.evaluate import load_questions, run_evaluation, save_results
    
    questions = load_questions()
    results, metrics = run_evaluation(questions, verbose=True)
    save_results(results, metrics)
    
    return results, metrics


def step_generate_report(results, metrics):
    """Step 3: Generate evaluation report."""
    print("Generating evaluation report...")
    
    report_path = "evaluation/results/EVALUATION_REPORT.md"
    os.makedirs("evaluation/results", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# RAG System Evaluation Report\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Questions Evaluated**: {len(results)}\n")
        f.write(f"- **Average Response Time**: {metrics['avg_response_time']:.3f}s\n")
        f.write(f"- **Retrieval Success Rate**: {metrics['retrieval_success_rate']*100:.1f}%\n\n")
        
        # Mandatory Metric
        f.write("## Mandatory Metric\n\n")
        f.write("### Mean Reciprocal Rank (MRR) at URL Level\n\n")
        f.write(f"**Score: {metrics['mrr']:.4f}**\n\n")
        f.write("**Definition**: MRR measures how quickly the system identifies the correct source document. ")
        f.write("It is calculated as the average of 1/rank across all questions, where rank is the position ")
        f.write("of the first correct Wikipedia URL in the retrieved results.\n\n")
        f.write("**Interpretation**: \n")
        f.write("- Score closer to 1.0 indicates the correct URL typically appears at the first position\n")
        f.write("- Score closer to 0.0 indicates the correct URL is either not found or appears in later positions\n")
        f.write(f"- Current score of {metrics['mrr']:.4f} suggests retrieval effectiveness at the document level\n\n")
        
        # Custom Metric 1: F1 Token Overlap
        f.write("## Custom Metric 1: F1 Token Overlap\n\n")
        f.write(f"**Score: {metrics['f1_token_overlap']:.4f}**\n\n")
        f.write("**Why Selected**: \n")
        f.write("- Measures answer quality at token level, capturing semantic content similarity\n")
        f.write("- Balances precision (no false positives) and recall (no false negatives)\n")
        f.write("- Works well for extractive/abstractive answer evaluation\n\n")
        f.write("**Calculation Method**: \n")
        f.write("- Tokenize ground truth and generated answers into word tokens\n")
        f.write("- Calculate: Precision = common_tokens / generated_tokens\n")
        f.write("- Calculate: Recall = common_tokens / ground_truth_tokens\n")
        f.write("- F1 = 2 * (Precision * Recall) / (Precision + Recall)\n\n")
        f.write("**Interpretation**: \n")
        f.write(f"- Current F1 score of {metrics['f1_token_overlap']:.4f} indicates {metrics['f1_token_overlap']*100:.1f}% token-level overlap\n")
        f.write("- Higher scores indicate better content alignment between generated and ground-truth answers\n\n")
        
        # Custom Metric 2: ROUGE-L
        f.write("## Custom Metric 2: ROUGE-L (Longest Common Subsequence)\n\n")
        f.write(f"**Score: {metrics['rouge_l']:.4f}**\n\n")
        f.write("**Why Selected**: \n")
        f.write("- Captures longest common subsequence between hypothesis and reference\n")
        f.write("- More lenient than F1 (allows word reordering within sequences)\n")
        f.write("- Industry-standard for summarization and NLG evaluation\n\n")
        f.write("**Calculation Method**: \n")
        f.write("- ROUGE-L F1 = 2 * (Recall * Precision) / (Recall + Precision)\n")
        f.write("- Recall_lcs = LCS(H,R) / len(R)\n")
        f.write("- Precision_lcs = LCS(H,R) / len(H)\n")
        f.write("- Where H is hypothesis, R is reference, LCS is longest common subsequence\n\n")
        f.write("**Interpretation**: \n")
        f.write(f"- Current ROUGE-L of {metrics['rouge_l']:.4f} indicates {metrics['rouge_l']*100:.1f}% LCS overlap\n")
        f.write("- Values > 0.3 typically indicate reasonable answer quality\n\n")
        
        # Additional Metrics
        f.write("## Retrieval Performance Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Precision@5 | {metrics['precision@5']:.4f} |\n")
        f.write(f"| Recall@5 | {metrics['recall@5']:.4f} |\n")
        f.write(f"| Hit Rate@5 | {metrics['hit_rate@5']:.4f} |\n")
        f.write(f"| NDCG@5 | {metrics['ndcg@5']:.4f} |\n")
        f.write(f"| Average Rank | {metrics['avg_rank']:.2f} |\n\n")
        
        # Generation Quality
        f.write("## Generation Quality Metrics\n\n")
        f.write("| Metric | Score |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Exact Match | {metrics['exact_match']:.4f} |\n")
        f.write(f"| F1 Token Overlap | {metrics['f1_token_overlap']:.4f} |\n")
        f.write(f"| ROUGE-L | {metrics['rouge_l']:.4f} |\n\n")
        
        # Result Details
        f.write("## Detailed Results\n\n")
        f.write("See `evaluation_detailed.csv` for per-question results.\n\n")
        
        # Error Analysis
        f.write("## Error Analysis\n\n")
        
        by_type = {}
        for r in results:
            qtype = r.get("question_type", "unknown")
            if qtype not in by_type:
                by_type[qtype] = {"count": 0, "mrr_sum": 0, "found": 0}
            by_type[qtype]["count"] += 1
            by_type[qtype]["mrr_sum"] += (1 / r["rank"]) if r["rank"] else 0
            if r["rank"] is not None:
                by_type[qtype]["found"] += 1
        
        f.write("### Performance by Question Type\n\n")
        f.write("| Type | Count | Success Rate | Avg MRR |\n")
        f.write("|------|-------|--------------|----------|\n")
        for qtype, stats in by_type.items():
            success_rate = stats["found"] / stats["count"] if stats["count"] > 0 else 0
            avg_mrr = stats["mrr_sum"] / stats["count"] if stats["count"] > 0 else 0
            f.write(f"| {qtype} | {stats['count']} | {success_rate:.1%} | {avg_mrr:.4f} |\n")
        
        f.write("\n")
        f.write("### Failed Retrievals\n\n")
        failed = [r for r in results if r["rank"] is None]
        if failed:
            f.write(f"Total: {len(failed)} questions\n\n")
            for r in failed[:5]:
                f.write(f"- Q: {r['question'][:80]}...\n")
        else:
            f.write("No failed retrievals!\n\n")
    
    print(f"  Report saved to {report_path}")
    return report_path


def main():
    """Run the complete evaluation pipeline."""
    print("\n" + "="*70)
    print(" "*15 + "RAG SYSTEM AUTOMATED EVALUATION PIPELINE")
    print("="*70)
    
    try:
        # Check prerequisites
        if not os.path.exists("data/corpus_chunks.json"):
            print("❌ Error: data/corpus_chunks.json not found!")
            print("   Please run: python indexing/build_corpus.py")
            sys.exit(1)
        
        # Step 1: Generate questions
        num_questions = run_step(
            "Generate Evaluation Questions",
            step_generate_questions
        )
        
        # Step 2: Run evaluation
        results, metrics = run_step(
            "Run Evaluation on All Questions",
            step_run_evaluation
        )
        
        # Step 3: Generate report
        report_path = run_step(
            "Generate Evaluation Report",
            step_generate_report,
            results,
            metrics
        )
        
        # Final Summary
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        print(f"\n📊 Evaluation Complete!")
        print(f"   Questions evaluated: {len(results)}")
        print(f"\n📈 Key Metrics:")
        print(f"   MRR:                  {metrics['mrr']:.4f}")
        print(f"   F1 Token Overlap:     {metrics['f1_token_overlap']:.4f}")
        print(f"   ROUGE-L:              {metrics['rouge_l']:.4f}")
        print(f"   Avg Response Time:    {metrics['avg_response_time']:.3f}s")
        print(f"\n📁 Results saved to: evaluation/results/")
        print(f"   - evaluation_detailed.csv (detailed per-question results)")
        print(f"   - metrics.json (all metric values)")
        print(f"   - summary.txt (quick summary)")
        print(f"   - EVALUATION_REPORT.md (comprehensive report)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

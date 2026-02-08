from rouge_score import rouge_scorer

def mean_reciprocal_rank(results):
    """
    results: list of dicts
    each dict must have: 'rank' (int or None)
    """
    rr = []
    for r in results:
        if r["rank"] is None:
            rr.append(0)
        else:
            rr.append(1 / r["rank"])
    return sum(rr) / len(rr)


def recall_at_k(results, k):
    """
    results: list of dicts
    each dict must have: 'rank'
    """
    hits = 0
    for r in results:
        if r["rank"] is not None and r["rank"] <= k:
            hits += 1
    return hits / len(results)


def rouge_l_score(generated_answers, ground_truths):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []

    for gen, gt in zip(generated_answers, ground_truths):
        score = scorer.score(gt, gen)["rougeL"].fmeasure
        scores.append(score)

    return sum(scores) / len(scores)

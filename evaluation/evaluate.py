import json
from rouge_score import rouge_scorer
from retriever import dense_retrieve, sparse_retrieve, rrf, corpus

questions = json.load(open("questions.json"))
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def url_rank(fused, urls):
    ranked_urls = [corpus[i]["url"] for i,_ in fused]
    for i,u in enumerate(ranked_urls):
        if u in urls:
            return i+1
    return None

mrr = []
rouge_scores = []

for q in questions:
    fused = rrf(dense_retrieve(q["question"]),
                sparse_retrieve(q["question"]))

    rank = url_rank(fused, q["source_urls"])
    mrr.append(1/rank if rank else 0)

    contexts = [corpus[i]["text"] for i,_ in fused[:5]]
    answer = " ".join(contexts)
    rouge_scores.append(
        scorer.score(q["answer"], answer)["rougeL"].fmeasure
    )

print("MRR:", sum(mrr)/len(mrr))
print("Avg ROUGE-L:", sum(rouge_scores)/len(rouge_scores))
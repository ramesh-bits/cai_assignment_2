import json
import random
from transformers import pipeline

# Load corpus
corpus = json.load(open("data/corpus_chunks.json"))

# LLM for question generation
qgen = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=128
)

QUESTION_TYPES = [
    "factual",
    "comparative",
    "inferential",
    "multi-hop"
]

def generate_question(context, qtype):
    prompt = f"""
Generate a {qtype} question and its answer based only on the context below.

Context:
{context}

Format:
Question: ...
Answer: ...
"""
    output = qgen(prompt)[0]["generated_text"]

    if "Answer:" not in output:
        return None, None

    q, a = output.split("Answer:", 1)
    return q.replace("Question:", "").strip(), a.strip()

questions = []
used_chunks = set()

while len(questions) < 100:
    chunk = random.choice(corpus)

    if chunk["chunk_id"] in used_chunks:
        continue

    used_chunks.add(chunk["chunk_id"])
    qtype = random.choice(QUESTION_TYPES)

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

with open("data/questions.json", "w") as f:
    json.dump(questions, f, indent=2)

print("Generated", len(questions), "questions")
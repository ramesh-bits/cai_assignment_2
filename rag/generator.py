from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def generate_answer(query, contexts):
    prompt = f"""
Context:
{' '.join(contexts)}

Question: {query}
Answer:
"""
    return generator(prompt)[0]["generated_text"]
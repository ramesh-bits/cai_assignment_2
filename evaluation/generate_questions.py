import os
import json
import random
import re
# transformers is optional; only used when LLM generation is enabled
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    pipeline = None
    _HAS_TRANSFORMERS = False

# Load corpus
corpus = json.load(open("data/corpus_chunks.json"))

# LLM for question generation (deterministic) - created lazily if available
qgen = None
if _HAS_TRANSFORMERS:
    try:
        qgen = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            do_sample=False,
            num_return_sequences=1,
            max_length=128,
        )
    except Exception:
        qgen = None

QUESTION_TYPES = [
    "factual",
    "comparative",
    "inferential",
    "multi-hop",
]


def _fallback_qa_from_text(text, qtype="factual"):
    """Create a Q/A from a context as a fallback, tailored to `qtype`.

    qtype: one of 'factual','comparative','inferential','multi-hop'
    Uses lightweight regex heuristics to extract subjects/entities and
    build simple question templates with short extractive answers.
    """
    sents = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
    if not sents:
        return None, None
    # prefer the longest sentence as the main extractive answer candidate
    sent = max(sents, key=len)

    # simple entity/subject extraction (capitalized phrases)
    entities = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", text)
    entities = [e for e in entities if len(e) > 1]
    subject = entities[0] if entities else None

    if qtype == "factual":
        if subject:
            question = f"What is {subject}?"
            answer = next((s for s in sents if subject in s), sent)
            return question, answer
        return "What is the main point of the context?", sent

    if qtype == "comparative":
        if len(entities) >= 2:
            a, b = entities[0], entities[1]
            question = f"How does {a} compare to {b}?"
            ans_a = next((s for s in sents if a in s), sent)
            ans_b = next((s for s in sents if b in s and s != ans_a), sent)
            answer = f"{a}: {ans_a}. {b}: {ans_b}."
            return question, answer
        if subject:
            return f"How does {subject} compare to related concepts?", sent
        return "Compare the main concept in the context to related concepts.", sent

    if qtype == "inferential":
        # look for causal / inferential cues
        for s in sents:
            if re.search(r"\b(because|due to|caused by|leads to|results in|therefore)\b", s, flags=re.I):
                if subject:
                    return f"Why is {subject} described this way?", s
                return "Why does the context present this information?", s
        if subject:
            return f"Why is {subject} important or significant?", sent
        return "Why does the context present this information?", sent

    if qtype == "multi-hop":
        if len(sents) >= 2:
            s1, s2 = sents[0], sents[1]
            ents1 = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", s1)
            ents2 = re.findall(r"\b([A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)*)\b", s2)
            if ents1 and ents2:
                a, b = ents1[0], ents2[0]
                return f"How is {a} related to {b}?", f"{s1} {s2}"
        return "How are the concepts in the context related?", sent

    # generic fallback
    return "Based on the context, summarize the main point.", sent


def _generate_with_llm(context, qtype):
    # If qgen isn't available (transformers missing or failed to load), skip LLM.
    if qgen is None:
        return None, None

    prompt = (
        f"Context: {context}\n\n"
        f"Generate a {qtype} question and answer grounded in the context only.\n"
        "Output format:\nQuestion: [your question]\nAnswer: [your answer]\n"
    )
    try:
        out = qgen(prompt, max_length=128)[0]["generated_text"]
    except Exception:
        return None, None

    # Try to parse model output
    m = re.search(r"Question\s*:\s*(.*?)\nAnswer\s*:\s*(.*)", out, flags=re.S)
    if m:
        q = m.group(1).strip()
        a = m.group(2).strip()
        # Reject if empty, contains placeholders (...), or template markers ([...])
        if q and a and "..." not in q and "..." not in a and "[" not in q and "[" not in a:
            return q, a
    # fallback if parsing fails
    return None, None


def generate_question(context, qtype="factual"):
    """Generate a single (question, answer) for given context and qtype.

    Uses LLM when available; otherwise falls back to extractive templates.
    """
    if os.getenv("FORCE_EXTRACTIVE", "0") in ("1", "true", "True"):
        return _fallback_qa_from_text(context, qtype)

    q, a = _generate_with_llm(context, qtype)
    if q and a:
        return q, a
    # fallback to extractive qtype-aware generator
    return _fallback_qa_from_text(context, qtype)


def generate_questions(n=100):
    """Generate `n` QA pairs from the corpus and return as a list of dicts."""
    ids = list(range(len(corpus)))
    if n <= len(ids):
        picks = random.sample(ids, n)
    else:
        picks = random.choices(ids, k=n)

    qs = []
    for i, idx in enumerate(picks):
        chunk = corpus[idx]
        context = chunk.get("text", "")
        qtype = random.choice(QUESTION_TYPES)
        q, a = generate_question(context, qtype)
        if not q or not a:
            continue
        qs.append(
            {
                "id": i,
                "qtype": qtype,
                "question": q,
                "answer": a,
                "chunk_id": chunk.get("chunk_id"),
                "url": chunk.get("url"),
                "title": chunk.get("title"),
            }
        )
    return qs


if __name__ == "__main__":
    n = int(os.getenv("NUM_QUESTIONS", "100"))
    print(f"Generating {n} questions...")
    qs = generate_questions(n)
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "questions.json")
    with open(out_path, "w") as f:
        json.dump(qs, f, indent=2)
    print(f"Wrote {len(qs)} questions to {out_path}")

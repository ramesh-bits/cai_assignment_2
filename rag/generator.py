import re

# Lightweight fallback: extract answers from context rather than using a heavy LLM
# This avoids segfaults and memory issues during terminal testing


def generate_answer(query, contexts):
    """Generate an answer grounded in the provided contexts.
    
    Uses a lightweight extractive approach: pulls relevant sentences from context
    rather than loading a heavy transformer model.
    """
    if not contexts:
        return "I don't know. No contexts provided."
    
    # Quick check: if none of the (meaningful) query tokens appear in any context,
    # return early to prevent hallucination.
    query_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 3]
    if query_tokens:
        found = False
        for c in contexts:
            lc = c.lower()
            if any(tok in lc for tok in query_tokens):
                found = True
                break
        if not found:
            return "I don't know. No relevant information found in the retrieved contexts."
    
    # Extract the main text from contexts (skip title/URL lines)
    context_texts = []
    for ctx in contexts:
        # Split by lines and find the 'Text:' section
        lines = ctx.split('\n')
        text_started = False
        text_lines = []
        for line in lines:
            if line.startswith('Text:'):
                text_started = True
                text_lines.append(line.replace('Text:', '').strip())
            elif text_started and line.strip():
                text_lines.append(line.strip())
        if text_lines:
            context_texts.append(' '.join(text_lines))
    
    if not context_texts:
        return "I don't know. Could not parse context text."
    
    combined_text = ' '.join(context_texts)
    
    # Simple extractive approach: find sentences that match query keywords
    sentences = re.split(r'[.!?]+', combined_text)
    
    # Score sentences based on query term overlap
    scored_sentences = []
    for sent in sentences:
        sent_lower = sent.lower().strip()
        if not sent_lower:
            continue
        matches = sum(1 for tok in query_tokens if tok in sent_lower)
        if matches > 0:
            scored_sentences.append((matches, sent.strip()))
    
    if not scored_sentences:
        return "I don't know. Could not find relevant sentences in the contexts."
    
    # Sort by match count and take the top sentences
    scored_sentences.sort(reverse=True)
    top_sentences = [s for _, s in scored_sentences[:3]]
    
    answer = ' '.join(top_sentences)
    return answer if answer.strip() else "I don't know."

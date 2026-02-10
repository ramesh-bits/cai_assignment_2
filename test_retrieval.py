#!/usr/bin/env python3
"""Simple terminal test for retrieval only (no generation)."""

from rag.retriever import get_fused_for_query, corpus

# Test queries
test_queries = [
    "What is artificial intelligence?",
    "What is salesforce?",
    "Tell me about machine learning",
]

for query in test_queries:
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print('='*70)
    
    fused = get_fused_for_query(query, top_n=5)
    
    if not fused:
        print("❌ NO RELEVANT DOCUMENTS FOUND")
    else:
        print(f"✅ RETRIEVED {len(fused)} DOCUMENT(S):\n")
        for idx, (doc_id, score) in enumerate(fused, 1):
            title = corpus[doc_id].get("title", "N/A")
            url = corpus[doc_id].get("url", "N/A")
            text_preview = corpus[doc_id]["text"][:200] + "..." if len(corpus[doc_id]["text"]) > 200 else corpus[doc_id]["text"]
            
            print(f"#{idx} - Score: {score:.4f}")
            print(f"  Title: {title}")
            print(f"  URL: {url}")
            print(f"  Text: {text_preview}\n")

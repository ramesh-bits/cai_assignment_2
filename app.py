#!/usr/bin/env python3
"""Streamlit RAG system."""

import streamlit as st
from rag.retriever import get_fused_for_query, corpus
from rag.generator import generate_answer
import time

st.set_page_config(page_title="Hybrid RAG", layout="wide")
st.title("🤖 Hybrid RAG System")

# User input
query = st.text_input("Enter your question:")

if query:
    start = time.time()
    
    # Retrieve
    fused = get_fused_for_query(query, top_n=5)
    
    if not fused:
        st.warning("❌ No relevant documents found in the corpus.")
        answer = "I don't know. No relevant information found in the corpus."
    else:
        # Show retrieved chunks
        st.subheader(f"📄 Retrieved Documents ({len(fused)})")
        for idx, (doc_id, score) in enumerate(fused, 1):
            title = corpus[doc_id].get("title", "N/A")
            url = corpus[doc_id].get("url", "N/A")
            with st.expander(f"[{idx}] {title} (Score: {score:.4f})"):
                st.write(f"**URL:** {url}")
                st.write(corpus[doc_id]["text"][:500] + "..." if len(corpus[doc_id]["text"]) > 500 else corpus[doc_id]["text"])
        
        # Generate answer from contexts
        contexts = [f"Title: {corpus[i].get('title','')}\nURL: {corpus[i].get('url','')}\nText: {corpus[i]['text']}" for i,_ in fused]
        contexts_for_generation = contexts[:3]
        answer = generate_answer(query, contexts_for_generation)
    
    # Display answer
    st.subheader("💡 Answer")
    st.write(answer)
    
    # Display timing
    elapsed = time.time() - start
    st.caption(f"Response time: {elapsed:.2f}s")

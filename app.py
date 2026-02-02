import streamlit as st
import time
from retriever import dense_retrieve, sparse_retrieve, rrf, corpus
from generator import generate_answer

st.title("Hybrid RAG System")

query = st.text_input("Enter your question")

if query:
    start = time.time()

    dense = dense_retrieve(query)
    sparse = sparse_retrieve(query)
    fused = rrf(dense, sparse)[:5]

    contexts = [corpus[i]["text"] for i,_ in fused]
    answer = generate_answer(query, contexts)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved Chunks")
    for i,score in fused:
        with st.expander(corpus[i]["title"]):
            st.write(corpus[i]["text"])
            st.write("Source:", corpus[i]["url"])
            st.write("RRF Score:", score)

    st.write("Response Time:", round(time.time()-start,2), "seconds")
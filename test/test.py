from rag.retriever import dense_retrieve, sparse_retrieve, rrf, corpus, get_fused_for_query
from rag.generator import generate_answer
# contexts = [f"Title: {corpus[i].get('title','')}\nURL: {corpus[i].get('url','')}\nText: {corpus[i]['text']}" for i,_ in fused]
query = "What is the capital of France?"
fused = get_fused_for_query(query, top_n=5)
contexts = [f"Title: {corpus[i].get('title','')}\nURL: {corpus[i].get('url','')}\nText: {corpus[i]['text']}" for i,_ in fused]
contexts_for_generation = contexts[:3]
answer = generate_answer(query, contexts_for_generation)
print(answer)
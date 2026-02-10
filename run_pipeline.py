import os

# Step 1: Generate Data - urls, build corpus, and create indexes
os.system("python indexing/generate_urls.py")
os.system("python indexing/build_corpus.py")
os.system("python indexing/dense_index.py")
os.system("python indexing/sparse_index.py")

# Step 2: Automated evaluation pipeline - generate questions, run evaluation, and compute metrics
os.system("python evaluation/evaluation_pipeline.py")
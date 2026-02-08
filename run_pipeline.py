import os

os.system("python indexing/generate_urls.py")
os.system("python indexing/build_corpus.py")
os.system("python indexing/dense_index.py")
os.system("python indexing/sparse_index.py")

os.system("python evaluation/generate_questions.py")
os.system("python evaluation/metrics.py")
os.system("python evaluation/evaluate.py")
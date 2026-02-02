import os

os.system("python build_corpus.py")
os.system("python dense_index.py")
os.system("python sparse_index.py")
os.system("python evaluate.py")
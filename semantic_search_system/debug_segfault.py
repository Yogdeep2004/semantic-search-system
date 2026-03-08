import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("1. Testing SentenceTransformer load...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("   Success")

print("2. Testing embedding generation...")
emb = model.encode(["hello world"])
print(f"   Success: {emb.shape}")

print("3. Testing FAISS index creation...")
d = 384
index = faiss.IndexFlatIP(d)
index.add(np.random.random((100, d)).astype('float32'))
print("   Success")

print("4. Testing FAISS load from disk (simulated)...")
embeddings_path = "/Users/yogdeepbenchimath/Documents/Trademarkia/semantic_search_system/data/processed/document_embeddings.npy"
if os.path.exists(embeddings_path):
    embeddings = np.load(embeddings_path)
    print(f"   Loaded embeddings: {embeddings.shape}")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    print("   Success")
else:
    print("   Embeddings file not found")

print("All tests passed!")

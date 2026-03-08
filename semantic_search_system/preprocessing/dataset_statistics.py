import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CLEANED_DOCS_PATH, LABELS_PATH

def compute_statistics():
    """
    Computes and prints statistics for the processed dataset.
    """
    if not CLEANED_DOCS_PATH.exists() or not LABELS_PATH.exists():
        print("Error: Processed data files not found. Run dataset_loader.py first.")
        return

    with open(CLEANED_DOCS_PATH, 'rb') as f:
        documents = pickle.load(f)
        
    with open(LABELS_PATH, 'rb') as f:
        labels = pickle.load(f)

    total_docs = len(documents)
    doc_lengths = [len(doc.split()) for doc in documents]
    
    avg_len = np.mean(doc_lengths)
    min_len = np.min(doc_lengths)
    max_len = np.max(doc_lengths)
    
    category_dist = Counter(labels)
    
    print("="*30)
    print(" DATASET STATISTICS")
    print("="*30)
    print(f"Total documents:         {total_docs}")
    print(f"Average document length: {avg_len:.2f} words")
    print(f"Minimum document length: {min_len} words")
    print(f"Maximum document length: {max_len} words")
    print("-" * 30)
    print("Category Distribution:")
    for cat, count in sorted(category_dist.items()):
        print(f"{cat:.<30} {count}")
    print("="*30)

if __name__ == "__main__":
    compute_statistics()

import os
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import EXTRACTED_FOLDER, CLEANED_DOCS_PATH, LABELS_PATH, PROCESSED_DATA_DIR
from preprocessing.text_cleaner import clean_text, is_valid_doc

def load_dataset():
    """
    Iterates through category folders, reads documents, cleans them,
    and returns lists of documents and labels.
    """
    documents = []
    labels = []
    
    if not EXTRACTED_FOLDER.exists():
        print(f"Error: Extracted folder not found at {EXTRACTED_FOLDER}")
        return [], []

    categories = sorted([d for d in os.listdir(EXTRACTED_FOLDER) if os.path.isdir(EXTRACTED_FOLDER / d)])
    
    print(f"Loading documents from {len(categories)} categories...")
    
    for category in tqdm(categories):
        category_path = EXTRACTED_FOLDER / category
        for doc_name in os.listdir(category_path):
            doc_path = category_path / doc_name
            if os.path.isfile(doc_path):
                try:
                    # Use latin-1 as requested
                    with open(doc_path, 'r', encoding='latin-1') as f:
                        text = f.read()
                    
                    cleaned = clean_text(text)
                    if is_valid_doc(cleaned):
                        documents.append(cleaned)
                        labels.append(category)
                except Exception as e:
                    print(f"Error reading {doc_path}: {e}")
                    
    print(f"Loaded {len(documents)} documents.")
    
    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CLEANED_DOCS_PATH, 'wb') as f:
        pickle.dump(documents, f)
        
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(labels, f)
        
    print(f"Saved cleaned docs to {CLEANED_DOCS_PATH}")
    print(f"Saved labels to {LABELS_PATH}")
    
    return documents, labels

if __name__ == "__main__":
    load_dataset()

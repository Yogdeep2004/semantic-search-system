import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cdist

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import CLEANED_DOCS_PATH, CLUSTER_MEMBERSHIPS_PATH, CLUSTER_CENTERS_PATH, EMBEDDINGS_PATH

def interpret_clusters():
    """
    Interprets discovered clusters using TF-IDF for keywords and
    finds representative documents closest to cluster centroids.
    """
    if not all(p.exists() for p in [CLEANED_DOCS_PATH, CLUSTER_MEMBERSHIPS_PATH, CLUSTER_CENTERS_PATH, EMBEDDINGS_PATH]):
        print(f"Error: Required files not found. Run clustering scripts first.")
        return

    print("Loading documents, embeddings, and cluster data...")
    with open(CLEANED_DOCS_PATH, 'rb') as f:
        documents = pickle.load(f)
    
    # memberships: (n_clusters, n_samples) in skfuzzy, but we saved as (n_samples, n_clusters)
    memberships = np.load(CLUSTER_MEMBERSHIPS_PATH)
    centers = np.load(CLUSTER_CENTERS_PATH) # (n_clusters, n_features)
    embeddings = np.load(EMBEDDINGS_PATH) # (n_samples, n_features)
    
    n_clusters = centers.shape[0]
    
    # Assign each document to its dominant cluster for keyword analysis
    hard_clusters = np.argmax(memberships, axis=1)
    
    print(f"Interpreting {n_clusters} clusters...")
    
    # Group documents by cluster
    clustered_docs = [[] for _ in range(n_clusters)]
    for doc, cluster_id in zip(documents, hard_clusters):
        clustered_docs[cluster_id].append(doc)
    
    # Identify representative documents (closest to centroid)
    # distance matrix: (n_clusters, n_samples)
    distances = cdist(centers, embeddings, metric='cosine')
    
    # TF-IDF per cluster
    cluster_texts = [" ".join(docs) if docs else "empty" for docs in clustered_docs]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
    feature_names = vectorizer.get_feature_names_out()
    
    print("\nCLUSTER INTERPRETATION SUMMARY:")
    print("=" * 60)
    
    for i in range(n_clusters):
        # 1. Top Keywords
        scores = tfidf_matrix[i].toarray()[0]
        top_keyword_indices = np.argsort(scores)[::-1][:10]
        top_keywords = [feature_names[idx] for idx in top_keyword_indices]
        
        # 2. Representative Documents
        # Get indices of 3 closest documents
        rep_indices = np.argsort(distances[i])[:3]
        
        # 3. Description (using top 3 keywords as a proxy for a description)
        description = f"Topic related to {', '.join(top_keywords[:3])}"
        
        print(f"CLUSTER {i:02d}")
        print(f"Description:   {description}")
        print(f"Top Keywords:  {', '.join(top_keywords)}")
        print(f"Rep. Docs:     Indices {list(rep_indices)}")
        # Print a snippet of the most representative doc
        best_doc_snippet = " ".join(documents[rep_indices[0]].split()[:30]) + "..."
        print(f"Sample Text:   {best_doc_snippet}")
        print("-" * 60)

if __name__ == "__main__":
    interpret_clusters()

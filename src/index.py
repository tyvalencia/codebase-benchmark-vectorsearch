import numpy as np
import faiss
import src.config as config

def build_index(embeddings: np.ndarray):
    """
    Build FAISS index given the BERT embeddings
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index

def search_index(searcher, query_embeddings: np.ndarray):
    """
    Query the FAISS ANN for the k nearest neighbors
    """
    distances, ids = searcher.search(query_embeddings.astype(np.float32), config.TOP_K)
    return ids, distances
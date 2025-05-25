from typing import List, Tuple
import numpy as np
import src.config as config
from src.embeddings import generate_embeddings
from src.index import search_index

def search(queries: List[str], searcher, corpus: List[str]) -> List[List[Tuple[str, float]]]:
    """
    Perform semantic search on the data using the provided queries
    """
    query_embeddings = generate_embeddings(queries) # Generate embeddings for the queries
    ids, distances = search_index(searcher, query_embeddings) # ANN search

    # Map results to sentences
    results: List[List[Tuple[str, float]]] = []
    for row_ids, row_dists in zip(ids, distances):
        hits: List[Tuple[str, float]] = []
        for idx, dist in zip(row_ids, row_dists):
            sentence = corpus[idx]
            hits.append((sentence, float(dist)))
        results.append(hits)
    return results
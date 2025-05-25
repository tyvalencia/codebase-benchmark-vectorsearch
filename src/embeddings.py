from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import src.config as config
from src.util import set_seed

# for compatibility with older versions of PyTorch
if not hasattr(torch, 'get_default_device'): 
    def get_default_device():
        return torch.device('cpu')
    torch.get_default_device = get_default_device

set_seed(config.SEED)
_model = SentenceTransformer(config.EMBEDDING_MODEL) # BERT model 

def generate_embeddings(sentences: List[str]) -> np.ndarray:
    embeddings = _model.encode(sentences, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
    return embeddings

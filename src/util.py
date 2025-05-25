import csv
from typing import List, Tuple
import numpy as np
import src.config as config


def load_tsv(path: str) -> List[Tuple[str, str]]:
    """
    Load the STS data file and return a list of (sentence1, sentence2) pairs
    """
    pairs: List[Tuple[str, str]] = []
    with open(path, newline='', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter='\t')
        next(reader, None) # Skip header
        for row in reader:
            if len(row) < 9:
                continue
            s1 = row[-3] # sentence 1
            s2 = row[-2] # sentence 2
            pairs.append((s1, s2))
    return pairs


def load_tsv_with_scores(path: str) -> List[Tuple[str, str, float]]:
    """
    Load data with scores, return a list of (sentence1, sentence2, score) tuples
    """
    rows: List[Tuple[str, str, float]] = []
    with open(path, newline='', encoding='utf-8') as fd:
        reader = csv.reader(fd, delimiter='\t')
        next(reader, None) # Skip header
        for row in reader:
            if len(row) < 3:
                continue
            try:
                score = float(row[-1])
            except ValueError:
                continue
            s1 = row[-3]
            s2 = row[-2]
            rows.append((s1, s2, score))
    return rows


def set_seed(seed: int = config.SEED) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)

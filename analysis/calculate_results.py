import numpy as np
from scipy.stats import pearsonr


def compute_recall(dev_pairs, results, ks):
    """
    Compute Recall at various K values
    """
    total = len(results)
    recalls = {k: 0 for k in ks}
    for (gold_s1, _), hits in zip(dev_pairs, results):
        retrieved = [sent for sent, _ in hits]
        for k in ks:
            if gold_s1 in retrieved[:k]:
                recalls[k] += 1
    for k in ks:
        recalls[k] /= total
    return recalls


def compute_regression(dev_rows, embed_fn):
    """
    Compute Pearson correlation and MSE for regression task
    """
    s1_list = [s1 for s1, _, _ in dev_rows]
    s2_list = [s2 for _, s2, _ in dev_rows]
    gold_scores = np.array([score for _, _, score in dev_rows], dtype=float)

    emb1 = embed_fn(s1_list)
    emb2 = embed_fn(s2_list)
    pred_scores = np.sum(emb1 * emb2, axis=1)

    pearson_r, _ = pearsonr(pred_scores, gold_scores)
    mse = float(np.mean((pred_scores - gold_scores) ** 2))
    return pearson_r, mse

import argparse
import sys
import logging
import json
import time
import faiss

import src.config as config
from src.util import load_tsv_with_scores, set_seed
from src.embeddings import generate_embeddings
from src.index import build_index, search_index
from analysis.calculate_results import compute_recall, compute_regression
from analysis.plot_results import plot_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="STS-B similarity-search pipeline")
    parser.add_argument("--stage", choices=["all", "index", "search"], default="all") # different stages of the pipeline
    parser.add_argument("--save-metrics", action="store_true") # save metrics to a file
    parser.add_argument("--plot", action="store_true") # show plot
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)

    # --- Load evaluation data ----
    dev_rows = load_tsv_with_scores(str(config.DEV_TSV))
    corpus = [s1 for s1, _, _ in dev_rows]
    dev_pairs = [(s1, s2) for s1, s2, _ in dev_rows]

    searcher = None
    # --- Build index ---
    if args.stage in ("all", "index"):
        embed_start = time.perf_counter()
        embeddings = generate_embeddings(corpus)
        faiss.normalize_L2(embeddings)
        embed_time = time.perf_counter() - embed_start
        print(f"Embedding construction took {embed_time:.2f} seconds") 

        index_start = time.perf_counter()
        searcher = build_index(embeddings)
        index_time = time.perf_counter() - index_start
        print(f"Index construction took {index_time:.2f} seconds") 

        faiss.write_index(searcher, str(config.INDEX_PATH))
        if args.stage == "index":
            return 

    # --- Search and eval ---
    if args.stage in ("all", "search"):
        if searcher is None:
            if config.INDEX_PATH.exists():
                searcher = faiss.read_index(str(config.INDEX_PATH))
            else:
                sys.exit(1) 

        queries = [s2 for _, s2 in dev_pairs]
        queries_start = time.perf_counter()
        query_embeddings = generate_embeddings(queries)
        faiss.normalize_L2(query_embeddings)
        query_embed_time = time.perf_counter() - queries_start
        print(f"Query embedding took {query_embed_time:.2f} seconds")

        search_start = time.perf_counter()
        ids, dists = search_index(searcher, query_embeddings)
        search_time = time.perf_counter() - search_start
        print(f"Search took {search_time:.2f} seconds")

        results = [] # map results to sentences
        for row_ids, row_dists in zip(ids, dists):
            hits = [(corpus[idx], float(dist)) for idx, dist in zip(row_ids, row_dists)]
            results.append(hits)

        # --- Calculate metrics ---
        ks = config.RECALL_KS
        recalls = compute_recall(dev_pairs, results, ks)
        reg_start = time.perf_counter()
        pearson_r, mse = compute_regression(dev_rows, generate_embeddings)
        reg_time = time.perf_counter() - reg_start
        print(f"Regression computation took {reg_time:.2f} seconds")

        # --- Print results ---
        for k in ks:
            print(f"Recall@{k}: {recalls[k]:.4f}")
        print(f"Pearson r: {pearson_r:.3f}")
        print(f"MSE: {mse:.3f}")

        # --- Save and plot ---
        metrics = {f"Recall@{k}": recalls[k] for k in ks}
        metrics.update({
            "PearsonR": pearson_r,
            "MSE": mse,
            "EmbedTime": embed_time,
            "IndexTime": index_time,
            "QueryEmbedTime": query_embed_time,
            "SearchTime": search_time,
            "RegressionTime": reg_time,
        })
        if args.save_metrics:
            with open("analysis/metrics.json", "w") as fd:
                json.dump(metrics, fd, indent=2)
        if args.plot:
            plot_metrics("analysis/metrics.json", "analysis/metrics.png")

if __name__ == "__main__":
    main()

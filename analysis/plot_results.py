import json
import matplotlib.pyplot as plt


def plot_metrics(metrics_path: str = "analysis/metrics.json", output_path: str = "analysis/metrics.png"):
    """
    Read metrics and plot recall and regression performance
    """
    with open(metrics_path, 'r', encoding='utf-8') as fd:
        metrics = json.load(fd)

    recall_keys = [key for key in metrics.keys() if key.startswith("Recall@")] 
    ks = sorted(int(key.split("@")[1]) for key in recall_keys)
    recalls = [metrics[f"Recall@{k}"] for k in ks]

    plt.figure()
    plt.plot(ks, recalls, marker='o', label='Recall@K')

    if 'PearsonR' in metrics:
        pearson = metrics['PearsonR']
        plt.axhline(y=pearson, linestyle='--', color='gray', label=f'Pearson r = {pearson:.3f}')

    plt.xlabel('K')
    plt.ylabel('Score')
    plt.title('STS Retrieval and Regression Performance')
    plt.xticks(ks)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show(block=True)
    plt.close()
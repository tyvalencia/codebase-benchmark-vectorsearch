# ML Model Proof‑of‑Concept for Codebase-wide Changes Benchmark 

**Author:** Ty Valencia

## Description

This baseline pipeline is a proof-of-concept for a similarity search model built on the STS dataset. It provides this framework:

1. Generates sentence embeddings using BERT
2. Builds and queries FAISS to create an index using the dev.tsv sentence1 entries for ANN search 
3. Evaluates retrieval by using the dev.tsv sentence2 entries as queries at varying Ks 
4. Evaluates embedding quality against human similarity scores by using Pearson correlation and Mean Squared Error
5. Visualizes the results

This baseline will be sufficient to test current LLM limitations with codebase-wide changes. 

---

## Repository Structure
```text
analysis/
  calculate_results.py   # computes recall and regression
  plot_results.py        # matplotlib implementation
  metrics.json           # metrics in json form
  metrics.png            # plotted results
dat/
  dev.tsv                # dev split
  train.tsv              # train split
  test.tsv               # test split 
  index.faiss            # where FAISS stores its index
src/
  __init__.py
  config.py              # constants and paths
  util.py                # helper functions for loading data
  embeddings.py          # BERT embeddings creation
  index.py               # FAISS index creation
  logger.py              # logging output
  main.py                # code orchestrator, run this 
  search.py              # top results
requirements.txt         # imports to install
README.md                # this documentation
```

---

## How to Run

### Install dependencies 

`pip install -r requirements.txt` <br>

### Run the project

`python -m src.main` 

Which comes with the following flags: <br>

#### --stage
Different stages of the pipeline, runs all of them by default. <br>
`index` just runs the FAISS index <br>
`search` just runs the search <br>

#### --save-metrics 
Saves the metrics to analysis/metrics.json. 

#### --plot
Generates a plot and saves it to analysis/metrics.png.


---

## Baseline Evaluation Scores

* Recall at K = 1: 0.571
* Recall at K = 2: 0.633
* Recall at K = 5: 0.707
* Recall at K = 10: 0.707
* Recall at K = 15: 0.707
* Recall at K = 20: 0.707
* PearsonR score: 0.858
* Mean Squared Error: 4.79

---

## Benchmark Tasks 

Our proof-of-concept repository contains at least 10 editable files, so it will serve as a good basis for codebase-wide changes. These LLMs also have the tendency to create new files for new tasks to work in tandem with the baseline structure. Below are the current benchmarks that can be used to improve the project and reach a variety of files: 

1. **Hybrid Retrieval** <br>
  Improve vector search by building a BM25 term-scoring index, which could be used to combine their scores by a weighted sum with FAISS to improve recall. <br>
  Files that should change: `config.py, main.py, search.py`

2. **Cross-Encoder Re-Ranking** <br>
  Pass FAISS's top-k results through a transformer cross-encoder to compute pairwise semantic scores, then reorder the results by those refined scores to improve FAISS's ranking. <br>
  Files that should change: `config.py, main.py, search.py/embeddings.py/reranker.py (new)`

3. **Pluggable ANN Backend** <br>
  Implement another ANN model, add a --model flag that allows for the selection of that model or the current FAISS model. <br>
  Files that should change: `requirements.txt, config.py, main.py, index.py`

4. **System-Wide Logging** <br>
  Replace printing calls with structured OpenTelemetry logging to observe timing and attribute data. <br>
  Files that should change: `requirements.txt, config.py, logger.py, main.py, embeddings.py, index.py`

5. **Global Seed Reproducibility** <br>
  Add a --seed flag to initialize all RNG, and write tests that verify two runs with the same seed produce identical embeddings and search outputs. <br>
  Files that should change: `util.py, main.py, test_reproducibility.py (new)`

---

## Testing the Benchmarks

To test the benchmarks, refer to the `benchmarks_prompts.txt` file for the prompts. 
Replace the `<feature>` and `<feature description>` with which benchmark you'd like to test. <br>

Provide the updated prompt, along with following files to the AI: <br>
Analysis folder: `calculate_results.py, plot_results.py` <br>
src folder: `config.py, embeddings.py, index.py, logger.py, main.py, search.py, util.py` <br>
Repo folder: `requirements.txt` <br>

---

## References 

[LiveCodeBench](https://arxiv.org/pdf/2403.07974)

[SWE-Bench](https://arxiv.org/pdf/2310.06770)

[Semantic Textual Similarity Benchmark dataset (GLUE Benchmark)](https://gluebenchmark.com/tasks/) 

[LLM Chatbot Arena](https://lmarena.ai/)
# RAG Portfolios

Research code for the paper [Retriever Portfolios: A Principled Approach to Adaptive RAG](https://arxiv.org/abs/2605.31176), accepted at ICML 2026. The main workflows prepare dense retrieval artifacts, compute DS/Vendi/GraphRAG retrievals, score recall, select portfolios, build answer prompts, run OpenAI-compatible LLM answers, and optionally train or evaluate routers.

Note that many jobs are cache-driven, path-sensitive, and intended for local or cluster artifact directories.

## Citation

If you find this code useful or use it in your work, please consider citing our paper:

```bibtex
@inproceedings{stouras2026retrieverportfolios,
  title = {Retriever Portfolios: A Principled Approach to Adaptive RAG},
  author = {Stouras, Miltiadis and Cohen-Addad, Vincent and Lattanzi, Silvio and Svensson, Ola},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year = {2026},
  note = {arXiv:2605.31176},
  url = {https://arxiv.org/abs/2605.31176}
}
```

## Setup

Create a Python environment and install the repo dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set artifact roots explicitly for your machine:

```bash
export CACHE_DIR=/path/to/cache/
export MODELS_DIR=/path/to/models/
export RESULTS_DIR=/path/to/results/
export PLOTS_DIR=/path/to/plots/
```

If unset, the code uses repo-local `cache/`, `models/`, `results/`, and `plots/` directories while preserving artifact names.

Set dataset roots if you do not want to use the default `datasets/` layout:

```bash
export HOTPOTQA_DIR=/path/to/datasets/HotpotQA-BEIR/
export MUSIQUE_DIR=/path/to/datasets/MuSiQue/
export TRIVIAQA_DIR=/path/to/datasets/TriviaQA/
export TWOWIKI_DIR=/path/to/datasets/2WikiMultiHopQA/
```

Download the four paper datasets:

```bash
bash setup/hotpotqa_download.sh
bash setup/download_musique.sh
bash setup/download_triviaqa.sh
bash setup/download_2wikimultihopqa.sh
```

Download the answer models and router model. Gemma and Llama are gated on Hugging Face, so set `HF_TOKEN` first:

```bash
export HF_TOKEN=...

python - <<'PY'
import os
from huggingface_hub import snapshot_download
from setup.download_llms import GEMMA27B_ID, LLAMA70B_ID, download_llm

models_dir = os.environ.get("MODELS_DIR", "models").rstrip("/")
download_llm(
    GEMMA27B_ID,
    os.environ.get("GEMMA27B_MODEL_DIR", f"{models_dir}/gemma-3-27b-it"),
)
download_llm(
    LLAMA70B_ID,
    os.environ.get("LLAMA70B_MODEL_DIR", f"{models_dir}/llama-3.1-70b-instruct"),
)
snapshot_download(
    repo_id="google/flan-t5-large",
    local_dir=os.environ.get("ROUTER_T5_DIR", f"{models_dir}/flan-t5-large"),
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN"),
)
PY
```

Dense embedder models are downloaded automatically on first use by `sentence-transformers`. To prefetch them:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
SentenceTransformer("intfloat/e5-large-v2")
PY
```

The answer commands expect OpenAI-compatible local servers at the base URLs in `constants.py`. Install `vllm` separately on a GPU machine, then launch each model in its own terminal:

```bash
# Gemma27B, port 5000, default tensor parallel size 2.
bash setup/launch_gemma27b.sh

# Llama70B, port 8000, default tensor parallel size 4.
bash setup/launch_llama70b.sh
```

The launch scripts read `GEMMA27B_MODEL_DIR`, `LLAMA70B_MODEL_DIR`, and `MODELS_DIR`. They also set `CUDA_VISIBLE_DEVICES`; edit the scripts or override the model path variables if your machine layout differs.

## Datasets

Primary datasets are defined in `constants.py`:

- `HotpotQA`
- `MusiQue`
- `TriviaQA`
- `2WikiMultiHopQA`

Dataset loaders live in `dataset_loaders.py`. Local dataset paths default to directories under `datasets/`.

## CLI

List commands and options:

```bash
python main.py --help
python main.py prep --help
```

Core examples:

```bash
# Prepare dataset artifacts.
python main.py prep --datasets HotpotQA --embedder mpnet

# Compute train retrievals and recall scores.
python main.py train-retrievals --datasets HotpotQA --retrievers ds --num-docs 4
python main.py full-pool-recalls --datasets HotpotQA --retrievers ds --num-docs 4 --splits train

# Select a universal portfolio and run test retrievals.
python main.py universal-portfolio --retrievers ds --num-docs 4 --portfolio-size 5
python main.py test-retrievals --datasets HotpotQA --retrievers ds --num-docs 4 --portfolio-size 5

# Build and answer prompts.
python main.py build-answer-prompts --datasets HotpotQA --retrievers ds --num-docs 4 --portfolio-size 5
python main.py answer-prompts --datasets HotpotQA --retrievers ds --num-docs 4 --llm Gemma27B
```

## Complete Pipeline From Scratch

The commands below run the complete paper-style pipeline: build artifacts for all datasets and dense embedders, compute full-pool retrievals and scores, select the all-pool portfolio, materialize test retrievals, train a portfolio router, write router test predictions, and generate final answers.

Set paths and common options first:

```bash
export CACHE_DIR=/path/to/cache/
export MODELS_DIR=/path/to/models/
export RESULTS_DIR=/path/to/results/
export ROUTER_T5_DIR="${MODELS_DIR}/flan-t5-large"

DATASETS="HotpotQA,MusiQue,TriviaQA,2WikiMultiHopQA"
NUM_DOCS=4
K=5
DEVICE=cuda
PORTFOLIO_ID=all_implemented
RUN_ID=k5_router
```

Prepare corpus indexes, question splits, question embeddings, and prefilters for both dense embedding backbones:

```bash
python main.py prep --datasets "$DATASETS" --embedder mpnet --device "$DEVICE" --num-docs "$NUM_DOCS"
python main.py prep --datasets "$DATASETS" --embedder e5 --device "$DEVICE" --num-docs "$NUM_DOCS"
```

Compute dense retriever-pool train/test retrievals and recall scores:

```bash
for EMB in mpnet e5; do
  python main.py train-retrievals --datasets "$DATASETS" --retrievers ds,vendi --embedder "$EMB" --num-docs "$NUM_DOCS" --device "$DEVICE"
  python main.py test-retrievals-pool --datasets "$DATASETS" --retrievers ds,vendi --embedder "$EMB" --num-docs "$NUM_DOCS" --device "$DEVICE" --compute-recalls
done
```

The default `all_implemented` pool includes `graph_dense`, so build graph artifacts and compute graph-dense retrievals as well. To skip graph-dense, use `PORTFOLIO_ID=dense_all` instead.

```bash
python main.py build-graph-index --datasets "$DATASETS" --max-workers 16

python - <<'PY'
from graph_index import (
    answer_saved_question_entity_extraction_prompt_files,
    prepare_all_question_entity_extraction_prompt_files,
)

datasets = ["HotpotQA", "MusiQue", "TriviaQA", "2WikiMultiHopQA"]
splits = ["train", "test"]
prepare_all_question_entity_extraction_prompt_files(datasets, splits=splits)
answer_saved_question_entity_extraction_prompt_files(
    datasets,
    splits=splits,
    max_workers=16,
    checkpoint_every=500,
)
PY

python main.py build-graph-query-entity-cache --datasets "$DATASETS" --splits train,test
python main.py train-retrievals --datasets "$DATASETS" --retrievers graph_dense --num-docs "$NUM_DOCS" --device "$DEVICE"
python main.py test-retrievals-pool --datasets "$DATASETS" --retrievers graph_dense --num-docs "$NUM_DOCS" --device "$DEVICE" --compute-recalls
```

Select and materialize the all-pool portfolio:

```bash
python main.py audit-pool-artifacts --datasets "$DATASETS" --pool-set "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --strict
python main.py universal-portfolio-union --datasets "$DATASETS" --pool-set "$PORTFOLIO_ID" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-size "$K" --device cpu
python main.py materialize-portfolio-test --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS"
```

Train the portfolio router:

```bash
python main.py portfolio-router-train --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-size "$K" --device "$DEVICE" --epochs 10 --batch-size 64 --no-wandb
```

Write test predictions from the trained router checkpoint for each dataset:

```bash
python main.py portfolio-router-predict-test --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-size "$K" --run-id "$RUN_ID" --device "$DEVICE" --batch-size 64
```

Build all-pool member answer prompts, answer them with an LLM, and optionally run the router judge over the router-ranked top-ell answers:

```bash
python main.py build-portfolio-union-answer-prompts --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-size "$K"
python main.py answer-portfolio-union-prompts --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-size "$K" --llm Gemma27B --max-workers 16

python main.py build-all-portfolio-router-judge-prompts --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-sizes "$K" --answer-llms Gemma27B --run-id-map "${K}:${RUN_ID}"
python main.py answer-all-portfolio-router-judge-prompts --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --portfolio-sizes "$K" --answer-llms Gemma27B --judge-llm Gemma27B --run-id-map "${K}:${RUN_ID}" --max-workers 16
```

For family-best paper baselines:

```bash
python main.py select-family-best-baselines --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --device cpu
python main.py compute-family-best-baselines-test --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --max-k "$K" --device "$DEVICE"
python main.py build-family-best-answer-prompts --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --max-k "$K"
python main.py answer-family-best-prompts --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --max-k "$K" --llm Gemma27B --max-workers 16
```

Generate the paper plots from the saved artifacts:

```bash
python main.py plot-all-pool-support --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --max-k "$K"
python main.py plot-portfolio-router-ablations --datasets "$DATASETS" --portfolio-id "$PORTFOLIO_ID" --num-docs "$NUM_DOCS" --max-k "$K" --answer-llms Gemma27B,Llama70B
python main.py plot-resources-vs-accuracy --models Gemma27B,Llama70B --retriever vendi --num-docs "$NUM_DOCS"
```

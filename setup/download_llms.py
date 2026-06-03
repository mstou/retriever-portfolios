import os
from huggingface_hub import snapshot_download

LLAMA70B_ID = "meta-llama/Llama-3.1-70B-Instruct"
GEMMA27B_ID = "google/gemma-3-27b-it"


def download_llm(model_id=GEMMA27B_ID, local_dir=None, token=None):
    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        raise ValueError("Please set HF_TOKEN in your environment: export HF_TOKEN=...")

    if local_dir is None:
        models_dir = os.environ.get("MODELS_DIR", "models")
        local_dir = os.environ.get(
            "GEMMA27B_MODEL_DIR",
            os.path.join(models_dir, "gemma-3-27b-it"),
        )

    return snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=token,
    )

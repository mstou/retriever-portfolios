import os
import torch
import re
import string
from collections import Counter
from datasets import load_dataset, load_from_disk
from typing import Callable, Optional, Tuple

def get_device():
    """Returns the best available device: CUDA, MPS (Mac), or CPU."""
    if torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    # elif torch.backends.mps.is_available():
    #     device = "mps"  # Apple Metal (M1/M2)
    else:
        device = "cpu"  # Default fallback
    
    return torch.device(device)

def get_hotpotqa_dataset(save_path: str = "datasets/hotpotqa_fullwiki"):
    """
        Download or load the HotpotQA fullwiki dataset.
        
        - If data exists at `save_path`, it loads from there.
        - Otherwise, it downloads using Hugging Face and saves as JSON.

        Returns:
            A Hugging Face DatasetDict with 'train' and 'validation' splits.
    """
    if os.path.exists(save_path):
        print("Loading HotpotQA from disk...")
        return load_from_disk(save_path)

    print("Downloading HotpotQA fullwiki from Hugging Face...")
    ds = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
    ds.save_to_disk(save_path)
    return ds


def majority_vote(labels, normalize: Optional[Callable[[str], str]] = None) -> Tuple[Optional[str], int]:
    """
    Return (winner_label, freq) where winner_label is one of the most frequent strings.
    Tie-break: earliest occurrence in `labels`.
    If `normalize` is provided, votes are tallied on normalize(s) but the original
    string from the earliest winner is returned.
    """
    seq = [s for s in labels if isinstance(s, str) and s.strip()]
    if not seq:
        return None, 0

    keys = [normalize(s) if normalize else s for s in seq]
    cnt = Counter(keys)
    top = max(cnt.values())
    winning_keys = {k for k, v in cnt.items() if v == top}

    # earliest original that belongs to any winning key
    for s, k in zip(seq, keys):
        if k in winning_keys:
            return s, top

    return None, 0

def extract_llm_answer_scifact(text):
    #re-write using tagged answer below
    # Returns "SUPPORT" or "CONTRADICT" (or None)
    match = re.search(r"<answer>\s*(SUPPORT|CONTRADICT)\s*</answer>", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None

def extract_tagged_answer(text, tag="judge"):
    """
    Extracts the answer inside the first occurrence of <tag>...</tag> in the string.
    Returns the answer as a string (stripped), or None if not found.
    Example: extract_tagged_answer("final: <judge>SUPPORT</judge>", tag="judge") -> "SUPPORT"
    """
    pattern = fr"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# normalize_answer, exact_match and f1_score are taken from hotpotqa/hotpot
# https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """
    Exact-match between a prediction and ground truth.

    - prediction: str or None
    - ground_truth: str or list[str]

    If `ground_truth` is a list, returns True if the prediction exactly
    matches *any* of the candidate answers (after normalization).
    """
    if prediction is None:
        return False

    # Support multiple gold answers (e.g., TriviaQA)
    if isinstance(ground_truth, (list, tuple)):
        return any(exact_match_score(prediction, gt) for gt in ground_truth)
    
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction, ground_truth):
    """
    Token-level F1 between a prediction and ground truth answer(s).

    - prediction: str or None
    - ground_truth: str or list[str]

    If `ground_truth` is a list, returns the triple (f1, precision, recall)
    for the *best* matching gold answer.
    """
    ZERO_METRIC = (0.0, 0.0, 0.0)

    if prediction is None:
        return ZERO_METRIC

    # Support multiple gold answers by taking the best F1
    if isinstance(ground_truth, (list, tuple)):
        best = ZERO_METRIC
        for gt in ground_truth:
            f1, p, r = f1_score(prediction, gt)
            if f1 > best[0]:
                best = (f1, p, r)
        return best

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


# Metrics for the retrieval of supported documents
def f1_support(predicted_docs, gold_docs):
    """
    Compute precision, recall, F1 for retrieved support documents.
    - predicted_docs: set of doc_ids retrieved
    - gold_docs: set of gold support doc_ids
    """
    pred_set = set(predicted_docs)
    gold_set = set(gold_docs)
    overlap = len(pred_set & gold_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    denom = min(len(pred_set), len(gold_set))
    recall = overlap / denom if denom > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall

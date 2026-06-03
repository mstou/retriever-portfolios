from __future__ import annotations

import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import constants as C
from portfolio_router import (
    PortfolioRouterModel,
    assert_t5_frozen,
    build_portfolio_router_optimizer,
    load_portfolio_router_checkpoint,
    load_portfolio_router_tokenizer,
    portfolio_router_collate,
    save_portfolio_router_checkpoint,
)
from portfolio_router_data import (
    PortfolioRouterData,
    load_portfolio_router_test_data,
    load_portfolio_router_train_data,
    make_portfolio_router_torch_dataset,
    portfolio_router_forward_kwargs,
    subset_portfolio_router_data,
    train_dev_split_portfolio_router_data,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

WANDB_PROJECT = "rag-portfolios"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")


class BalancedRoundRobinBatchSampler:
    def __init__(
        self,
        bins: Dict[str, np.ndarray],
        bin_names: List[str],
        batch_size: int,
        rng_seed: int = 0,
        steps_per_epoch: Optional[int] = None,
    ):
        self.all_bin_names = list(bin_names)
        self.bins = {bn: np.array(bins[bn], dtype=int) for bn in self.all_bin_names}
        self.bin_names = [bn for bn in self.all_bin_names if len(self.bins[bn]) > 0]
        if not self.bin_names:
            raise ValueError("BalancedRoundRobinBatchSampler requires at least one non-empty bin.")
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(rng_seed)

        self._bufs = {}
        self._ptrs = {}
        nonempty_sizes = [len(self.bins[bn]) for bn in self.bin_names]
        n_total = float(sum(nonempty_sizes))
        B = float(len(self.bin_names))
        max_oversample = 10.0
        target_per_bin = (n_total / B) if (B > 0 and n_total > 0.0) else 0.0
        self._quota: Dict[str, int] = {}
        self._emitted: Dict[str, int] = {}
        for bn in self.bin_names:
            self._bufs[bn] = self.rng.permutation(self.bins[bn])
            self._ptrs[bn] = 0
            size = float(len(self.bins[bn]))
            if size <= 0.0 or target_per_bin <= 0.0:
                quota = 0
            else:
                cap = min(target_per_bin, max_oversample * size)
                quota = int(np.floor(cap))
                if quota <= 0 and size > 0.0:
                    quota = int(size)
            self._quota[bn] = quota
            self._emitted[bn] = 0

        total_quota = int(sum(self._quota[bn] for bn in self.bin_names))
        if total_quota <= 0:
            raise ValueError("BalancedRoundRobinBatchSampler computed zero total quota.")

        max_steps = (total_quota + self.batch_size - 1) // self.batch_size
        self.steps_per_epoch = max_steps if steps_per_epoch is None else min(int(steps_per_epoch), max_steps)
        self._remainder_offset = 0

    def _take_from_bin(self, bn: str, k: int) -> List[int]:
        out = []
        while k > 0:
            buf = self._bufs[bn]
            ptr = self._ptrs[bn]
            rem = len(buf) - ptr
            if rem == 0:
                self._bufs[bn] = self.rng.permutation(self.bins[bn])
                self._ptrs[bn] = 0
                continue
            take = min(k, rem)
            out.extend(buf[ptr : ptr + take].tolist())
            self._ptrs[bn] += take
            k -= take
        return out

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            active_bins = [
                bn
                for bn in self.bin_names
                if self._quota.get(bn, 0) > self._emitted.get(bn, 0)
            ]
            if not active_bins:
                yield np.array([], dtype=int)
                continue

            base = self.batch_size // len(active_bins)
            remainder = self.batch_size % len(active_bins)
            order = list(range(len(active_bins)))
            order = order[self._remainder_offset:] + order[:self._remainder_offset]
            extra_bins = set(order[:remainder]) if remainder > 0 else set()
            self._remainder_offset = (self._remainder_offset + 1) % max(1, len(active_bins))

            batch = []
            for i, bn in enumerate(active_bins):
                k = base + (1 if i in extra_bins else 0)
                remaining_quota = self._quota.get(bn, 0) - self._emitted.get(bn, 0)
                to_take = min(k, remaining_quota)
                if to_take <= 0:
                    continue
                taken = self._take_from_bin(bn, to_take)
                self._emitted[bn] += len(taken)
                batch.extend(taken)
            yield np.array(batch, dtype=int)

    def __len__(self) -> int:
        return self.steps_per_epoch


class NaturalIndexBatchSampler:
    def __init__(
        self,
        q_len: int,
        batch_size: int,
        rng_seed: int = 0,
        steps_per_epoch: Optional[int] = None,
    ):
        self.q_len = int(q_len)
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(rng_seed)
        self.steps_per_epoch = (
            (self.q_len + self.batch_size - 1) // self.batch_size
            if steps_per_epoch is None
            else int(steps_per_epoch)
        )

        need = self.steps_per_epoch * self.batch_size
        reps = (need + self.q_len - 1) // self.q_len
        idx = np.concatenate([self.rng.permutation(self.q_len) for _ in range(reps)])
        self._stream = idx[:need].reshape(self.steps_per_epoch, self.batch_size)

    def __iter__(self):
        for batch in self._stream:
            yield batch

    def __len__(self):
        return self.steps_per_epoch


def _translated_artifact_path(path: Union[str, os.PathLike[str]]) -> Path:
    return Path(path)


def _artifact_read_path(path: Union[str, os.PathLike[str]]) -> str:
    expected_path = Path(path)
    if expected_path.exists():
        return str(expected_path)
    translated_path = _translated_artifact_path(expected_path)
    if translated_path != expected_path and translated_path.exists():
        return str(translated_path)
    return str(expected_path)


def _artifact_write_dir(path: Union[str, os.PathLike[str]]) -> str:
    expected_path = Path(path)
    if expected_path.exists() or expected_path.parent.exists():
        return str(expected_path)
    return str(expected_path)


def _parse_datasets(datasets: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if datasets is None:
        return list(C.DATASETS)
    if isinstance(datasets, str):
        values = [item.strip() for item in datasets.split(",") if item.strip()]
    else:
        values = list(datasets)
    if not values:
        raise ValueError("No datasets provided.")
    unknown = [value for value in values if value not in C.DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Allowed: {C.DATASETS}")
    return values


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_limit_for_train_load(
    max_train_questions: Optional[int],
    max_dev_questions: Optional[int],
    dev_ratio: float,
) -> Optional[int]:
    if max_train_questions is not None and max_dev_questions is not None:
        return int(max_train_questions) + int(max_dev_questions)
    if max_train_questions is not None:
        return int(max_train_questions)
    if max_dev_questions is not None:
        if dev_ratio > 0:
            return max(int(round(int(max_dev_questions) / dev_ratio)), int(max_dev_questions))
        return int(max_dev_questions)
    return None


def _cap_questions(data: PortfolioRouterData, max_questions: Optional[int]) -> PortfolioRouterData:
    if max_questions is None or len(data.texts) <= int(max_questions):
        return data
    return subset_portfolio_router_data(data, np.arange(int(max_questions), dtype=int))


def _infer_embedding_dim(name: str, embeddings: Optional[np.ndarray]) -> int:
    if embeddings is None:
        raise ValueError(f"{name} embeddings are unavailable; cannot infer model input dimension.")
    if embeddings.ndim != 2 or embeddings.shape[1] <= 0:
        raise ValueError(f"{name} embeddings must be [Q, D], got shape={embeddings.shape}.")
    return int(embeddings.shape[1])


def _build_eval_loader(
    data: PortfolioRouterData,
    tokenizer: Any,
    *,
    batch_size: int,
    max_length: int,
) -> DataLoader:
    torch_dataset = make_portfolio_router_torch_dataset(
        data,
        tokenizer,
        max_length=max_length,
    )
    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=portfolio_router_collate,
    )


def _build_balanced_loader(
    data: PortfolioRouterData,
    tokenizer: Any,
    *,
    batch_size: int,
    max_length: int,
    rng_seed: int,
) -> DataLoader:
    torch_dataset = make_portfolio_router_torch_dataset(
        data,
        tokenizer,
        max_length=max_length,
    )
    try:
        if data.bins is None or data.bin_names is None:
            raise ValueError("PortfolioRouterData is missing bins for balanced sampling.")
        sampler = BalancedRoundRobinBatchSampler(
            bins=data.bins,
            bin_names=data.bin_names,
            batch_size=batch_size,
            rng_seed=rng_seed,
        )
    except ValueError:
        sampler = NaturalIndexBatchSampler(
            q_len=len(data.texts),
            batch_size=batch_size,
            rng_seed=rng_seed,
        )
    return DataLoader(torch_dataset, batch_sampler=sampler, collate_fn=portfolio_router_collate)


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def contrastive_argmax_loss(
    sims: torch.Tensor,
    recall_k: torch.Tensor,
    *,
    max_tie_size: int = 3,
    eps: float = 1e-12,
) -> torch.Tensor:
    if max_tie_size < 1:
        raise ValueError("max_tie_size must be >= 1.")
    max_vals, _ = recall_k.max(dim=1, keepdim=True)
    winners = (recall_k >= (max_vals - eps)) & (max_vals > 0.0)
    tie_counts = winners.sum(dim=1)
    valid = (tie_counts >= 1) & (tie_counts <= max_tie_size)
    if not valid.any():
        return sims.new_tensor(0.0)

    sims_valid = sims[valid]
    winners_valid = winners[valid].float()
    logp = F.log_softmax(sims_valid, dim=1)
    denom = winners_valid.sum(dim=1).clamp_min(1.0)
    loss = -(logp * winners_valid).sum(dim=1) / denom
    return loss.mean()


def _sample_recalls(sims: torch.Tensor, recall_k: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(sims, dim=-1)
    return recall_k[torch.arange(recall_k.size(0), device=recall_k.device), preds]


def _topn_oracle_recalls(sims: torch.Tensor, recall_k: torch.Tensor, top_n: int) -> torch.Tensor:
    batch_size, portfolio_size = recall_k.shape
    if batch_size == 0 or portfolio_size == 0:
        return recall_k.new_zeros((0,))
    top_n = min(int(top_n), portfolio_size)
    top_idx = torch.topk(sims, k=top_n, dim=-1).indices
    return recall_k.gather(1, top_idx).max(dim=1).values


def _filter_zero_recall_queries(data: PortfolioRouterData) -> Tuple[PortfolioRouterData, int]:
    keep = np.where(np.max(data.recalls, axis=0) > 0)[0]
    skipped = len(data.texts) - int(keep.size)
    if skipped <= 0:
        return data, 0
    return subset_portfolio_router_data(data, keep), skipped


def _finalize_recall_metrics(accum: Dict[str, float]) -> Dict[str, float]:
    examples = max(1.0, accum.get("examples", 0.0))
    return {
        "examples": accum.get("examples", 0.0),
        "argmax_recall": accum.get("argmax_recall_sum", 0.0) / examples,
        "top2_oracle_recall": accum.get("top2_recall_sum", 0.0) / examples,
    }


def _as_prediction_matrix(
    name: str,
    value: Any,
    *,
    num_questions: int,
    portfolio_size: int,
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {matrix.shape}")
    if matrix.shape[0] != int(num_questions):
        raise ValueError(
            f"{name} row count mismatch: rows={matrix.shape[0]} questions={num_questions}"
        )
    if matrix.shape[1] != int(portfolio_size):
        raise ValueError(
            f"{name} width mismatch: width={matrix.shape[1]} portfolio_size={portfolio_size}"
        )
    return matrix


def _softmax_np(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32, copy=True)
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = exp.sum(axis=1, keepdims=True)
    return (exp / np.clip(denom, 1e-12, None)).astype(np.float32)


def _top_indices_from_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.argsort(-np.asarray(matrix, dtype=np.float32), axis=1, kind="mergesort").astype(
        np.int64,
        copy=False,
    )


def _top_indices_from_argmax(argmax: np.ndarray, portfolio_size: int) -> np.ndarray:
    base = np.arange(int(portfolio_size), dtype=np.int64)
    top_indices = np.empty((argmax.shape[0], int(portfolio_size)), dtype=np.int64)
    for row_idx, winner in enumerate(argmax):
        top_indices[row_idx, 0] = int(winner)
        top_indices[row_idx, 1:] = base[base != int(winner)]
    return top_indices


def evaluate_portfolio_router(
    model: PortfolioRouterModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    assert_t5_frozen(model)
    accum: Dict[str, float] = {}
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device)
            sims = model(**portfolio_router_forward_kwargs(batch))
            recalls = _sample_recalls(sims, batch["recall_k"])
            top2 = _topn_oracle_recalls(sims, batch["recall_k"], top_n=2)
            accum["examples"] = accum.get("examples", 0.0) + float(recalls.numel())
            accum["argmax_recall_sum"] = accum.get("argmax_recall_sum", 0.0) + float(
                recalls.sum().detach().cpu().item()
            )
            accum["top2_recall_sum"] = accum.get("top2_recall_sum", 0.0) + float(
                top2.sum().detach().cpu().item()
            )
    assert_t5_frozen(model)
    return _finalize_recall_metrics(accum)


def write_portfolio_router_test_predictions(
    predictor,
    *,
    portfolio_id: str,
    dataset_name: str,
    num_docs_to_fetch: int,
    portfolio_size: int,
    split: str = "test",
    run_id: str,
    max_questions: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Run a predictor on loaded portfolio-router data and write prediction artifacts.

    `predictor` can be a callable, expose `predict_proba(data)`, or expose
    `predict(data)`. A 2D output is interpreted as per-retriever scores and a
    1D output as predicted retriever labels.
    """
    data = load_portfolio_router_test_data(
        portfolio_id,
        datasets=[dataset_name],
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        max_questions=max_questions,
        strict=strict,
    )

    raw_kind = "callable_scores"
    if hasattr(predictor, "predict_proba"):
        raw = predictor.predict_proba(data)
        raw_kind = "probabilities"
    elif hasattr(predictor, "predict"):
        raw = predictor.predict(data)
        raw_kind = "labels"
    elif callable(predictor):
        raw = predictor(data)
    else:
        raise TypeError("predictor must be callable or expose predict/predict_proba.")

    scores = None
    probabilities = None
    argmax = None
    if isinstance(raw, dict):
        scores = raw.get("scores", raw.get("logits"))
        probabilities = raw.get("probabilities", raw.get("probs"))
        argmax = raw.get("argmax", raw.get("predictions", raw.get("labels")))
    else:
        arr = np.asarray(raw)
        if arr.ndim == 2:
            if raw_kind == "probabilities":
                probabilities = arr
            else:
                scores = arr
        elif arr.ndim == 1:
            argmax = arr
        else:
            raise ValueError(f"Unsupported predictor output shape: {arr.shape}")

    if scores is not None:
        scores = _as_prediction_matrix(
            "scores",
            scores,
            num_questions=len(data.texts),
            portfolio_size=portfolio_size,
        )

    if probabilities is not None:
        probabilities = _as_prediction_matrix(
            "probabilities",
            probabilities,
            num_questions=len(data.texts),
            portfolio_size=portfolio_size,
        )

    ranking_matrix = scores if scores is not None else probabilities
    if ranking_matrix is not None:
        top_indices = _top_indices_from_matrix(ranking_matrix)
        top_indices_source = "scores" if scores is not None else "probabilities"
        if argmax is None:
            argmax = top_indices[:, 0]
    else:
        top_indices = None
        top_indices_source = "argmax_with_natural_order"

    if argmax is None:
        raise ValueError("predictor output must provide labels, scores, or probabilities.")
    argmax = np.asarray(argmax, dtype=np.int64)
    if argmax.ndim != 1 or argmax.shape[0] != len(data.texts):
        raise ValueError(f"argmax must be shape [num_questions], got {argmax.shape}")
    if argmax.size and (argmax.min() < 0 or argmax.max() >= int(portfolio_size)):
        raise ValueError(
            f"argmax contains out-of-range portfolio ranks: "
            f"min={int(argmax.min())}, max={int(argmax.max())}, k={portfolio_size}"
        )
    if top_indices is None:
        top_indices = _top_indices_from_argmax(argmax, int(portfolio_size))
    elif np.any(argmax != top_indices[:, 0]):
        raise ValueError("argmax is inconsistent with the top-ranked prediction matrix column.")
    if probabilities is None and scores is not None:
        probabilities = _softmax_np(scores)

    output_path = Path(
        C.get_portfolio_router_predictions(
            portfolio_id,
            dataset_name,
            num_docs_to_fetch,
            portfolio_size,
            split,
            run_id,
        )
    )
    metadata_path = Path(
        C.get_portfolio_router_prediction_metadata(
            portfolio_id,
            dataset_name,
            num_docs_to_fetch,
            portfolio_size,
            split,
            run_id,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "portfolio_router_predictions",
        "schema_version": 2,
        "portfolio_id": portfolio_id,
        "dataset": dataset_name,
        "num_docs": int(num_docs_to_fetch),
        "portfolio_size": int(portfolio_size),
        "split": split,
        "run_id": run_id,
        "argmax": argmax,
        "top_indices": top_indices,
        "top_indices_source": top_indices_source,
        "scores": scores,
        "probabilities": probabilities,
        "question_datasets": data.question_datasets,
        "question_indices": data.question_indices,
        "selected_retrievers": data.selected_retrievers,
    }
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "schema": "portfolio_router_prediction_metadata",
        "schema_version": 2,
        "portfolio_id": portfolio_id,
        "dataset": dataset_name,
        "num_docs": int(num_docs_to_fetch),
        "portfolio_size": int(portfolio_size),
        "split": split,
        "run_id": run_id,
        "num_questions": len(data.texts),
        "has_scores": scores is not None,
        "has_probabilities": probabilities is not None,
        "top_indices_source": top_indices_source,
        "predictions_path": str(output_path),
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return payload


def write_portfolio_router_test_predictions_from_checkpoint(
    *,
    portfolio_id: str = C.POOL_SET_ALL_IMPLEMENTED,
    datasets: Optional[Union[str, Sequence[str]]] = None,
    num_docs_to_fetch: int = 4,
    portfolio_size: int = C.PORTFOLIO_SIZE,
    run_id: str,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    device: str = "cuda",
    batch_size: int = 64,
    max_questions: Optional[int] = None,
    strict: bool = True,
    t5_model_name_or_path: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_list = _parse_datasets(datasets)
    device_obj = torch.device(device)
    resolved_checkpoint = str(
        checkpoint_path
        or Path(C.MODELS_DIR) / "portfolio_router" / portfolio_id / f"k{int(portfolio_size)}" / "portfolio_router_best.pt"
    )
    checkpoint_read_path = _artifact_read_path(resolved_checkpoint)
    print(
        f"[portfolio-router-predict] checkpoint={resolved_checkpoint} "
        f"loaded_path={checkpoint_read_path} device={device_obj}",
        flush=True,
    )
    model, checkpoint = load_portfolio_router_checkpoint(
        checkpoint_read_path,
        map_location=device_obj,
        t5_model_name_or_path=t5_model_name_or_path,
    )
    model.to(device_obj)
    model.eval()
    assert_t5_frozen(model)
    tokenizer_name_or_path = (
        checkpoint.get("tokenizer_name_or_path")
        or (checkpoint.get("model_config") or {}).get("tokenizer_name_or_path")
        or t5_model_name_or_path
        or model.config.resolved_tokenizer_path()
    )
    tokenizer_path = _artifact_read_path(tokenizer_name_or_path)
    tokenizer = load_portfolio_router_tokenizer(tokenizer_path)

    def predict_scores(data: PortfolioRouterData) -> np.ndarray:
        torch_dataset = make_portfolio_router_torch_dataset(data, tokenizer)
        loader = DataLoader(
            torch_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=portfolio_router_collate,
        )
        score_rows = []
        with torch.no_grad():
            for batch in loader:
                batch = _batch_to_device(batch, device_obj)
                scores = model(**portfolio_router_forward_kwargs(batch))
                score_rows.append(scores.detach().cpu())
        if not score_rows:
            return np.empty((0, int(portfolio_size)), dtype=np.float32)
        return torch.cat(score_rows, dim=0).numpy()

    outputs: Dict[str, Any] = {
        "checkpoint": resolved_checkpoint,
        "checkpoint_loaded_path": checkpoint_read_path,
        "checkpoint_extra": checkpoint.get("extra", {}),
        "datasets": {},
    }
    for dataset_name in dataset_list:
        print(
            f"[portfolio-router-predict] dataset={dataset_name} "
            f"portfolio_id={portfolio_id} k={portfolio_size} run_id={run_id}",
            flush=True,
        )
        payload = write_portfolio_router_test_predictions(
            predict_scores,
            portfolio_id=portfolio_id,
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs_to_fetch,
            portfolio_size=portfolio_size,
            split="test",
            run_id=run_id,
            max_questions=max_questions,
            strict=strict,
        )
        outputs["datasets"][dataset_name] = {
            "num_questions": len(payload["question_indices"]),
            "predictions_path": C.get_portfolio_router_predictions(
                portfolio_id,
                dataset_name,
                num_docs_to_fetch,
                portfolio_size,
                "test",
                run_id,
            ),
            "metadata_path": C.get_portfolio_router_prediction_metadata(
                portfolio_id,
                dataset_name,
                num_docs_to_fetch,
                portfolio_size,
                "test",
                run_id,
            ),
        }
    return outputs


def _train_one_epoch(
    model: PortfolioRouterModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip: Optional[float],
    grad_accum_steps: int,
    max_tie_size: int,
) -> Dict[str, float]:
    model.train()
    assert_t5_frozen(model)
    running = 0.0
    running_recall = 0.0
    running_count = 0
    steps = 0
    grad_accum_steps = int(grad_accum_steps)
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1.")
    optimizer.zero_grad()
    num_batches = len(loader)
    for batch in loader:
        steps += 1
        batch = _batch_to_device(batch, device)
        sims = model(**portfolio_router_forward_kwargs(batch))
        loss = contrastive_argmax_loss(sims, batch["recall_k"], max_tie_size=max_tie_size)
        batch_recalls = _sample_recalls(sims, batch["recall_k"])
        running_recall += float(batch_recalls.sum().detach().cpu().item())
        running_count += int(batch_recalls.numel())
        (loss / grad_accum_steps).backward()
        if (steps % grad_accum_steps == 0) or (steps == num_batches):
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        running += float(loss.detach().cpu().item())
        assert_t5_frozen(model)
    return {
        "loss": running / max(1, steps),
        "argmax_recall": running_recall / running_count if running_count > 0 else 0.0,
        "examples": float(running_count),
    }


def _default_output_dir(portfolio_id: str, portfolio_size: int) -> str:
    return str(Path(C.MODELS_DIR) / "portfolio_router" / portfolio_id / f"k{portfolio_size}")


def _write_metrics_json(path: Union[str, os.PathLike[str]], payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _read_metrics_history(path: Union[str, os.PathLike[str]]) -> Tuple[List[Dict[str, Any]], float]:
    read_path = _artifact_read_path(path)
    if not os.path.exists(read_path):
        return [], -1.0
    try:
        with open(read_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[portfolio-router-train] could not read metrics for resume: {read_path}: {exc}", flush=True)
        return [], -1.0
    history = payload.get("history", [])
    if not isinstance(history, list):
        history = []
    best_dev = payload.get("best_dev_recall", payload.get("best_dev_argmax_recall", -1.0))
    try:
        best_dev = float(best_dev)
    except (TypeError, ValueError):
        best_dev = -1.0
    return history, best_dev


def _start_wandb_run(
    config: Dict[str, Any],
    run_name: str,
    *,
    enabled: bool,
    resume_id: Optional[str] = None,
) -> Optional[Any]:
    if not enabled:
        return None
    if wandb is None:
        print("[portfolio-router-train][wandb] wandb is not installed; skipping logging.", flush=True)
        return None
    try:
        return wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config=config,
            name=run_name,
            reinit=True,
            id=resume_id,
            resume="allow" if resume_id else None,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[portfolio-router-train][wandb] failed to initialize run: {exc}", flush=True)
        return None


def _resolve_resume_checkpoint(
    resume_from: Optional[Union[str, os.PathLike[str]]],
    *,
    auto_resume: bool,
    last_checkpoint_path: str,
) -> Optional[str]:
    if resume_from is not None:
        raw = Path(resume_from)
        candidate = raw / "portfolio_router_last.pt" if raw.is_dir() else raw
        read_path = _artifact_read_path(candidate)
        if not os.path.exists(read_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_from} (checked {read_path})")
        return read_path
    if not auto_resume:
        return None
    read_path = _artifact_read_path(last_checkpoint_path)
    return read_path if os.path.exists(read_path) else None


def train_portfolio_router(
    portfolio_id: str = C.POOL_SET_ALL_IMPLEMENTED,
    datasets: Optional[Union[str, Sequence[str]]] = None,
    num_docs_to_fetch: int = 4,
    portfolio_size: int = C.PORTFOLIO_SIZE,
    output_dir: Optional[Union[str, os.PathLike[str]]] = None,
    device: str = "cuda",
    batch_size: int = 64,
    max_length: int = 256,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    dev_ratio: float = 0.1,
    seed: int = 0,
    tie_2_fraction_of_unique: float = 0.5,
    tie_3_fraction_of_unique: float = 0.1,
    max_tie_size: int = 3,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    max_train_questions: Optional[int] = None,
    max_dev_questions: Optional[int] = None,
    max_test_questions: Optional[int] = None,
    load_test: bool = True,
    t5_model_name_or_path: Optional[str] = None,
    tokenizer_name_or_path: Optional[str] = None,
    use_wandb: bool = True,
    wandb_run_name: Optional[str] = None,
    resume_from: Optional[Union[str, os.PathLike[str]]] = None,
    auto_resume: bool = True,
) -> Dict[str, Any]:
    _set_seed(seed)
    dataset_list = _parse_datasets(datasets)
    requested_output_dir = str(output_dir or _default_output_dir(portfolio_id, portfolio_size))
    write_output_dir = _artifact_write_dir(requested_output_dir)
    Path(write_output_dir).mkdir(parents=True, exist_ok=True)

    actual_t5_path = t5_model_name_or_path or _artifact_read_path(C.ROUTER_T5_DIR)
    actual_tokenizer_path = tokenizer_name_or_path or actual_t5_path
    device_obj = torch.device(device)

    train_load_limit = _dataset_limit_for_train_load(
        max_train_questions,
        max_dev_questions,
        dev_ratio,
    )
    print(
        f"[portfolio-router-train] portfolio_id={portfolio_id} datasets={','.join(dataset_list)} "
        f"k={portfolio_size} num_docs={num_docs_to_fetch} device={device_obj}",
        flush=True,
    )
    print(
        f"[portfolio-router-train] output_dir={requested_output_dir} write_dir={write_output_dir}",
        flush=True,
    )
    print(
        f"[portfolio-router-train] t5_path={actual_t5_path} tokenizer_path={actual_tokenizer_path}",
        flush=True,
    )

    train_full = load_portfolio_router_train_data(
        portfolio_id,
        datasets=dataset_list,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        max_questions=train_load_limit,
        strict=True,
    )
    train_data, dev_data = train_dev_split_portfolio_router_data(
        train_full,
        dev_ratio=dev_ratio,
        random_seed=seed,
        label_mode="argmax",
        temperature=1.0,
        tie_2_fraction_of_unique=tie_2_fraction_of_unique,
        tie_3_fraction_of_unique=tie_3_fraction_of_unique,
        break_ties_to_lowest=False,
    )
    train_data = _cap_questions(train_data, max_train_questions)
    dev_data = _cap_questions(dev_data, max_dev_questions)

    test_data: Optional[PortfolioRouterData] = None
    test_error: Optional[str] = None
    if load_test:
        try:
            raw_test = load_portfolio_router_test_data(
                portfolio_id,
                datasets=dataset_list,
                num_docs_to_fetch=num_docs_to_fetch,
                portfolio_size=portfolio_size,
                max_questions=max_test_questions,
                strict=True,
            )
            test_data, skipped_test = _filter_zero_recall_queries(raw_test)
            if skipped_test > 0:
                print(
                    f"[portfolio-router-train] skipping {skipped_test} zero-recall test questions",
                    flush=True,
                )
        except FileNotFoundError as exc:
            test_error = str(exc)
            print(f"[portfolio-router-train] skipping test data: {test_error}", flush=True)

    k = len(train_data.selected_retrievers)
    mpnet_dim = _infer_embedding_dim("mpnet", train_data.mpnet_embeddings)
    e5_dim = _infer_embedding_dim("e5", train_data.e5_embeddings)
    print(
        f"[portfolio-router-train] loaded train={len(train_data.texts)} dev={len(dev_data.texts)} "
        f"test={len(test_data.texts) if test_data is not None else 0} "
        f"train_total={train_data.total_questions} train_zero={train_data.zero_questions} "
        f"dev_total={dev_data.total_questions} dev_zero={dev_data.zero_questions} "
        f"mpnet_dim={mpnet_dim} e5_dim={e5_dim}",
        flush=True,
    )
    for member in train_data.selected_retrievers:
        print(
            f"[portfolio-router-train] member rank={member.get('rank')} "
            f"pool={member.get('pool_id', member.get('pool_label'))} "
            f"retriever={member.get('retriever')} embedder={member.get('artifact_embedder_key')} "
            f"local_idx={member.get('local_idx')}",
            flush=True,
        )

    tokenizer = load_portfolio_router_tokenizer(actual_tokenizer_path)
    model = PortfolioRouterModel(
        portfolio_size=k,
        mpnet_embedding_dim=mpnet_dim,
        e5_embedding_dim=e5_dim,
        t5_model_name_or_path=actual_t5_path,
        tokenizer_name_or_path=actual_tokenizer_path,
    )
    model.to(device_obj)
    assert_t5_frozen(model)
    optimizer = build_portfolio_router_optimizer(
        model,
        lr=lr,
        weight_decay=weight_decay,
    )

    dev_loader = _build_eval_loader(
        dev_data,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    test_loader = (
        _build_eval_loader(
            test_data,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
        )
        if test_data is not None
        else None
    )

    best_expected_path = str(Path(requested_output_dir) / "portfolio_router_best.pt")
    last_expected_path = str(Path(requested_output_dir) / "portfolio_router_last.pt")
    metrics_expected_path = str(Path(requested_output_dir) / "metrics.json")
    best_write_path = str(Path(write_output_dir) / "portfolio_router_best.pt")
    last_write_path = str(Path(write_output_dir) / "portfolio_router_last.pt")
    metrics_write_path = str(Path(write_output_dir) / "metrics.json")
    best_dev_recall = -1.0
    history: List[Dict[str, Any]] = []
    start_epoch = 0
    resume_wandb_id: Optional[str] = None

    resume_checkpoint = _resolve_resume_checkpoint(
        resume_from,
        auto_resume=auto_resume,
        last_checkpoint_path=last_write_path,
    )
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state")
        if not isinstance(state, dict):
            raise ValueError(f"Resume checkpoint is missing model_state: {resume_checkpoint}")
        load_result = model.load_state_dict(state, strict=False)
        non_t5_missing = [key for key in load_result.missing_keys if not key.startswith("t5.")]
        if non_t5_missing or load_result.unexpected_keys:
            raise RuntimeError(
                f"Failed to resume portfolio router from {resume_checkpoint}: "
                f"missing_non_t5={non_t5_missing}, unexpected={load_result.unexpected_keys}"
            )
        extra = checkpoint.get("extra") or {}
        optimizer_state = extra.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        start_epoch = int(extra.get("epoch", checkpoint.get("epoch", 0)) or 0)
        resume_wandb_id = extra.get("wandb_run_id") or checkpoint.get("wandb_run_id")
        history, best_dev_recall = _read_metrics_history(metrics_write_path)
        print(
            f"[portfolio-router-train] resumed checkpoint={resume_checkpoint} "
            f"start_epoch={start_epoch} best_dev_recall={best_dev_recall:.4f}",
            flush=True,
        )
        assert_t5_frozen(model)

    base_metadata = {
        "portfolio_id": portfolio_id,
        "portfolio_size": k,
        "datasets": dataset_list,
        "num_docs": int(num_docs_to_fetch),
        "selected_retrievers": train_data.selected_retrievers,
        "output_dir": requested_output_dir,
        "written_output_dir": write_output_dir,
        "t5_model_name_or_path": C.ROUTER_T5_DIR,
        "t5_loaded_path": actual_t5_path,
        "tokenizer_name_or_path": actual_tokenizer_path,
        "mpnet_embedding_dim": mpnet_dim,
        "e5_embedding_dim": e5_dim,
        "label_and_split_style": "portfolio_router",
        "label_mode": "argmax",
        "temperature": 1.0,
        "tie_2_fraction_of_unique": float(tie_2_fraction_of_unique),
        "tie_3_fraction_of_unique": float(tie_3_fraction_of_unique),
        "max_tie_size": int(max_tie_size),
        "grad_clip": grad_clip,
        "grad_accum_steps": int(grad_accum_steps),
        "t5_trainable": False,
    }

    wandb_config = {
        "portfolio_id": portfolio_id,
        "datasets": dataset_list,
        "num_docs": int(num_docs_to_fetch),
        "portfolio_size": k,
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "dev_ratio": float(dev_ratio),
        "seed": int(seed),
        "label_and_split_style": "portfolio_router",
        "label_mode": "argmax",
        "temperature": 1.0,
        "tie_2_fraction_of_unique": float(tie_2_fraction_of_unique),
        "tie_3_fraction_of_unique": float(tie_3_fraction_of_unique),
        "max_tie_size": int(max_tie_size),
        "grad_clip": grad_clip,
        "grad_accum_steps": int(grad_accum_steps),
        "t5_trainable": False,
        "t5_saved_in_checkpoint": False,
        "output_dir": requested_output_dir,
        "written_output_dir": write_output_dir,
    }
    resolved_wandb_name = (
        wandb_run_name
        or f"portfolio-router-{portfolio_id}-k{k}-lr{lr:g}-wd{weight_decay:g}-seed{seed}"
    )
    wandb_run = _start_wandb_run(
        wandb_config,
        resolved_wandb_name,
        enabled=use_wandb,
        resume_id=resume_wandb_id,
    )

    for epoch in range(start_epoch + 1, int(epochs) + 1):
        train_loader = _build_balanced_loader(
            train_data,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            rng_seed=seed + epoch,
        )
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device_obj,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            max_tie_size=max_tie_size,
        )
        dev_metrics = evaluate_portfolio_router(model, dev_loader, device=device_obj)
        test_metrics = (
            evaluate_portfolio_router(model, test_loader, device=device_obj)
            if test_loader is not None
            else None
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "test": test_metrics,
        }
        history.append(epoch_record)
        print(
            f"[portfolio-router-train] epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_recall={train_metrics['argmax_recall']:.4f} "
            f"dev_recall={dev_metrics['argmax_recall']:.4f} "
            f"dev_top2={dev_metrics['top2_oracle_recall']:.4f}",
            flush=True,
        )
        if test_metrics is not None:
            print(
                f"[portfolio-router-train] epoch={epoch:03d} "
                f"test_recall={test_metrics['argmax_recall']:.4f} "
                f"test_top2={test_metrics['top2_oracle_recall']:.4f}",
                flush=True,
            )

        checkpoint_extra = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
            "optimizer_state": optimizer.state_dict(),
            "wandb_run_id": wandb_run.id if wandb_run is not None else resume_wandb_id,
        }
        save_portfolio_router_checkpoint(
            model,
            last_write_path,
            portfolio_metadata={
                **base_metadata,
                "checkpoint_path": last_expected_path,
                "checkpoint_written_path": last_write_path,
            },
            extra=checkpoint_extra,
        )
        if dev_metrics["argmax_recall"] > best_dev_recall:
            best_dev_recall = dev_metrics["argmax_recall"]
            save_portfolio_router_checkpoint(
                model,
                best_write_path,
                portfolio_metadata={
                    **base_metadata,
                    "checkpoint_path": best_expected_path,
                    "checkpoint_written_path": best_write_path,
                    "best_epoch": epoch,
                },
                extra=checkpoint_extra,
            )

        metrics_payload = {
            "config": {
                "portfolio_id": portfolio_id,
                "datasets": dataset_list,
                "num_docs_to_fetch": int(num_docs_to_fetch),
                "portfolio_size": k,
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "epochs": int(epochs),
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "dev_ratio": float(dev_ratio),
                "seed": int(seed),
                "label_and_split_style": "portfolio_router",
                "label_mode": "argmax",
                "temperature": 1.0,
                "tie_2_fraction_of_unique": float(tie_2_fraction_of_unique),
                "tie_3_fraction_of_unique": float(tie_3_fraction_of_unique),
                "max_tie_size": int(max_tie_size),
                "grad_clip": grad_clip,
                "grad_accum_steps": int(grad_accum_steps),
                "t5_trainable": False,
                "use_wandb": bool(use_wandb),
                "wandb_run_name": resolved_wandb_name,
                "resume_from": str(resume_from) if resume_from is not None else None,
                "auto_resume": bool(auto_resume),
                "max_train_questions": max_train_questions,
                "max_dev_questions": max_dev_questions,
                "max_test_questions": max_test_questions,
            },
            "paths": {
                "output_dir": requested_output_dir,
                "written_output_dir": write_output_dir,
                "best_checkpoint": best_expected_path,
                "best_checkpoint_written": best_write_path,
                "last_checkpoint": last_expected_path,
                "last_checkpoint_written": last_write_path,
                "metrics": metrics_expected_path,
                "metrics_written": metrics_write_path,
            },
            "test_error": test_error,
            "history": history,
            "best_dev_recall": best_dev_recall,
        }
        _write_metrics_json(metrics_write_path, metrics_payload)
        assert_t5_frozen(model)
        if wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_recall": train_metrics["argmax_recall"],
                "dev_recall": dev_metrics["argmax_recall"],
                "dev_top2_oracle_recall": dev_metrics["top2_oracle_recall"],
                "best_dev_recall": best_dev_recall,
            }
            if test_metrics is not None:
                log_payload["test_recall"] = test_metrics["argmax_recall"]
                log_payload["test_top2_oracle_recall"] = test_metrics["top2_oracle_recall"]
            wandb.log(log_payload, step=epoch)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"[portfolio-router-train] best_checkpoint={best_expected_path}", flush=True)
    print(f"[portfolio-router-train] best_checkpoint_written={best_write_path}", flush=True)
    print(f"[portfolio-router-train] last_checkpoint={last_expected_path}", flush=True)
    print(f"[portfolio-router-train] last_checkpoint_written={last_write_path}", flush=True)
    print(f"[portfolio-router-train] metrics={metrics_expected_path}", flush=True)
    print(f"[portfolio-router-train] metrics_written={metrics_write_path}", flush=True)
    return {
        "output_dir": requested_output_dir,
        "written_output_dir": write_output_dir,
        "best_checkpoint": best_expected_path,
        "best_checkpoint_written": best_write_path,
        "last_checkpoint": last_expected_path,
        "last_checkpoint_written": last_write_path,
        "metrics": metrics_expected_path,
        "metrics_written": metrics_write_path,
        "history": history,
        "test_error": test_error,
    }

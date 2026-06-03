"""
Standalone final portfolio router.

This module defines the model and checkpoint contract for all-pool portfolios where a
single selected portfolio can contain retrievers backed by mpnet, e5,
graph_dense, or mixed artifacts.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import T5EncoderModel, T5TokenizerFast

import constants as C


CHECKPOINT_SCHEMA = "portfolio_router_checkpoint"
CHECKPOINT_VERSION = 2


@dataclass(frozen=True)
class PortfolioRouterConfig:
    portfolio_size: int
    mpnet_embedding_dim: int
    e5_embedding_dim: int
    t5_model_name_or_path: str = C.ROUTER_T5_DIR
    tokenizer_name_or_path: Optional[str] = None
    router_hidden_dim: int = 256
    fusion_hidden_dim: int = 256
    dropout: float = 0.1

    def resolved_tokenizer_path(self) -> str:
        return self.tokenizer_name_or_path or self.t5_model_name_or_path


@dataclass
class PortfolioRouterBatch:
    """
    Batch contract for PortfolioRouterModel.

    Missing embeddings may be passed as None.  If an embedding tensor is
    present, its mask is optional and defaults to all ones.  Mask values of 0
    suppress that modality after projection so projection biases cannot create
    fake features for missing rows.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mpnet_embedding: Optional[torch.Tensor] = None
    e5_embedding: Optional[torch.Tensor] = None
    mpnet_mask: Optional[torch.Tensor] = None
    e5_mask: Optional[torch.Tensor] = None


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}.")
    return value


def _coerce_config(config: Union[PortfolioRouterConfig, Dict[str, Any]]) -> PortfolioRouterConfig:
    if isinstance(config, PortfolioRouterConfig):
        cfg = config
    else:
        cfg = PortfolioRouterConfig(**dict(config))

    dropout = float(cfg.dropout)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError(f"dropout must be in [0, 1); got {cfg.dropout}.")
    return replace(
        cfg,
        portfolio_size=_validate_positive_int("portfolio_size", cfg.portfolio_size),
        mpnet_embedding_dim=_validate_positive_int("mpnet_embedding_dim", cfg.mpnet_embedding_dim),
        e5_embedding_dim=_validate_positive_int("e5_embedding_dim", cfg.e5_embedding_dim),
        router_hidden_dim=_validate_positive_int("router_hidden_dim", cfg.router_hidden_dim),
        fusion_hidden_dim=_validate_positive_int("fusion_hidden_dim", cfg.fusion_hidden_dim),
        dropout=dropout,
    )


class PortfolioRouterModel(nn.Module):
    """
    Final portfolio router scoring K selected portfolio members.

    The trainable router produces a normalized query vector and scores it
    against K learned normalized retriever vectors.  The output is a [B, K]
    similarity matrix.

    Inputs:
        input_ids, attention_mask: T5 tokenizer outputs for question text.
        mpnet_embedding: optional [B, D_mpnet] cached query embedding.
        e5_embedding: optional [B, D_e5] cached query embedding.
        mpnet_mask, e5_mask: optional [B] masks.  If an embedding tensor is
            None, the modality is treated as all missing regardless of mask.

    T5 is always frozen.  Calling model.train() keeps T5 in eval mode, and the
    forward pass wraps T5 encoding in torch.no_grad().
    """

    def __init__(
        self,
        portfolio_size: int,
        mpnet_embedding_dim: int,
        e5_embedding_dim: int,
        *,
        t5_model_name_or_path: str = C.ROUTER_T5_DIR,
        tokenizer_name_or_path: Optional[str] = None,
        router_hidden_dim: int = 256,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = _coerce_config(
            PortfolioRouterConfig(
                portfolio_size=portfolio_size,
                mpnet_embedding_dim=mpnet_embedding_dim,
                e5_embedding_dim=e5_embedding_dim,
                t5_model_name_or_path=t5_model_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                router_hidden_dim=router_hidden_dim,
                fusion_hidden_dim=fusion_hidden_dim,
                dropout=dropout,
            )
        )

        self.t5 = T5EncoderModel.from_pretrained(self.config.t5_model_name_or_path)
        t5_dim = int(self.t5.config.d_model)
        h = self.config.router_hidden_dim

        self.text_proj = self._projection_head(t5_dim, h, dropout)
        self.mpnet_proj = self._projection_head(self.config.mpnet_embedding_dim, h, dropout)
        self.e5_proj = self._projection_head(self.config.e5_embedding_dim, h, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(3 * h, self.config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.config.fusion_hidden_dim),
            nn.Linear(self.config.fusion_hidden_dim, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(h),
        )
        self.retriever_vectors = nn.Parameter(torch.empty(self.config.portfolio_size, h))
        self._reset_parameters()

        self._freeze_t5()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.retriever_vectors, mean=0.0, std=0.02)

    @staticmethod
    def _projection_head(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def _freeze_t5(self) -> None:
        for param in self.t5.parameters():
            param.requires_grad = False
        self.t5.eval()

    def train(self, mode: bool = True) -> "PortfolioRouterModel":
        super().train(mode)
        self._freeze_t5()
        return self

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.to(dtype=hidden.dtype, device=hidden.device).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def _project_optional_embedding(
        self,
        name: str,
        embedding: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        projection: nn.Module,
        expected_dim: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        h = self.config.router_hidden_dim
        if embedding is None:
            return torch.zeros(batch_size, h, device=device)

        embedding = embedding.to(device=device, dtype=torch.float32)
        if embedding.ndim != 2 or embedding.shape[0] != batch_size or embedding.shape[1] != expected_dim:
            raise ValueError(
                f"{name} must have shape [B, {expected_dim}], got {tuple(embedding.shape)} "
                f"for B={batch_size}."
            )

        if mask is None:
            mask_f = torch.ones(batch_size, 1, device=device, dtype=torch.float32)
        else:
            mask = mask.to(device=device)
            if mask.ndim != 1 or mask.shape[0] != batch_size:
                raise ValueError(f"{name}_mask must have shape [B], got {tuple(mask.shape)}.")
            mask_f = mask.to(dtype=torch.float32).view(batch_size, 1)

        projected = projection(embedding)
        return projected * mask_f

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self._freeze_t5()
        attention_mask = attention_mask.to(device=input_ids.device)
        with torch.no_grad():
            encoded = self.t5(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            pooled = self._masked_mean(encoded, attention_mask)
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mpnet_embedding: Optional[torch.Tensor] = None,
        e5_embedding: Optional[torch.Tensor] = None,
        mpnet_mask: Optional[torch.Tensor] = None,
        e5_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        device = input_ids.device

        text_pooled = self.encode_text(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = self.text_proj(text_pooled)
        mpnet_feat = self._project_optional_embedding(
            "mpnet_embedding",
            mpnet_embedding,
            mpnet_mask,
            self.mpnet_proj,
            self.config.mpnet_embedding_dim,
            batch_size,
            device,
        )
        e5_feat = self._project_optional_embedding(
            "e5_embedding",
            e5_embedding,
            e5_mask,
            self.e5_proj,
            self.config.e5_embedding_dim,
            batch_size,
            device,
        )
        fused = self.fusion(torch.cat([text_feat, mpnet_feat, e5_feat], dim=-1))
        q_vec = F.normalize(fused, dim=-1)
        retriever_vecs = F.normalize(self.retriever_vectors, dim=-1)
        return q_vec @ retriever_vecs.t()


class PortfolioRouterTorchDataset(Dataset):
    """
    Local dataset for the final router.

    Each item contains tokenized question text plus optional query embeddings.
    After portfolio_router_collate, the batch contract is:
        input_ids: [B, T]
        attention_mask: [B, T]
        mpnet_embedding: [B, D_mpnet] when mpnet embeddings are available
        e5_embedding: [B, D_e5] when e5 embeddings are available
        mpnet_mask: [B] when mpnet embeddings are available
        e5_mask: [B] when e5 embeddings are available
        labels: [B, K] when labels are provided
        recall_k: [B, K] when recall scores are provided

    Text is tokenized lazily.  If mpnet/e5 embedding arrays are omitted, the
    collated batch omits that tensor and the model treats the modality as None.
    If arrays are present, masks default to ones unless provided.
    """

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: T5TokenizerFast,
        *,
        mpnet_embeddings: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        e5_embeddings: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        mpnet_mask: Optional[Union[np.ndarray, Sequence[float]]] = None,
        e5_mask: Optional[Union[np.ndarray, Sequence[float]]] = None,
        labels: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        recalls: Optional[Union[np.ndarray, Sequence[Sequence[float]]]] = None,
        max_length: int = 256,
    ) -> None:
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.mpnet_embeddings = _optional_float_array("mpnet_embeddings", mpnet_embeddings, len(self.texts))
        self.e5_embeddings = _optional_float_array("e5_embeddings", e5_embeddings, len(self.texts))
        self.mpnet_mask = _optional_mask_array("mpnet_mask", mpnet_mask, len(self.texts))
        self.e5_mask = _optional_mask_array("e5_mask", e5_mask, len(self.texts))
        self.labels = _optional_float_array("labels", labels, len(self.texts))
        self.recalls = _optional_recall_array(recalls, len(self.texts))
        self._token_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.texts)

    def _encode(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx not in self._token_cache:
            encoded = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding=False,
                max_length=self.max_length,
                return_tensors="pt",
            )
            self._token_cache[idx] = (encoded.input_ids[0], encoded.attention_mask[0])
        return self._token_cache[idx]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask = self._encode(idx)
        sample: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "_idx": torch.tensor(idx, dtype=torch.long),
        }
        if self.mpnet_embeddings is not None:
            sample["mpnet_embedding"] = torch.as_tensor(self.mpnet_embeddings[idx], dtype=torch.float32)
            sample["mpnet_mask"] = torch.as_tensor(
                1.0 if self.mpnet_mask is None else self.mpnet_mask[idx],
                dtype=torch.float32,
            )
        if self.e5_embeddings is not None:
            sample["e5_embedding"] = torch.as_tensor(self.e5_embeddings[idx], dtype=torch.float32)
            sample["e5_mask"] = torch.as_tensor(
                1.0 if self.e5_mask is None else self.e5_mask[idx],
                dtype=torch.float32,
            )
        if self.labels is not None:
            sample["labels"] = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        if self.recalls is not None:
            sample["recall_k"] = torch.as_tensor(self.recalls[:, idx], dtype=torch.float32)
        return sample


def _optional_float_array(
    name: str,
    value: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    expected_rows: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != expected_rows:
        raise ValueError(f"{name} must have shape [Q, D] with Q={expected_rows}; got {arr.shape}.")
    return arr


def _optional_mask_array(
    name: str,
    value: Optional[Union[np.ndarray, Sequence[float]]],
    expected_rows: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != expected_rows:
        raise ValueError(f"{name} must have shape [Q] with Q={expected_rows}; got {arr.shape}.")
    return arr


def _optional_recall_array(
    value: Optional[Union[np.ndarray, Sequence[Sequence[float]]]],
    expected_questions: int,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_questions:
        raise ValueError(
            f"recalls must have shape [K, Q] with Q={expected_questions}; got {arr.shape}."
        )
    return arr


def portfolio_router_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate PortfolioRouterTorchDataset samples.

    The forward-compatible keys are input_ids, attention_mask,
    mpnet_embedding, e5_embedding, mpnet_mask, and e5_mask.  Training helpers
    may also consume labels [B, K] and recall_k [B, K].
    """
    pad = torch.nn.utils.rnn.pad_sequence
    collated: Dict[str, torch.Tensor] = {
        "input_ids": pad([sample["input_ids"] for sample in batch], batch_first=True, padding_value=0),
        "attention_mask": pad(
            [sample["attention_mask"] for sample in batch],
            batch_first=True,
            padding_value=0,
        ),
        "_idx": torch.stack([sample["_idx"] for sample in batch]),
    }

    for key in ("mpnet_embedding", "e5_embedding", "labels", "recall_k"):
        if key in batch[0]:
            collated[key] = torch.stack([sample[key] for sample in batch])
    for key in ("mpnet_mask", "e5_mask"):
        if key in batch[0]:
            collated[key] = torch.stack([sample[key] for sample in batch]).to(dtype=torch.float32)
    return collated


def load_portfolio_router_tokenizer(
    tokenizer_name_or_path: Optional[str] = None,
) -> T5TokenizerFast:
    return T5TokenizerFast.from_pretrained(tokenizer_name_or_path or C.ROUTER_T5_DIR)


def list_trainable_parameters(model: nn.Module) -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": tuple(param.shape),
            "numel": int(param.numel()),
        }
        for name, param in model.named_parameters()
        if param.requires_grad
    ]


def assert_t5_frozen(model: PortfolioRouterModel) -> None:
    trainable = [name for name, param in model.t5.named_parameters() if param.requires_grad]
    if trainable:
        preview = ", ".join(trainable[:5])
        raise AssertionError(f"T5 has trainable parameters: {preview}")
    if model.t5.training:
        raise AssertionError("T5 must stay in eval mode.")


def filter_non_t5_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
        if not key.startswith("t5.")
    }


def assert_no_t5_checkpoint_keys(state_dict: Dict[str, Any]) -> None:
    t5_keys = [key for key in state_dict if key.startswith("t5.")]
    if t5_keys:
        preview = ", ".join(t5_keys[:5])
        raise ValueError(f"Checkpoint must not contain T5 weights; found keys: {preview}")


def build_portfolio_router_optimizer(
    model: PortfolioRouterModel,
    *,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    assert_t5_frozen(model)
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if name.startswith("t5."):
            continue
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "LayerNorm" in name or ".norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups: List[Dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "lr": lr, "weight_decay": weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("No trainable non-T5 parameters found.")
    return torch.optim.AdamW(param_groups)


def _checkpoint_config_dict(model: PortfolioRouterModel) -> Dict[str, Any]:
    cfg = asdict(model.config)
    cfg["tokenizer_name_or_path"] = model.config.resolved_tokenizer_path()
    return cfg


def save_portfolio_router_checkpoint(
    model: PortfolioRouterModel,
    path: Union[str, os.PathLike[str]],
    *,
    portfolio_metadata: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    assert_t5_frozen(model)
    state = filter_non_t5_state_dict(model.state_dict())
    assert_no_t5_checkpoint_keys(state)
    config = _checkpoint_config_dict(model)
    payload: Dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_VERSION,
        "model_state": state,
        "model_config": config,
        "portfolio_metadata": portfolio_metadata,
        "t5_model_name_or_path": model.config.t5_model_name_or_path,
        "tokenizer_name_or_path": model.config.resolved_tokenizer_path(),
        "input_embedding_dims": {
            "mpnet": model.config.mpnet_embedding_dim,
            "e5": model.config.e5_embedding_dim,
        },
        "router_hidden_dim": model.config.router_hidden_dim,
        "fusion_hidden_dim": model.config.fusion_hidden_dim,
        "portfolio_size": model.config.portfolio_size,
    }
    if extra is not None:
        payload["extra"] = extra

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_portfolio_router_checkpoint(
    path: Union[str, os.PathLike[str]],
    *,
    map_location: Union[str, torch.device] = "cpu",
    t5_model_name_or_path: Optional[str] = None,
    strict: bool = True,
) -> Tuple[PortfolioRouterModel, Dict[str, Any]]:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing a model_state dict.")
    assert_no_t5_checkpoint_keys(state)

    config_dict = dict(checkpoint.get("model_config") or {})
    if not config_dict:
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing model_config.")
    if t5_model_name_or_path is not None:
        config_dict["t5_model_name_or_path"] = t5_model_name_or_path
    config = _coerce_config(config_dict)

    model = PortfolioRouterModel(**asdict(config))
    load_result = model.load_state_dict(state, strict=False)
    non_t5_missing = [key for key in load_result.missing_keys if not key.startswith("t5.")]
    unexpected = list(load_result.unexpected_keys)
    if strict and (non_t5_missing or unexpected):
        raise RuntimeError(
            "Failed to load portfolio router checkpoint strictly: "
            f"missing_non_t5={non_t5_missing}, unexpected={unexpected}"
        )
    model._freeze_t5()
    assert_t5_frozen(model)
    return model, checkpoint

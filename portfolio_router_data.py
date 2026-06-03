"""
Manifest-aware data loading for the final portfolio router.

It reads the all-pool portfolio manifest introduced for union portfolios,
slices train recall rows from source pool score matrices, reads Goal 3
materialized test scores, and loads both mpnet/e5 query embeddings.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, replace
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import constants as C


@dataclass
class PortfolioRouterData:
    recalls: np.ndarray
    texts: List[str]
    mpnet_embeddings: Optional[np.ndarray]
    e5_embeddings: Optional[np.ndarray]
    mpnet_mask: np.ndarray
    e5_mask: np.ndarray
    selected_retrievers: List[Dict[str, Any]]
    question_datasets: List[str]
    question_indices: List[int]
    portfolio_metadata: Dict[str, Any]
    portfolio_id: str
    portfolio_path: str
    portfolio_loaded_path: str
    split: str
    num_docs: int
    labels: Optional[np.ndarray] = None
    bins: Optional[Dict[str, np.ndarray]] = None
    bin_names: Optional[List[str]] = None
    bin_membership: Optional[List[str]] = None
    total_questions: int = 0
    zero_questions: int = 0

    def __post_init__(self) -> None:
        validate_portfolio_router_data(self)


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


def _looks_like_path(value: str) -> bool:
    return (
        os.sep in value
        or value.endswith(".pickle")
        or value.endswith(".pkl")
        or value.startswith(".")
    )


def _read_pickle(expected_path: Union[str, os.PathLike[str]], *, context: str) -> Tuple[Any, str]:
    read_path = _artifact_read_path(expected_path)
    if not os.path.exists(read_path):
        raise FileNotFoundError(
            f"Missing {context}: expected_path={expected_path}, checked_path={read_path}"
        )
    with open(read_path, "rb") as f:
        return pickle.load(f), read_path


def load_portfolio_manifest(
    portfolio_path_or_id: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    num_docs_to_fetch: int = 4,
) -> Dict[str, Any]:
    """
    Load the Goals 1-2 all-pool portfolio manifest.

    If portfolio_path_or_id looks like a file path, it is read directly.
    Otherwise it is treated as an id for
    C.get_universal_portfolio_union_manifest(id, num_docs_to_fetch).
    """
    if portfolio_path_or_id is None:
        portfolio_id = C.POOL_SET_ALL_IMPLEMENTED
        expected_path = C.get_universal_portfolio_union_manifest(portfolio_id, num_docs_to_fetch)
    else:
        raw = str(portfolio_path_or_id)
        if _looks_like_path(raw):
            portfolio_id = None
            expected_path = raw
        else:
            portfolio_id = raw
            expected_path = C.get_universal_portfolio_union_manifest(portfolio_id, num_docs_to_fetch)

    manifest, read_path = _read_pickle(expected_path, context="portfolio union manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Portfolio manifest must be a dict: path={expected_path}")

    manifest_num_docs = manifest.get("num_docs")
    if manifest_num_docs is not None and int(manifest_num_docs) != int(num_docs_to_fetch):
        raise ValueError(
            f"Portfolio manifest num_docs mismatch: path={expected_path}, "
            f"manifest={manifest_num_docs}, requested={num_docs_to_fetch}"
        )

    resolved_id = (
        manifest.get("portfolio_id")
        or manifest.get("portfolio_name")
        or portfolio_id
        or Path(str(expected_path)).stem
    )
    loaded = dict(manifest)
    loaded["_portfolio_id"] = str(resolved_id)
    loaded["_portfolio_path"] = str(expected_path)
    loaded["_portfolio_loaded_path"] = str(read_path)
    return loaded


def _member_pool_label(member: Dict[str, Any]) -> str:
    return str(
        member.get("pool_id")
        or member.get("pool_label")
        or member.get("label")
        or "-"
    )


def _member_retriever(member: Dict[str, Any]) -> str:
    retriever = member.get("retriever") or member.get("family")
    if retriever not in {C.DS, C.VENDI, C.GRAPH_DENSE}:
        raise ValueError(
            f"Selected retriever has unsupported family: {retriever!r}; "
            f"pool={_member_pool_label(member)}, local_idx={member.get('local_idx', '-')}"
        )
    return str(retriever)


def _member_local_idx(member: Dict[str, Any]) -> int:
    if "local_idx" not in member:
        raise ValueError(f"Selected retriever is missing local_idx: pool={_member_pool_label(member)}")
    local_idx = int(member["local_idx"])
    if local_idx < 0:
        raise ValueError(
            f"Selected retriever has negative local_idx={local_idx}: pool={_member_pool_label(member)}"
        )
    return local_idx


def _member_artifact_embedder_key(member: Dict[str, Any]) -> str:
    retriever = _member_retriever(member)
    if retriever == C.GRAPH_DENSE:
        artifact_key = member.get("artifact_embedder_key", C.GRAPH_DENSE_MIXED_EMBEDDER_KEY)
        artifact_key = C.normalize_embedder_key(artifact_key)
        if artifact_key != C.GRAPH_DENSE_MIXED_EMBEDDER_KEY:
            raise ValueError(
                f"graph_dense selected retriever must use artifact_embedder_key="
                f"{C.GRAPH_DENSE_MIXED_EMBEDDER_KEY!r}; got {artifact_key!r}; "
                f"pool={_member_pool_label(member)}, local_idx={member.get('local_idx', '-')}"
            )
        return artifact_key

    artifact_key = member.get("artifact_embedder_key")
    if artifact_key is None:
        raise ValueError(
            f"Selected retriever is missing artifact_embedder_key: "
            f"pool={_member_pool_label(member)}, retriever={retriever}, "
            f"local_idx={member.get('local_idx', '-')}"
        )
    return C.normalize_embedder_key(artifact_key)


def _selected_retrievers_from_manifest(
    manifest: Dict[str, Any],
    *,
    portfolio_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    selected = manifest.get("selected_retrievers")
    if selected is not None:
        if not isinstance(selected, list):
            raise ValueError("Portfolio manifest selected_retrievers must be a list.")
        members = [dict(member) for member in selected]
    else:
        retriever_map = manifest.get("retriever_map")
        portfolio = manifest.get("portfolio", manifest.get("selected_global_indices"))
        if retriever_map is None or portfolio is None:
            raise ValueError(
                "Portfolio manifest must contain selected_retrievers or both "
                "retriever_map and portfolio/selected_global_indices."
            )
        members = []
        for rank, global_idx in enumerate(portfolio, start=1):
            global_idx = int(global_idx)
            if global_idx < 0 or global_idx >= len(retriever_map):
                raise ValueError(
                    f"Portfolio manifest global_idx={global_idx} is out of range "
                    f"for retriever_map size={len(retriever_map)}."
                )
            member = dict(retriever_map[global_idx])
            member["rank"] = rank
            member.setdefault("global_idx", global_idx)
            members.append(member)

    expected_size = manifest.get("actual_portfolio_size", manifest.get("portfolio_size"))
    if expected_size is not None and int(expected_size) != len(members):
        raise ValueError(
            f"Portfolio manifest size mismatch: expected {expected_size}, "
            f"selected_retrievers={len(members)}"
        )

    if portfolio_size is not None:
        k = int(portfolio_size)
        if k <= 0:
            raise ValueError(f"portfolio_size must be positive; got {portfolio_size}.")
        if k > len(members):
            raise ValueError(
                f"Requested portfolio_size={k} exceeds manifest selected size={len(members)}."
            )
        members = members[:k]

    normalized: List[Dict[str, Any]] = []
    for rank, member in enumerate(members, start=1):
        retriever = _member_retriever(member)
        artifact_key = _member_artifact_embedder_key(member)
        local_idx = _member_local_idx(member)
        normalized_member = dict(member)
        normalized_member["rank"] = int(normalized_member.get("rank", rank))
        normalized_member["portfolio_rank"] = rank - 1
        normalized_member["retriever"] = retriever
        normalized_member["family"] = normalized_member.get("family", retriever)
        normalized_member["artifact_embedder_key"] = artifact_key
        normalized_member["local_idx"] = local_idx
        normalized.append(normalized_member)
    return normalized


def _questions_path(dataset_name: str, split: str) -> str:
    if split == "train":
        return C.get_questions_train(dataset_name)
    if split == "test":
        return C.get_questions_test(dataset_name)
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


def _load_questions(dataset_name: str, split: str) -> Tuple[List[str], str, str]:
    expected_path = _questions_path(dataset_name, split)
    questions_dataset, read_path = _read_pickle(
        expected_path,
        context=f"questions file for dataset={dataset_name} split={split}",
    )
    questions = getattr(questions_dataset, "questions", None)
    if not isinstance(questions, list):
        raise ValueError(
            f"Questions payload has no list .questions: dataset={dataset_name}, "
            f"split={split}, path={expected_path}"
        )
    texts = [q["question"] for q in questions]
    return texts, expected_path, read_path


def _resolve_question_subset(
    q_count: int,
    *,
    max_questions: Optional[int],
    question_indices: Optional[Sequence[int]],
    dataset_name: str,
    split: str,
) -> Optional[np.ndarray]:
    if max_questions is not None and question_indices is not None:
        raise ValueError("Pass either max_questions or question_indices, not both.")
    if max_questions is None and question_indices is None:
        return None
    if max_questions is not None:
        limit = int(max_questions)
        if limit < 0:
            raise ValueError(f"max_questions must be non-negative; got {max_questions}.")
        return np.arange(min(limit, q_count), dtype=int)

    idx = np.asarray(question_indices, dtype=int)
    if idx.ndim != 1:
        raise ValueError(f"question_indices must be 1D, got shape={idx.shape}.")
    if idx.size and (idx.min() < 0 or idx.max() >= q_count):
        raise IndexError(
            f"question_indices out of range for dataset={dataset_name} split={split} "
            f"with Q={q_count}: min={idx.min()}, max={idx.max()}"
        )
    return idx


def _slice_optional_rows(arr: Optional[np.ndarray], idx: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None or idx is None:
        return arr
    return arr[idx]


def _score_matrix_from_payload(payload: Any, *, expected_path: str) -> np.ndarray:
    if isinstance(payload, dict):
        for key in ("scores", "recalls", "recall_matrix"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(
                f"Score payload dict must contain scores/recalls/recall_matrix: path={expected_path}"
            )
    scores = np.asarray(payload, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"Score matrix must be 2D [K, Q]: path={expected_path}, shape={scores.shape}")
    return scores


def _train_score_path(dataset_name: str, member: Dict[str, Any], num_docs_to_fetch: int) -> str:
    return C.get_retriever_scores_train(
        dataset_name,
        _member_retriever(member),
        num_docs_to_fetch,
        embedder=_member_artifact_embedder_key(member),
    )


def _load_train_scores_for_dataset(
    dataset_name: str,
    texts: Sequence[str],
    selected_retrievers: Sequence[Dict[str, Any]],
    *,
    portfolio_id: str,
    num_docs_to_fetch: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    q_count = len(texts)
    rows: List[np.ndarray] = []
    artifacts: List[Dict[str, Any]] = []
    score_cache: Dict[str, Tuple[np.ndarray, str]] = {}

    for rank, member in enumerate(selected_retrievers):
        expected_path = _train_score_path(dataset_name, member, num_docs_to_fetch)
        read_path = _artifact_read_path(expected_path)
        context = (
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, pool={_member_pool_label(member)}, "
            f"retriever={member.get('retriever', member.get('family', '-'))}, "
            f"artifact_embedder_key={member.get('artifact_embedder_key', '-')}, "
            f"local_idx={member.get('local_idx', '-')}, expected_path={expected_path}"
        )
        if expected_path not in score_cache:
            if not os.path.exists(read_path):
                raise FileNotFoundError(
                    f"Missing source pool train scores for selected retriever: {context}; "
                    f"checked_path={read_path}"
                )
            payload, loaded_path = _read_pickle(expected_path, context="source pool train scores")
            matrix = _score_matrix_from_payload(payload, expected_path=expected_path)
            if matrix.shape[1] != q_count:
                raise ValueError(
                    f"Source train score question-count mismatch: {context}; "
                    f"scores_shape={matrix.shape}, question_count={q_count}"
                )
            score_cache[expected_path] = (matrix, loaded_path)

        matrix, loaded_path = score_cache[expected_path]
        local_idx = _member_local_idx(member)
        if local_idx >= matrix.shape[0]:
            raise IndexError(
                f"Selected local retriever index out of range for train scores: {context}; "
                f"available_retrievers={matrix.shape[0]}"
            )
        rows.append(np.asarray(matrix[local_idx], dtype=np.float32))
        artifacts.append(
            {
                "dataset": dataset_name,
                "portfolio_rank": rank,
                "rank": member.get("rank", rank + 1),
                "pool_id": _member_pool_label(member),
                "retriever": _member_retriever(member),
                "artifact_embedder_key": _member_artifact_embedder_key(member),
                "local_idx": local_idx,
                "path": expected_path,
                "loaded_path": loaded_path,
            }
        )

    if rows:
        return np.stack(rows, axis=0), artifacts
    return np.zeros((0, q_count), dtype=np.float32), artifacts


def _load_test_scores_for_dataset(
    dataset_name: str,
    texts: Sequence[str],
    selected_retrievers: Sequence[Dict[str, Any]],
    *,
    portfolio_id: str,
    num_docs_to_fetch: int,
    strict: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    expected_path = C.get_portfolio_union_scores_test(portfolio_id, dataset_name, num_docs_to_fetch)
    payload, loaded_path = _read_pickle(
        expected_path,
        context=(
            f"materialized portfolio test scores for dataset={dataset_name} "
            f"portfolio_id={portfolio_id} num_docs={num_docs_to_fetch}"
        ),
    )
    scores = _score_matrix_from_payload(payload, expected_path=expected_path)
    expected_k = len(selected_retrievers)
    expected_q = len(texts)
    if scores.shape[1] != expected_q:
        raise ValueError(
            f"Materialized test score shape mismatch: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, expected_Q={expected_q}, "
            f"actual_shape={scores.shape}, path={expected_path}"
        )
    if scores.shape[0] < expected_k:
        raise ValueError(
            f"Materialized test score portfolio size is smaller than requested prefix: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, requested_K={expected_k}, "
            f"actual_shape={scores.shape}, path={expected_path}"
        )
    if scores.shape[0] > expected_k:
        scores = scores[:expected_k, :]

    metadata_path = C.get_portfolio_union_materialization_metadata(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    metadata_read_path = _artifact_read_path(metadata_path)
    metadata: Dict[str, Any] = {
        "scores_path": expected_path,
        "scores_loaded_path": loaded_path,
        "metadata_path": metadata_path,
        "metadata_loaded_path": None,
    }
    if not os.path.exists(metadata_read_path):
        message = (
            f"Missing Goal 3 materialization metadata sidecar: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, expected_path={metadata_path}, "
            f"checked_path={metadata_read_path}"
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"[portfolio-router-data] warning: {message}", flush=True)
        return scores, metadata

    sidecar, sidecar_loaded_path = _read_pickle(metadata_path, context="portfolio materialization metadata")
    if not isinstance(sidecar, dict):
        raise ValueError(f"Materialization metadata must be a dict: path={metadata_path}")
    metadata.update(sidecar)
    metadata["metadata_path"] = metadata_path
    metadata["metadata_loaded_path"] = sidecar_loaded_path
    _validate_test_materialization_metadata(
        sidecar,
        selected_retrievers,
        dataset_name=dataset_name,
        portfolio_id=portfolio_id,
        num_docs_to_fetch=num_docs_to_fetch,
        expected_k=expected_k,
        expected_q=expected_q,
        metadata_path=metadata_path,
    )
    return scores, metadata


def _member_identity(member: Dict[str, Any]) -> Tuple[Any, str, str, int]:
    return (
        member.get("global_idx"),
        _member_retriever(member),
        _member_artifact_embedder_key(member),
        _member_local_idx(member),
    )


def _validate_test_materialization_metadata(
    metadata: Dict[str, Any],
    selected_retrievers: Sequence[Dict[str, Any]],
    *,
    dataset_name: str,
    portfolio_id: str,
    num_docs_to_fetch: int,
    expected_k: int,
    expected_q: int,
    metadata_path: str,
) -> None:
    if metadata.get("dataset") not in {None, dataset_name}:
        raise ValueError(
            f"Materialization metadata dataset mismatch: expected={dataset_name}, "
            f"actual={metadata.get('dataset')}, path={metadata_path}"
        )
    if metadata.get("split") not in {None, "test"}:
        raise ValueError(f"Materialization metadata split is not test: path={metadata_path}")
    if metadata.get("portfolio_id") not in {None, portfolio_id}:
        raise ValueError(
            f"Materialization metadata portfolio_id mismatch: expected={portfolio_id}, "
            f"actual={metadata.get('portfolio_id')}, path={metadata_path}"
        )
    if metadata.get("num_docs") is not None and int(metadata["num_docs"]) != int(num_docs_to_fetch):
        raise ValueError(
            f"Materialization metadata num_docs mismatch: expected={num_docs_to_fetch}, "
            f"actual={metadata.get('num_docs')}, path={metadata_path}"
        )
    if metadata.get("recall_matrix_shape") is not None:
        shape = list(metadata["recall_matrix_shape"])
        if len(shape) != 2 or shape[0] < expected_k or shape[1] != expected_q:
            raise ValueError(
                f"Materialization metadata recall shape mismatch: "
                f"expected_at_least_K_and_Q={[expected_k, expected_q]}, "
                f"actual={shape}, path={metadata_path}"
            )
    meta_selected = metadata.get("selected_retrievers")
    if not isinstance(meta_selected, list) or len(meta_selected) < expected_k:
        raise ValueError(
            f"Materialization metadata selected_retrievers length mismatch: "
            f"expected_at_least={expected_k}, actual={len(meta_selected) if isinstance(meta_selected, list) else None}, "
            f"path={metadata_path}"
        )
    for rank, (expected_member, actual_member) in enumerate(zip(selected_retrievers, meta_selected[:expected_k])):
        if _member_identity(expected_member) != _member_identity(actual_member):
            raise ValueError(
                f"Materialized test scores are not in manifest portfolio order: "
                f"dataset={dataset_name}, portfolio_id={portfolio_id}, rank={rank}, "
                f"expected={_member_identity(expected_member)}, actual={_member_identity(actual_member)}, "
                f"path={metadata_path}"
            )


def _embedding_path(dataset_name: str, split: str, embedder: str) -> str:
    if split == "train":
        return C.get_embeddings_train(dataset_name, embedder=embedder)
    if split == "test":
        return C.get_embeddings_test(dataset_name, embedder=embedder)
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


def _load_embedding_file(
    dataset_name: str,
    split: str,
    embedder: str,
    *,
    q_count: int,
    strict: bool,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    expected_path = _embedding_path(dataset_name, split, embedder)
    read_path = _artifact_read_path(expected_path)
    artifact = {
        "dataset": dataset_name,
        "split": split,
        "embedder": embedder,
        "path": expected_path,
        "loaded_path": None,
        "available": False,
    }
    if not os.path.exists(read_path):
        message = (
            f"Missing {embedder} query embeddings: dataset={dataset_name}, split={split}, "
            f"expected_path={expected_path}, checked_path={read_path}"
        )
        if strict:
            raise FileNotFoundError(message)
        print(f"[portfolio-router-data] warning: {message}", flush=True)
        return None, artifact

    payload, loaded_path = _read_pickle(expected_path, context=f"{embedder} query embeddings")
    if not isinstance(payload, dict) or "embeddings" not in payload:
        raise ValueError(
            f"Embedding payload must be a dict with key 'embeddings': "
            f"dataset={dataset_name}, split={split}, embedder={embedder}, path={expected_path}"
        )
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"{embedder} embeddings must be 2D [Q, D]: dataset={dataset_name}, "
            f"split={split}, path={expected_path}, shape={embeddings.shape}"
        )
    if embeddings.shape[0] != q_count:
        raise ValueError(
            f"{embedder} embedding row-count mismatch: dataset={dataset_name}, split={split}, "
            f"path={expected_path}, rows={embeddings.shape[0]}, question_count={q_count}"
        )
    artifact.update(
        {
            "loaded_path": loaded_path,
            "available": True,
            "shape": list(embeddings.shape),
        }
    )
    return embeddings, artifact


def _finalize_embedding_parts(
    parts: Sequence[Optional[np.ndarray]],
    counts: Sequence[int],
    *,
    embedder: str,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    total_q = int(sum(counts))
    available = [part for part in parts if part is not None]
    if not available:
        return None, np.zeros(total_q, dtype=np.float32)

    dim = available[0].shape[1]
    for part in available:
        if part.shape[1] != dim:
            raise ValueError(
                f"{embedder} embedding dimension mismatch across datasets: "
                f"expected_dim={dim}, actual_dim={part.shape[1]}"
            )

    filled_parts: List[np.ndarray] = []
    mask_parts: List[np.ndarray] = []
    for part, q_count in zip(parts, counts):
        if part is None:
            filled_parts.append(np.zeros((q_count, dim), dtype=np.float32))
            mask_parts.append(np.zeros(q_count, dtype=np.float32))
        else:
            filled_parts.append(part)
            mask_parts.append(np.ones(q_count, dtype=np.float32))
    return np.concatenate(filled_parts, axis=0), np.concatenate(mask_parts, axis=0)


def validate_recall_matrix(
    recalls: np.ndarray,
    *,
    selected_retrievers: Sequence[Dict[str, Any]],
    q_count: int,
    context: str,
) -> None:
    if recalls.ndim != 2:
        raise ValueError(f"recalls must be 2D [K, Q]: {context}, shape={recalls.shape}")
    expected_k = len(selected_retrievers)
    if recalls.shape[0] != expected_k:
        raise ValueError(
            f"Portfolio size mismatch in recalls: {context}, expected_K={expected_k}, "
            f"actual_K={recalls.shape[0]}"
        )
    if recalls.shape[1] != q_count:
        raise ValueError(
            f"Question count mismatch in recalls: {context}, expected_Q={q_count}, "
            f"actual_Q={recalls.shape[1]}"
        )


def validate_portfolio_router_data(data: PortfolioRouterData) -> None:
    q_count = len(data.texts)
    validate_recall_matrix(
        np.asarray(data.recalls),
        selected_retrievers=data.selected_retrievers,
        q_count=q_count,
        context=f"portfolio_id={data.portfolio_id} split={data.split}",
    )
    if len(data.question_datasets) != q_count:
        raise ValueError(
            f"question_datasets length mismatch: expected={q_count}, actual={len(data.question_datasets)}"
        )
    if len(data.question_indices) != q_count:
        raise ValueError(
            f"question_indices length mismatch: expected={q_count}, actual={len(data.question_indices)}"
        )
    if data.mpnet_mask.shape != (q_count,):
        raise ValueError(f"mpnet_mask must have shape [Q], got {data.mpnet_mask.shape}")
    if data.e5_mask.shape != (q_count,):
        raise ValueError(f"e5_mask must have shape [Q], got {data.e5_mask.shape}")
    if data.mpnet_embeddings is not None and data.mpnet_embeddings.shape[0] != q_count:
        raise ValueError(
            f"mpnet_embeddings row mismatch: expected={q_count}, actual={data.mpnet_embeddings.shape[0]}"
        )
    if data.e5_embeddings is not None and data.e5_embeddings.shape[0] != q_count:
        raise ValueError(
            f"e5_embeddings row mismatch: expected={q_count}, actual={data.e5_embeddings.shape[0]}"
        )
    if data.labels is not None and data.labels.shape != (q_count, len(data.selected_retrievers)):
        raise ValueError(
            f"labels must have shape [Q, K], got {data.labels.shape}, "
            f"expected={(q_count, len(data.selected_retrievers))}"
        )


def load_portfolio_router_data(
    portfolio_path_or_id: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    split: str,
    datasets: Optional[Sequence[str]] = None,
    num_docs_to_fetch: int = 4,
    portfolio_size: Optional[int] = None,
    max_questions: Optional[int] = None,
    question_indices: Optional[Sequence[int]] = None,
    strict: bool = True,
) -> PortfolioRouterData:
    """
    Load final-router inputs for an all-pool portfolio.

    Train recalls are sliced from source pool train score matrices in manifest
    portfolio order.  Test recalls are read from Goal 3 materialized portfolio
    test score files.

    max_questions and question_indices are optional per-dataset smoke-test
    controls.  They are applied after each dataset's questions, scores, and
    embeddings have been loaded and validated against the full artifact shape.
    Defaults preserve full-dataset behavior.
    """
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    dataset_list = list(C.DATASETS if datasets is None else datasets)
    if not dataset_list:
        raise ValueError("No datasets provided for portfolio router data loading.")

    manifest = load_portfolio_manifest(
        portfolio_path_or_id,
        num_docs_to_fetch=num_docs_to_fetch,
    )
    portfolio_id = manifest["_portfolio_id"]
    selected_retrievers = _selected_retrievers_from_manifest(
        manifest,
        portfolio_size=portfolio_size,
    )
    if not selected_retrievers:
        raise ValueError(f"Portfolio manifest has no selected retrievers: {manifest['_portfolio_path']}")

    print(
        f"[portfolio-router-data] loading split={split} portfolio_id={portfolio_id} "
        f"K={len(selected_retrievers)} datasets={','.join(dataset_list)}",
        flush=True,
    )

    recalls_parts: List[np.ndarray] = []
    text_parts: List[str] = []
    question_datasets: List[str] = []
    question_indices_all: List[int] = []
    mpnet_parts: List[Optional[np.ndarray]] = []
    e5_parts: List[Optional[np.ndarray]] = []
    q_counts: List[int] = []
    dataset_metadata: List[Dict[str, Any]] = []

    for dataset_name in dataset_list:
        texts, questions_path, questions_loaded_path = _load_questions(dataset_name, split)
        full_q_count = len(texts)
        if split == "train":
            recalls_ds, score_artifacts = _load_train_scores_for_dataset(
                dataset_name,
                texts,
                selected_retrievers,
                portfolio_id=portfolio_id,
                num_docs_to_fetch=num_docs_to_fetch,
            )
            score_metadata: Dict[str, Any] = {"source_score_artifacts": score_artifacts}
        else:
            recalls_ds, score_metadata = _load_test_scores_for_dataset(
                dataset_name,
                texts,
                selected_retrievers,
                portfolio_id=portfolio_id,
                num_docs_to_fetch=num_docs_to_fetch,
                strict=strict,
            )
        validate_recall_matrix(
            recalls_ds,
            selected_retrievers=selected_retrievers,
            q_count=full_q_count,
            context=f"dataset={dataset_name} split={split} portfolio_id={portfolio_id}",
        )

        mpnet_embeddings, mpnet_artifact = _load_embedding_file(
            dataset_name,
            split,
            C.DEFAULT_EMBEDDER_KEY,
            q_count=full_q_count,
            strict=strict,
        )
        e5_embeddings, e5_artifact = _load_embedding_file(
            dataset_name,
            split,
            C.E5_EMBEDDER_KEY,
            q_count=full_q_count,
            strict=strict,
        )

        subset_idx = _resolve_question_subset(
            full_q_count,
            max_questions=max_questions,
            question_indices=question_indices,
            dataset_name=dataset_name,
            split=split,
        )
        if subset_idx is not None:
            texts = [texts[i] for i in subset_idx]
            recalls_ds = recalls_ds[:, subset_idx]
            mpnet_embeddings = _slice_optional_rows(mpnet_embeddings, subset_idx)
            e5_embeddings = _slice_optional_rows(e5_embeddings, subset_idx)
            original_question_indices = subset_idx.tolist()
        else:
            original_question_indices = list(range(full_q_count))
        q_count = len(texts)

        recalls_parts.append(recalls_ds)
        text_parts.extend(texts)
        question_datasets.extend([dataset_name] * q_count)
        question_indices_all.extend(original_question_indices)
        mpnet_parts.append(mpnet_embeddings)
        e5_parts.append(e5_embeddings)
        q_counts.append(q_count)
        dataset_metadata.append(
            {
                "dataset": dataset_name,
                "split": split,
                "num_questions": q_count,
                "full_num_questions": full_q_count,
                "question_indices": original_question_indices,
                "questions_path": questions_path,
                "questions_loaded_path": questions_loaded_path,
                "scores": score_metadata,
                "mpnet_embeddings": mpnet_artifact,
                "e5_embeddings": e5_artifact,
            }
        )
        print(
            f"[portfolio-router-data] loaded dataset={dataset_name} split={split} "
            f"questions={q_count} recalls_shape={recalls_ds.shape}",
            flush=True,
        )

    recalls = np.concatenate(recalls_parts, axis=1).astype(np.float32, copy=False)
    mpnet_embeddings_all, mpnet_mask = _finalize_embedding_parts(
        mpnet_parts,
        q_counts,
        embedder=C.DEFAULT_EMBEDDER_KEY,
    )
    e5_embeddings_all, e5_mask = _finalize_embedding_parts(
        e5_parts,
        q_counts,
        embedder=C.E5_EMBEDDER_KEY,
    )

    metadata = {
        "manifest": {
            "schema": manifest.get("schema"),
            "schema_version": manifest.get("schema_version"),
            "portfolio_id": portfolio_id,
            "portfolio_path": manifest["_portfolio_path"],
            "portfolio_loaded_path": manifest["_portfolio_loaded_path"],
            "num_docs": int(num_docs_to_fetch),
        },
        "datasets": dataset_metadata,
    }
    return PortfolioRouterData(
        recalls=recalls,
        texts=text_parts,
        mpnet_embeddings=mpnet_embeddings_all,
        e5_embeddings=e5_embeddings_all,
        mpnet_mask=mpnet_mask,
        e5_mask=e5_mask,
        selected_retrievers=selected_retrievers,
        question_datasets=question_datasets,
        question_indices=question_indices_all,
        portfolio_metadata=metadata,
        portfolio_id=portfolio_id,
        portfolio_path=manifest["_portfolio_path"],
        portfolio_loaded_path=manifest["_portfolio_loaded_path"],
        split=split,
        num_docs=int(num_docs_to_fetch),
        total_questions=len(text_parts),
        zero_questions=int(np.all(recalls <= 1e-10, axis=0).sum()),
    )


def load_portfolio_router_train_data(
    portfolio_path_or_id: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    datasets: Optional[Sequence[str]] = None,
    num_docs_to_fetch: int = 4,
    portfolio_size: Optional[int] = None,
    max_questions: Optional[int] = None,
    question_indices: Optional[Sequence[int]] = None,
    strict: bool = True,
) -> PortfolioRouterData:
    return load_portfolio_router_data(
        portfolio_path_or_id,
        split="train",
        datasets=datasets,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        max_questions=max_questions,
        question_indices=question_indices,
        strict=strict,
    )


def load_portfolio_router_test_data(
    portfolio_path_or_id: Optional[Union[str, os.PathLike[str]]] = None,
    *,
    datasets: Optional[Sequence[str]] = None,
    num_docs_to_fetch: int = 4,
    portfolio_size: Optional[int] = None,
    max_questions: Optional[int] = None,
    question_indices: Optional[Sequence[int]] = None,
    strict: bool = True,
) -> PortfolioRouterData:
    return load_portfolio_router_data(
        portfolio_path_or_id,
        split="test",
        datasets=datasets,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        max_questions=max_questions,
        question_indices=question_indices,
        strict=strict,
    )


def _compute_labels_and_bins(
    recalls: np.ndarray,
    *,
    mode: str = "argmax",
    temperature: float = 1.0,
    break_ties_to_lowest: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], List[str]]:
    scores = np.asarray(recalls, dtype=np.float32)
    if scores.ndim != 2:
        raise ValueError(f"recalls must be 2D [K, Q], got shape={scores.shape}")
    k, q = scores.shape
    zero = 1e-10
    all_zero = np.all(scores <= zero, axis=0)

    if mode == "argmax":
        labels = np.zeros((q, k), dtype=np.float32)
        row_scores = scores.T
        row_max = row_scores.max(axis=1, keepdims=True)
        if break_ties_to_lowest:
            valid = row_max.squeeze(1) > zero
            if np.any(valid):
                valid_rows = np.where(valid)[0]
                winners = np.argmax(row_scores[valid_rows, :], axis=1)
                labels[valid_rows, winners] = 1.0
            winners_mask = labels.astype(bool)
            tie_count = np.zeros(q, dtype=int)
            tie_count[valid] = 1
        else:
            winners_mask = (np.abs(row_scores - row_max) <= 1e-12) & (row_max > zero)
            tie_count = winners_mask.sum(axis=1)
            labels[winners_mask] = 1.0
            denom = labels.sum(axis=1, keepdims=True).clip(min=1.0)
            labels = labels / denom
    elif mode == "softmax":
        if temperature <= 0:
            raise ValueError(f"temperature must be positive for softmax labels; got {temperature}.")
        col_max = scores.max(axis=0, keepdims=True)
        logits = (scores - col_max) / float(temperature)
        exp = np.exp(logits)
        labels = (exp / (exp.sum(axis=0, keepdims=True) + 1e-12)).T.astype(np.float32)
        labels[all_zero, :] = 0.0
        row_scores = scores.T
        row_max = row_scores.max(axis=1, keepdims=True)
        winners_mask = (np.abs(row_scores - row_max) <= 1e-12) & (row_max > zero)
        tie_count = winners_mask.sum(axis=1)
    else:
        raise ValueError("mode must be 'argmax' or 'softmax'.")

    bin_names = [f"R{rank}_unique" for rank in range(k)]
    if k >= 2 and not break_ties_to_lowest:
        bin_names.append("tie2")
    if k >= 3 and not break_ties_to_lowest:
        bin_names.append("tie3")

    bins: Dict[str, np.ndarray] = {}
    for rank in range(k):
        bins[f"R{rank}_unique"] = np.where((~all_zero) & (tie_count == 1) & winners_mask[:, rank])[0]
    if "tie2" in bin_names:
        bins["tie2"] = np.where((~all_zero) & (tie_count == 2))[0]
    if "tie3" in bin_names:
        bins["tie3"] = np.where((~all_zero) & (tie_count == 3))[0]

    membership = np.array(["zero_recall" if z else "unassigned" for z in all_zero], dtype=object)
    for name, idxs in bins.items():
        membership[idxs] = name
    return labels, bin_names, bins, membership.tolist()


def with_portfolio_router_labels(
    data: PortfolioRouterData,
    *,
    mode: str = "argmax",
    temperature: float = 1.0,
    break_ties_to_lowest: bool = False,
    zero_recall_policy: str = "keep",
) -> PortfolioRouterData:
    """
    Attach [Q, K] labels for training.

    zero_recall_policy='keep' keeps zero-recall questions with all-zero labels.
    zero_recall_policy='drop' removes them before labels are computed.
    """
    if zero_recall_policy not in {"keep", "drop"}:
        raise ValueError("zero_recall_policy must be 'keep' or 'drop'.")
    working = data
    if zero_recall_policy == "drop":
        keep = np.where(~np.all(data.recalls <= 1e-10, axis=0))[0]
        working = subset_portfolio_router_data(data, keep)

    labels, bin_names, bins, membership = _compute_labels_and_bins(
        working.recalls,
        mode=mode,
        temperature=temperature,
        break_ties_to_lowest=break_ties_to_lowest,
    )
    return replace(
        working,
        labels=labels,
        bin_names=bin_names,
        bins=bins,
        bin_membership=membership,
        total_questions=len(working.texts),
        zero_questions=int(np.all(working.recalls <= 1e-10, axis=0).sum()),
    )


def _reindex_bins_for_subset(
    bins_full: Dict[str, np.ndarray],
    subset_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    if len(subset_idx) == 0:
        return {name: np.array([], dtype=int) for name in bins_full}

    new_index = -np.ones(subset_idx.max() + 1, dtype=int)
    new_index[subset_idx] = np.arange(len(subset_idx))

    remapped: Dict[str, np.ndarray] = {}
    for name, idxs in bins_full.items():
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) == 0:
            remapped[name] = np.array([], dtype=int)
            continue
        mask = np.isin(idxs, subset_idx)
        if not np.any(mask):
            remapped[name] = np.array([], dtype=int)
            continue
        remapped[name] = new_index[idxs[mask]]
    return remapped


def _membership_from_bins(bins: Dict[str, np.ndarray], length: int) -> List[str]:
    membership = np.array(["unassigned"] * length, dtype=object)
    for name, idxs in bins.items():
        membership[np.asarray(idxs, dtype=int)] = name
    return membership.tolist()


def subset_portfolio_router_data(
    data: PortfolioRouterData,
    indices: Union[np.ndarray, Sequence[int]],
) -> PortfolioRouterData:
    idx = np.asarray(indices, dtype=int)
    if idx.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape={idx.shape}")
    q_count = len(data.texts)
    if idx.size and (idx.min() < 0 or idx.max() >= q_count):
        raise IndexError(f"indices out of range for Q={q_count}: min={idx.min()}, max={idx.max()}")

    def maybe_rows(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return None if arr is None else arr[idx]

    labels = None if data.labels is None else data.labels[idx]
    bins = None
    bin_names = data.bin_names
    bin_membership = None
    if data.bins is not None:
        bins = _reindex_bins_for_subset(data.bins, idx)
        if bin_names is not None:
            bins = {name: bins.get(name, np.array([], dtype=int)) for name in bin_names}
        bin_membership = _membership_from_bins(bins, int(idx.size))
    elif data.bin_membership is not None:
        bin_membership = [data.bin_membership[i] for i in idx]
    metadata = dict(data.portfolio_metadata)
    metadata["subset_from_questions"] = q_count
    metadata["subset_indices"] = idx.tolist()
    return PortfolioRouterData(
        recalls=data.recalls[:, idx],
        texts=[data.texts[i] for i in idx],
        mpnet_embeddings=maybe_rows(data.mpnet_embeddings),
        e5_embeddings=maybe_rows(data.e5_embeddings),
        mpnet_mask=data.mpnet_mask[idx],
        e5_mask=data.e5_mask[idx],
        selected_retrievers=data.selected_retrievers,
        question_datasets=[data.question_datasets[i] for i in idx],
        question_indices=[data.question_indices[i] for i in idx],
        portfolio_metadata=metadata,
        portfolio_id=data.portfolio_id,
        portfolio_path=data.portfolio_path,
        portfolio_loaded_path=data.portfolio_loaded_path,
        split=data.split,
        num_docs=data.num_docs,
        labels=labels,
        bins=bins,
        bin_names=bin_names,
        bin_membership=bin_membership,
        total_questions=int(idx.size),
        zero_questions=int(np.all(data.recalls[:, idx] <= 1e-10, axis=0).sum()) if idx.size else 0,
    )


def _compute_portfolio_router_soft_targets_and_bins(
    scores: np.ndarray,
    mode: str,
    temperature: float,
    break_ties_to_lowest: bool = False,
) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray], np.ndarray]:
    zero = 1e-10
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array [K, Q].")
    k, q = scores.shape

    labels = np.zeros((q, k), dtype=np.float64)
    all_zero = np.all(scores <= zero, axis=0)

    if mode == "argmax":
        for q_idx in range(q):
            if all_zero[q_idx]:
                continue
            q_scores = scores[:, q_idx]
            max_score = q_scores.max()
            winners = np.flatnonzero(np.abs(q_scores - max_score) <= zero)
            if winners.size == 0:
                continue
            if break_ties_to_lowest:
                labels[q_idx, int(winners[0])] = 1.0
            else:
                labels[q_idx, winners] = 1.0 / len(winners)
    elif mode == "softmax":
        col_max = scores.max(axis=0, keepdims=True)
        logits = (scores - col_max) / temperature
        exp = np.exp(logits)
        exp_sum = exp.sum(axis=0, keepdims=True) + 1e-12
        labels = (exp / exp_sum).T
        labels[all_zero, :] = 0.0
    else:
        raise ValueError("mode must be 'softmax' or 'argmax'.")

    scores_t = scores.T
    row_max = scores_t.max(axis=1, keepdims=True)
    if break_ties_to_lowest:
        winners_mask = np.zeros_like(scores_t, dtype=bool)
        valid = row_max.squeeze(1) > 0
        if np.any(valid):
            valid_rows = np.where(valid)[0]
            winner_cols = np.argmax(scores_t[valid_rows, :], axis=1)
            winners_mask[valid_rows, winner_cols] = True
        tie_count = np.zeros(scores_t.shape[0], dtype=int)
        tie_count[valid] = 1
    else:
        winners_mask = (np.abs(scores_t - row_max) <= 1e-12) & (row_max > 0)
        tie_count = winners_mask.sum(axis=1)

    bin_names = [f"R{rank}_unique" for rank in range(k)]
    if k >= 2 and not break_ties_to_lowest:
        bin_names.append("tie2")
    if k >= 3 and not break_ties_to_lowest:
        bin_names.append("tie3")

    bins_full: Dict[str, np.ndarray] = {}
    for rank in range(k):
        bins_full[f"R{rank}_unique"] = np.where(
            (~all_zero) & (tie_count == 1) & winners_mask[:, rank]
        )[0]
    if "tie2" in bin_names:
        bins_full["tie2"] = np.where((~all_zero) & (tie_count == 2))[0]
    if "tie3" in bin_names:
        bins_full["tie3"] = np.where((~all_zero) & (tie_count == 3))[0]

    return labels.astype(np.float32), bin_names, bins_full, all_zero


def train_dev_split_portfolio_router_data(
    data: PortfolioRouterData,
    *,
    dev_ratio: float = 0.1,
    random_seed: int = 0,
    shuffle: bool = True,
    zero_recall_policy: str = "keep",
    label_mode: str = "argmax",
    temperature: float = 1.0,
    tie_2_fraction_of_unique: float = 0.5,
    tie_3_fraction_of_unique: float = 0.1,
    break_ties_to_lowest: bool = False,
) -> Tuple[PortfolioRouterData, PortfolioRouterData]:
    """
    Non-zero unique-winner examples are split per retriever bin.  tie2/tie3
    examples are downsampled into train relative to the number of unique train
    examples, and all-zero plus tie>3 questions are not materialized in either
    split.  With zero_recall_policy='keep', all-zero counts are retained in
    total_questions/zero_questions to mirror the historical router label/split
    convention; with 'drop', they are omitted from split counts as well.
    """
    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in [0, 1), got {dev_ratio}.")
    if zero_recall_policy not in {"keep", "drop"}:
        raise ValueError("zero_recall_policy must be 'keep' or 'drop'.")
    scores = np.asarray(data.recalls, dtype=np.float64)
    labels, bin_names, bins_full, all_zero = _compute_portfolio_router_soft_targets_and_bins(
        scores,
        label_mode,
        temperature,
        break_ties_to_lowest=break_ties_to_lowest,
    )
    k = scores.shape[0]
    total_questions = scores.shape[1]
    total_zero = int(all_zero.sum())
    rng = np.random.RandomState(random_seed)

    def ordered_indices(idxs: np.ndarray) -> np.ndarray:
        idxs = np.asarray(idxs, dtype=int)
        return rng.permutation(idxs) if shuffle else idxs

    train_parts: List[np.ndarray] = []
    dev_parts: List[np.ndarray] = []
    for name in [f"R{rank}_unique" for rank in range(k)]:
        idxs = bins_full[name]
        if len(idxs) == 0:
            continue
        perm = ordered_indices(idxs)
        n_dev = int(round(len(idxs) * dev_ratio))
        dev_parts.append(perm[:n_dev])
        train_parts.append(perm[n_dev:])

    sum_unique_qs = int(sum(len(arr) for arr in train_parts))

    if "tie2" in bins_full:
        tie2_pool = bins_full["tie2"]
        target_n2 = min(
            int(round(sum_unique_qs * tie_2_fraction_of_unique)),
            floor((1 - dev_ratio) * len(tie2_pool)),
        )
        if len(tie2_pool) > 0:
            perm2 = ordered_indices(tie2_pool)
            train_parts.append(perm2[:target_n2])
            dev_parts.append(perm2[target_n2:])

    if "tie3" in bins_full:
        tie3_pool = bins_full["tie3"]
        target_n3 = min(
            int(round(sum_unique_qs * tie_3_fraction_of_unique)),
            floor((1 - dev_ratio) * len(tie3_pool)),
        )
        if len(tie3_pool) > 0:
            perm3 = ordered_indices(tie3_pool)
            train_parts.append(perm3[:target_n3])
            dev_parts.append(perm3[target_n3:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    dev_idx = np.sort(np.concatenate(dev_parts)) if dev_parts else np.array([], dtype=int)

    nonzero_total = total_questions - total_zero
    train_nonzero = int(len(train_idx))
    dev_nonzero = int(len(dev_idx))
    dev_frac = dev_nonzero / float(nonzero_total) if nonzero_total > 0 else 0.0
    if zero_recall_policy == "keep":
        zero_dev = int(round(total_zero * dev_frac))
        zero_train = total_zero - zero_dev
    else:
        zero_dev = 0
        zero_train = 0

    def build_split(indices: np.ndarray, total_q: int, zero_q: int) -> PortfolioRouterData:
        split = subset_portfolio_router_data(data, indices)
        split_bins = _reindex_bins_for_subset(bins_full, indices)
        split_bins = {name: split_bins.get(name, np.array([], dtype=int)) for name in bin_names}
        metadata = dict(split.portfolio_metadata)
        metadata["portfolio_router_split"] = {
            "random_seed": int(random_seed),
            "dev_ratio": float(dev_ratio),
            "label_mode": label_mode,
            "temperature": float(temperature),
            "tie_2_fraction_of_unique": float(tie_2_fraction_of_unique),
            "tie_3_fraction_of_unique": float(tie_3_fraction_of_unique),
            "break_ties_to_lowest": bool(break_ties_to_lowest),
            "shuffle": bool(shuffle),
            "zero_recall_policy": zero_recall_policy,
            "source_total_questions": int(total_questions),
            "source_zero_questions": int(total_zero),
        }
        return replace(
            split,
            labels=labels[indices].astype(np.float32, copy=False),
            bins=split_bins,
            bin_names=list(bin_names),
            bin_membership=_membership_from_bins(split_bins, int(len(indices))),
            portfolio_metadata=metadata,
            total_questions=int(total_q),
            zero_questions=int(zero_q),
        )

    return (
        build_split(train_idx, train_nonzero + zero_train, zero_train),
        build_split(dev_idx, dev_nonzero + zero_dev, zero_dev),
    )


def simple_train_dev_split_portfolio_router_data(
    data: PortfolioRouterData,
    *,
    dev_ratio: float = 0.1,
    random_seed: int = 0,
    shuffle: bool = True,
    zero_recall_policy: str = "keep",
    label_mode: str = "argmax",
    temperature: float = 1.0,
    break_ties_to_lowest: bool = False,
) -> Tuple[PortfolioRouterData, PortfolioRouterData]:
    """
    Build a simple train/dev split and attach labels to both splits.
    """
    if not 0.0 <= dev_ratio < 1.0:
        raise ValueError(f"dev_ratio must be in [0, 1), got {dev_ratio}.")
    labeled = with_portfolio_router_labels(
        data,
        mode=label_mode,
        temperature=temperature,
        break_ties_to_lowest=break_ties_to_lowest,
        zero_recall_policy=zero_recall_policy,
    )
    q_count = len(labeled.texts)
    indices = np.arange(q_count)
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(indices)
    n_dev = int(round(q_count * dev_ratio))
    dev_idx = np.sort(indices[:n_dev])
    train_idx = np.sort(indices[n_dev:])
    train_data = with_portfolio_router_labels(
        subset_portfolio_router_data(labeled, train_idx),
        mode=label_mode,
        temperature=temperature,
        break_ties_to_lowest=break_ties_to_lowest,
        zero_recall_policy="keep",
    )
    dev_data = with_portfolio_router_labels(
        subset_portfolio_router_data(labeled, dev_idx),
        mode=label_mode,
        temperature=temperature,
        break_ties_to_lowest=break_ties_to_lowest,
        zero_recall_policy="keep",
    )
    return train_data, dev_data


def make_portfolio_router_torch_dataset(
    data: PortfolioRouterData,
    tokenizer: Any,
    *,
    max_length: int = 256,
):
    """
    Convert loaded data to portfolio_router.PortfolioRouterTorchDataset.

    The import is local so this data module can be inspected without loading
    torch/transformers until a training loop needs them.
    """
    from portfolio_router import PortfolioRouterTorchDataset

    return PortfolioRouterTorchDataset(
        data.texts,
        tokenizer,
        mpnet_embeddings=data.mpnet_embeddings,
        e5_embeddings=data.e5_embeddings,
        mpnet_mask=data.mpnet_mask,
        e5_mask=data.e5_mask,
        labels=data.labels,
        recalls=data.recalls,
        max_length=max_length,
    )


def portfolio_router_forward_kwargs(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract the PortfolioRouterModel.forward-compatible tensors from a batch.
    """
    keys = (
        "input_ids",
        "attention_mask",
        "mpnet_embedding",
        "e5_embedding",
        "mpnet_mask",
        "e5_mask",
    )
    return {key: batch[key] for key in keys if key in batch}


def validate_portfolio_router_batch(
    batch: Dict[str, Any],
    *,
    portfolio_size: int,
    mpnet_embedding_dim: Optional[int] = None,
    e5_embedding_dim: Optional[int] = None,
    require_labels: bool = False,
) -> Dict[str, Tuple[int, ...]]:
    """
    Validate a collated final-router batch.

    Expected keys:
        input_ids: [B, T]
        attention_mask: [B, T]
        mpnet_embedding: [B, D_mpnet] when present
        e5_embedding: [B, D_e5] when present
        mpnet_mask: [B] when mpnet_embedding is present
        e5_mask: [B] when e5_embedding is present
        labels: [B, K] when labels are requested/provided
        recall_k: [B, K] when recalls are provided
    """
    for key in ("input_ids", "attention_mask"):
        if key not in batch:
            raise ValueError(f"Batch is missing required key: {key}")
    shapes = {key: tuple(value.shape) for key, value in batch.items() if hasattr(value, "shape")}
    input_shape = shapes["input_ids"]
    attention_shape = shapes["attention_mask"]
    if len(input_shape) != 2:
        raise ValueError(f"input_ids must be [B, T], got {input_shape}")
    if attention_shape != input_shape:
        raise ValueError(f"attention_mask shape must match input_ids: {attention_shape} vs {input_shape}")
    batch_size = input_shape[0]

    def check_2d(name: str, expected_dim: Optional[int]) -> None:
        if name not in batch:
            return
        shape = shapes[name]
        if len(shape) != 2 or shape[0] != batch_size:
            raise ValueError(f"{name} must be [B, D], got {shape}")
        if expected_dim is not None and shape[1] != expected_dim:
            raise ValueError(f"{name} dim mismatch: expected={expected_dim}, actual={shape[1]}")

    def check_mask(name: str) -> None:
        if name not in batch:
            return
        shape = shapes[name]
        if shape != (batch_size,):
            raise ValueError(f"{name} must be [B], got {shape}")

    check_2d("mpnet_embedding", mpnet_embedding_dim)
    check_2d("e5_embedding", e5_embedding_dim)
    check_mask("mpnet_mask")
    check_mask("e5_mask")
    if "mpnet_embedding" in batch and "mpnet_mask" not in batch:
        raise ValueError("Batch has mpnet_embedding but is missing mpnet_mask.")
    if "e5_embedding" in batch and "e5_mask" not in batch:
        raise ValueError("Batch has e5_embedding but is missing e5_mask.")
    for key in ("labels", "recall_k"):
        if key not in batch:
            if key == "labels" and require_labels:
                raise ValueError("Batch is missing labels.")
            continue
        shape = shapes[key]
        if shape != (batch_size, portfolio_size):
            raise ValueError(f"{key} must be [B, K], got {shape}, expected={(batch_size, portfolio_size)}")
    return shapes


class _SmokeTokenizerOutput:
    def __init__(self, input_ids: Any, attention_mask: Any) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _WhitespaceSmokeTokenizer:
    """
    Tiny tokenizer for data-path smoke tests.

    It avoids loading T5-large while still producing tokenizer-shaped
    input_ids/attention_mask tensors for PortfolioRouterTorchDataset.
    """

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = True,
        padding: bool = False,
        max_length: int = 256,
        return_tensors: str = "pt",
    ) -> _SmokeTokenizerOutput:
        del padding
        if return_tensors != "pt":
            raise ValueError("_WhitespaceSmokeTokenizer only supports return_tensors='pt'.")
        import torch

        pieces = str(text).split()
        if truncation:
            pieces = pieces[:max_length]
        token_ids = [min(abs(hash(piece)) % 32000, 31999) + 1 for piece in pieces]
        if not token_ids:
            token_ids = [1]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return _SmokeTokenizerOutput(input_ids=input_ids, attention_mask=attention_mask)


def smoke_portfolio_router_batch(
    portfolio_path_or_id: Optional[Union[str, os.PathLike[str]]] = C.POOL_SET_ALL_IMPLEMENTED,
    *,
    dataset: str = C.HotpotQA,
    split: str = "train",
    num_docs_to_fetch: int = 4,
    portfolio_size: int = 2,
    max_questions: int = 8,
    batch_size: int = 4,
    strict: bool = True,
    dev_ratio: float = 0.25,
    max_length: int = 64,
) -> Dict[str, Any]:
    """
    Read-only smoke test for the final-router data path.

    The helper loads a small dataset subset, attaches argmax labels, optionally
    creates a train/dev split, converts one split to PortfolioRouterTorchDataset,
    collates a small batch, and validates that batch keys/shapes match
    PortfolioRouterModel.forward plus training labels/recalls.
    """
    split = split.lower()
    data = load_portfolio_router_data(
        portfolio_path_or_id,
        split=split,
        datasets=[dataset],
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        max_questions=max_questions,
        strict=strict,
    )
    labeled = with_portfolio_router_labels(data, mode="argmax", zero_recall_policy="keep")
    dev_data = None
    batch_data = labeled
    if split == "train" and len(labeled.texts) > 1:
        train_data, dev_data = train_dev_split_portfolio_router_data(
            labeled,
            dev_ratio=dev_ratio,
            random_seed=0,
            label_mode="argmax",
            zero_recall_policy="keep",
        )
        batch_data = train_data if len(train_data.texts) > 0 else dev_data

    tokenizer = _WhitespaceSmokeTokenizer()
    torch_dataset = make_portfolio_router_torch_dataset(
        batch_data,
        tokenizer,
        max_length=max_length,
    )
    if len(torch_dataset) == 0:
        raise ValueError("Smoke dataset is empty after subsetting/splitting.")
    from portfolio_router import portfolio_router_collate

    n = min(int(batch_size), len(torch_dataset))
    batch = portfolio_router_collate([torch_dataset[i] for i in range(n)])
    mpnet_dim = None if batch_data.mpnet_embeddings is None else batch_data.mpnet_embeddings.shape[1]
    e5_dim = None if batch_data.e5_embeddings is None else batch_data.e5_embeddings.shape[1]
    batch_shapes = validate_portfolio_router_batch(
        batch,
        portfolio_size=len(batch_data.selected_retrievers),
        mpnet_embedding_dim=mpnet_dim,
        e5_embedding_dim=e5_dim,
        require_labels=True,
    )
    forward_kwargs = portfolio_router_forward_kwargs(batch)
    return {
        "dataset": dataset,
        "split": split,
        "portfolio_id": data.portfolio_id,
        "portfolio_size": len(data.selected_retrievers),
        "loaded_questions": len(data.texts),
        "zero_questions": labeled.zero_questions,
        "recalls_shape": tuple(data.recalls.shape),
        "labels_shape": tuple(labeled.labels.shape) if labeled.labels is not None else None,
        "train_questions": len(batch_data.texts) if split == "train" else None,
        "dev_questions": len(dev_data.texts) if dev_data is not None else None,
        "batch_shapes": batch_shapes,
        "forward_keys": sorted(forward_kwargs),
    }

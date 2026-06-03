import pickle
from pathlib import Path

import constants as C
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from transformers import AutoTokenizer

from utils import exact_match_score, extract_tagged_answer, f1_score, f1_support
from vendi_rag import evaluate_vendirag_early_stopping

# Shared Plot Helpers

def _progress_bar(idx, total, prefix):
    width = 24
    if total <= 0:
        total = 1
    filled = int(width * idx / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {idx}/{total}", end="", flush=True)
    if idx >= total:
        print("", flush=True)

def _translated_artifact_path(path):
    return Path(path)

def _artifact_read_path(path):
    expected_path = Path(path)
    if expected_path.exists():
        return expected_path
    translated_path = _translated_artifact_path(expected_path)
    if translated_path != expected_path and translated_path.exists():
        return translated_path
    return expected_path

def _artifact_write_path(path):
    expected_path = Path(path)
    if expected_path.parent.exists():
        return expected_path
    return expected_path

def _load_pickle_artifact(path, label):
    read_path = _artifact_read_path(path)
    if not read_path.exists():
        raise FileNotFoundError(
            f"Missing {label}: expected_path={path}, checked_path={read_path}"
        )
    with read_path.open("rb") as f:
        return pickle.load(f), read_path

def _save_pickle_artifact(path, payload):
    write_path = _artifact_write_path(path)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    with write_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return write_path

def _safe_plot_name(value):
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("@", "_at_")
        .replace("=", "")
        .replace(",", "_")
    )

# All-Pool Portfolio Support Recall/F1 Plot

def _load_test_questions_for_support(dataset_name):
    questions_path = C.get_questions_test(dataset_name)
    payload, read_path = _load_pickle_artifact(
        questions_path,
        f"test questions for dataset={dataset_name}",
    )
    if not hasattr(payload, "questions"):
        raise ValueError(
            f"Test questions payload must have .questions for dataset={dataset_name}: "
            f"expected_path={questions_path}, loaded_path={read_path}"
        )
    return payload.questions

def _support_gold_docs(question):
    return question.get("target") or question.get("support_docs") or []

def _retrieved_doc_id(text_unit):
    doc_id = getattr(text_unit, "doc_id", None)
    if doc_id is not None:
        return doc_id
    if isinstance(text_unit, dict):
        return text_unit.get("doc_id") or text_unit.get("id")
    return None

def _retrieved_doc_ids(retrieved_units):
    return [
        doc_id
        for doc_id in (_retrieved_doc_id(text_unit) for text_unit in retrieved_units or [])
        if doc_id is not None
    ]

def _as_score_matrix(value, label, expected_questions=None, min_rows=None):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a 2D score matrix, got {type(value).__name__}")
    matrix = []
    for row_idx, row in enumerate(value):
        if hasattr(row, "tolist"):
            row = row.tolist()
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{label} row {row_idx} is not a list-like sequence")
        row_values = [float(score) for score in row]
        if expected_questions is not None and len(row_values) != expected_questions:
            raise ValueError(
                f"{label} row {row_idx} has {len(row_values)} questions; "
                f"expected {expected_questions}"
            )
        matrix.append(row_values)
    if min_rows is not None and len(matrix) < min_rows:
        raise ValueError(f"{label} has {len(matrix)} rows; expected at least {min_rows}")
    return matrix

def _score_matrix_num_questions(matrix):
    return len(matrix[0]) if matrix else 0

def _mean_prefix_oracle_curve(scores, max_k, included_rows=None):
    scores = _as_score_matrix(scores, "prefix score matrix", min_rows=max_k)
    num_questions = _score_matrix_num_questions(scores)
    if num_questions == 0:
        return [0.0 for _ in range(max_k)]
    if included_rows is None:
        included_rows = set(range(len(scores)))
    else:
        included_rows = {int(row_idx) for row_idx in included_rows}
    curve = []
    for k in range(1, max_k + 1):
        prefix_rows = [
            row_idx
            for row_idx in range(min(k, len(scores)))
            if row_idx in included_rows
        ]
        if not prefix_rows:
            curve.append(0.0)
            continue
        total = 0.0
        for q_idx in range(num_questions):
            total += max(scores[row_idx][q_idx] for row_idx in prefix_rows)
        curve.append(total / num_questions)
    return curve

def _mean_rows_curve(scores, max_k):
    scores = _as_score_matrix(scores, "budget score matrix", min_rows=max_k)
    curve = []
    for row_idx in range(max_k):
        row = scores[row_idx]
        curve.append(sum(row) / len(row) if row else 0.0)
    return curve

def _load_or_compute_retriever_scores_test(dataset_name, retriever, num_docs_to_fetch, embedder=None):
    scores_path = C.get_retriever_scores_test(
        dataset_name,
        retriever,
        num_docs_to_fetch,
        embedder=embedder,
    )
    scores_read_path = _artifact_read_path(scores_path)
    if scores_read_path.exists():
        with scores_read_path.open("rb") as f:
            return pickle.load(f)

    retrievals_path = C.get_retrievals_test(
        dataset_name,
        retriever,
        num_docs_to_fetch,
        embedder=embedder,
    )
    retrievals_read_path = _artifact_read_path(retrievals_path)
    if not retrievals_read_path.exists():
        raise FileNotFoundError(
            f"Retrievals test file not found: expected_path={retrievals_path}, "
            f"checked_path={retrievals_read_path}"
        )
    with retrievals_read_path.open("rb") as f:
        payload = pickle.load(f)
    results = payload.get("results", [])

    questions = _load_test_questions_for_support(dataset_name)
    num_questions = len(results)
    if num_questions == 0:
        return []
    num_retrievers = len(results[0])
    recall_matrix = [[0.0 for _ in range(num_questions)] for _ in range(num_retrievers)]

    for qid in range(num_questions):
        gold_docs = questions[qid].get("target") or questions[qid].get("support_docs") or []
        if not gold_docs:
            continue
        for rid in range(num_retrievers):
            retrieved_units = results[qid][rid]
            retrieved_doc_ids = [tu.doc_id for tu in retrieved_units]
            _, _, recall = f1_support(retrieved_doc_ids, gold_docs)
            recall_matrix[rid][qid] = recall

    scores_write_path = _artifact_write_path(scores_path)
    scores_write_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_write_path.open("wb") as f:
        pickle.dump(recall_matrix, f)
    return recall_matrix

def _load_or_compute_retriever_scores_test_f1(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    embedder=None,
    compute_missing=True,
):
    scores_path = C.get_retriever_scores_test_f1(
        dataset_name,
        retriever,
        num_docs_to_fetch,
        embedder=embedder,
    )
    scores_read_path = _artifact_read_path(scores_path)
    if scores_read_path.exists():
        with scores_read_path.open("rb") as f:
            return pickle.load(f)
    if not compute_missing:
        raise FileNotFoundError(
            f"F1 score file not found: expected_path={scores_path}, "
            f"checked_path={scores_read_path}"
        )

    retrievals_path = C.get_retrievals_test(
        dataset_name,
        retriever,
        num_docs_to_fetch,
        embedder=embedder,
    )
    retrievals_read_path = _artifact_read_path(retrievals_path)
    if not retrievals_read_path.exists():
        raise FileNotFoundError(
            f"Retrievals test file not found: expected_path={retrievals_path}, "
            f"checked_path={retrievals_read_path}"
        )
    print(
        f"[plot-all-pool-support] computing full-pool F1 scores: "
        f"dataset={dataset_name} retriever={retriever} embedder={embedder} "
        f"retrievals={retrievals_path}",
        flush=True,
    )
    with retrievals_read_path.open("rb") as f:
        payload = pickle.load(f)
    results = payload.get("results", [])

    questions = _load_test_questions_for_support(dataset_name)
    num_questions = len(results)
    if num_questions == 0:
        return []
    num_retrievers = len(results[0])
    f1_matrix = [[0.0 for _ in range(num_questions)] for _ in range(num_retrievers)]

    for qid in range(num_questions):
        gold_docs = questions[qid].get("target") or questions[qid].get("support_docs") or []
        if not gold_docs:
            continue
        for rid in range(num_retrievers):
            retrieved_units = results[qid][rid]
            retrieved_doc_ids = [tu.doc_id for tu in retrieved_units]
            f1, _prec, _rec = f1_support(retrieved_doc_ids, gold_docs)
            f1_matrix[rid][qid] = f1

    scores_write_path = _artifact_write_path(scores_path)
    scores_write_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_write_path.open("wb") as f:
        pickle.dump(f1_matrix, f)
    print(
        f"[plot-all-pool-support] wrote full-pool F1 scores: "
        f"expected_path={scores_path}, written_path={scores_write_path}",
        flush=True,
    )
    return f1_matrix

def _compute_portfolio_union_scores_test_f1(portfolio_id, dataset_name, num_docs_to_fetch):
    retrievals_path = C.get_portfolio_union_retrievals_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    payload, retrievals_read_path = _load_pickle_artifact(
        retrievals_path,
        f"portfolio union retrievals for dataset={dataset_name}",
    )
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError(
            f"Portfolio union retrieval payload must contain list results: "
            f"expected_path={retrievals_path}, loaded_path={retrievals_read_path}"
        )

    questions = _load_test_questions_for_support(dataset_name)
    if len(results) != len(questions):
        raise ValueError(
            f"Portfolio union retrieval question count mismatch for dataset={dataset_name}: "
            f"results={len(results)}, questions={len(questions)}, path={retrievals_path}"
        )
    portfolio_size = len(results[0]) if results else 0
    f1_matrix = [[0.0 for _ in questions] for _ in range(portfolio_size)]
    for q_idx, per_question_results in enumerate(results):
        if len(per_question_results) != portfolio_size:
            raise ValueError(
                f"Portfolio union retrieval width mismatch for dataset={dataset_name} "
                f"question_idx={q_idx}: expected={portfolio_size}, got={len(per_question_results)}, "
                f"path={retrievals_path}"
            )
        gold_docs = _support_gold_docs(questions[q_idx])
        for rank, retrieved_units in enumerate(per_question_results):
            f1, _prec, _recall = f1_support(_retrieved_doc_ids(retrieved_units), gold_docs)
            f1_matrix[rank][q_idx] = f1

    scores_path = C.get_portfolio_union_scores_test_f1(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    write_path = _save_pickle_artifact(scores_path, f1_matrix)
    print(
        f"[plot-all-pool-support] computed portfolio F1 scores: "
        f"expected_path={scores_path}, written_path={write_path}",
        flush=True,
    )
    return f1_matrix

def _compute_family_best_scores_test_f1(
    portfolio_id,
    dataset_name,
    family,
    num_docs_to_fetch,
    max_k,
):
    retrievals_path = C.get_family_best_retrievals_test(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    payload, retrievals_read_path = _load_pickle_artifact(
        retrievals_path,
        f"family-best retrievals for dataset={dataset_name} family={family}",
    )
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError(
            f"Family-best retrieval payload must contain list results: "
            f"expected_path={retrievals_path}, loaded_path={retrievals_read_path}"
        )

    questions = _load_test_questions_for_support(dataset_name)
    if len(results) != len(questions):
        raise ValueError(
            f"Family-best retrieval question count mismatch for dataset={dataset_name} "
            f"family={family}: results={len(results)}, questions={len(questions)}, "
            f"path={retrievals_path}"
        )
    f1_matrix = [[0.0 for _ in questions] for _ in range(max_k)]
    for q_idx, retrieved_units in enumerate(results):
        gold_docs = _support_gold_docs(questions[q_idx])
        for k_idx in range(max_k):
            doc_budget = (k_idx + 1) * int(num_docs_to_fetch)
            f1, _prec, _recall = f1_support(
                _retrieved_doc_ids(retrieved_units[:doc_budget]),
                gold_docs,
            )
            f1_matrix[k_idx][q_idx] = f1

    scores_path = C.get_family_best_scores_test_f1(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    write_path = _save_pickle_artifact(scores_path, f1_matrix)
    print(
        f"[plot-all-pool-support] computed family-best F1 scores: "
        f"dataset={dataset_name} family={family} expected_path={scores_path}, "
        f"written_path={write_path}",
        flush=True,
    )
    return f1_matrix

def _load_or_compute_portfolio_union_f1(
    portfolio_id,
    dataset_name,
    num_docs_to_fetch,
    compute_missing_f1,
):
    scores_path = C.get_portfolio_union_scores_test_f1(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    read_path = _artifact_read_path(scores_path)
    if read_path.exists():
        matrix, _ = _load_pickle_artifact(
            scores_path,
            f"portfolio union F1 scores for dataset={dataset_name}",
        )
        return matrix
    if not compute_missing_f1:
        raise FileNotFoundError(
            f"Missing portfolio union F1 scores for dataset={dataset_name}: "
            f"expected_path={scores_path}, checked_path={read_path}. "
            "Pass --compute-missing-f1 to compute them from retrievals."
        )
    return _compute_portfolio_union_scores_test_f1(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )

def _load_or_compute_family_best_f1(
    portfolio_id,
    dataset_name,
    family,
    num_docs_to_fetch,
    max_k,
    compute_missing_f1,
):
    scores_path = C.get_family_best_scores_test_f1(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    read_path = _artifact_read_path(scores_path)
    if read_path.exists():
        matrix, _ = _load_pickle_artifact(
            scores_path,
            f"family-best F1 scores for dataset={dataset_name} family={family}",
        )
        return matrix
    if compute_missing_f1:
        return _compute_family_best_scores_test_f1(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
        )
    raise FileNotFoundError(
        f"Missing family-best F1 scores for dataset={dataset_name} family={family}: "
        f"expected_path={scores_path}, checked_path={read_path}. "
        "Pass --compute-missing-f1 to compute them from retrievals."
    )

def _load_portfolio_union_manifest_for_support(portfolio_id, num_docs_to_fetch):
    manifest_path = C.get_universal_portfolio_union_manifest(
        portfolio_id,
        num_docs_to_fetch,
    )
    manifest, manifest_read_path = _load_pickle_artifact(
        manifest_path,
        f"portfolio union manifest for portfolio_id={portfolio_id}",
    )
    if not isinstance(manifest, dict):
        raise ValueError(
            f"Portfolio union manifest must be a dict: expected_path={manifest_path}, "
            f"loaded_path={manifest_read_path}"
        )
    return manifest, manifest_path, manifest_read_path

def _pool_specs_from_manifest(manifest):
    pool_specs = manifest.get("pool_specs") or C.get_pool_specs_for_set(C.POOL_SET_ALL_IMPLEMENTED)
    return [C.normalize_pool_spec(spec) for spec in pool_specs]

def _normalized_excluded_families(excluded_families):
    return {str(family) for family in (excluded_families or [])}

def _member_family(member):
    return member.get("family") or member.get("retriever")

def _filter_pool_specs_for_support(pool_specs, excluded_families):
    excluded = _normalized_excluded_families(excluded_families)
    return [
        spec
        for spec in pool_specs
        if spec.get("family", spec.get("retriever")) not in excluded
        and spec.get("retriever") not in excluded
    ]

def _selected_portfolio_members_from_manifest(manifest):
    selected = manifest.get("selected_retrievers")
    if selected is not None:
        return list(selected)

    retriever_map = manifest.get("retriever_map") or []
    selected = []
    for rank, global_idx in enumerate(manifest.get("portfolio", []), start=1):
        if 0 <= int(global_idx) < len(retriever_map):
            selected.append(
                {
                    **retriever_map[int(global_idx)],
                    "rank": rank,
                    "global_idx": int(global_idx),
                }
            )
    return selected

def _included_portfolio_rows(manifest, max_k, excluded_families):
    excluded = _normalized_excluded_families(excluded_families)
    selected = _selected_portfolio_members_from_manifest(manifest)
    included = []
    excluded_members = []
    for row_idx, member in enumerate(selected[:max_k]):
        family = _member_family(member)
        retriever = member.get("retriever")
        if family in excluded or retriever in excluded:
            excluded_members.append(
                {
                    "row_idx": row_idx,
                    "rank": member.get("rank", row_idx + 1),
                    "pool_id": member.get("pool_id", member.get("pool_label")),
                    "retriever": retriever,
                    "family": family,
                    "local_idx": member.get("local_idx"),
                    "global_idx": member.get("global_idx"),
                }
            )
            continue
        included.append(row_idx)
    return included, excluded_members

def _member_metric_context(member):
    retriever = member.get("retriever") or member.get("family")
    if retriever not in {C.DS, C.VENDI, C.GRAPH_DENSE}:
        raise ValueError(f"Unsupported member retriever in manifest: {retriever!r}")
    if "local_idx" not in member:
        raise ValueError(f"Manifest member is missing local_idx: {member}")
    local_idx = int(member["local_idx"])
    artifact_embedder_key = member.get("artifact_embedder_key")
    if retriever == C.GRAPH_DENSE:
        artifact_embedder_key = C.GRAPH_DENSE_MIXED_EMBEDDER_KEY
    elif artifact_embedder_key is None:
        raise ValueError(f"Dense manifest member is missing artifact_embedder_key: {member}")
    artifact_embedder_key = C.normalize_embedder_key(artifact_embedder_key)
    return retriever, artifact_embedder_key, local_idx

def _load_full_pool_metric_matrix(
    dataset_name,
    retriever,
    artifact_embedder_key,
    num_docs_to_fetch,
    metric,
    expected_questions,
    compute_missing_f1,
    matrix_cache,
):
    cache_key = (dataset_name, retriever, artifact_embedder_key, metric)
    if cache_key in matrix_cache:
        return matrix_cache[cache_key]

    if metric == "recall":
        raw_matrix = _load_or_compute_retriever_scores_test(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=artifact_embedder_key,
        )
        score_path = C.get_retriever_scores_test(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=artifact_embedder_key,
        )
    elif metric == "f1":
        raw_matrix = _load_or_compute_retriever_scores_test_f1(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=artifact_embedder_key,
            compute_missing=compute_missing_f1,
        )
        score_path = C.get_retriever_scores_test_f1(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=artifact_embedder_key,
        )
    else:
        raise ValueError(f"Unsupported support metric: {metric}")

    matrix = _as_score_matrix(
        raw_matrix,
        (
            f"full-pool {metric} scores for dataset={dataset_name} "
            f"retriever={retriever} embedder={artifact_embedder_key}"
        ),
        expected_questions=expected_questions,
    )
    artifact = {
        "scores_path": score_path,
        "scores_loaded_path": str(_artifact_read_path(score_path)),
        "retriever": retriever,
        "artifact_embedder_key": artifact_embedder_key,
        "metric": metric,
    }
    matrix_cache[cache_key] = (matrix, artifact)
    return matrix, artifact

def _pool_optimal_score(
    dataset_name,
    pool_specs,
    num_docs_to_fetch,
    metric,
    expected_questions,
    compute_missing_f1,
    matrix_cache,
):
    best_by_question = [0.0 for _ in range(expected_questions)]
    artifacts = []
    total_rows = 0
    for spec in pool_specs:
        matrix, artifact = _load_full_pool_metric_matrix(
            dataset_name=dataset_name,
            retriever=spec["retriever"],
            artifact_embedder_key=spec["artifact_embedder_key"],
            num_docs_to_fetch=num_docs_to_fetch,
            metric=metric,
            expected_questions=expected_questions,
            compute_missing_f1=compute_missing_f1,
            matrix_cache=matrix_cache,
        )
        if len(matrix) != int(spec["pool_size"]):
            raise ValueError(
                f"Full-pool score row mismatch for dataset={dataset_name} "
                f"pool={spec['pool_id']} metric={metric}: "
                f"scores={len(matrix)}, expected={spec['pool_size']}"
            )
        total_rows += len(matrix)
        artifacts.append(artifact)
        for row in matrix:
            for q_idx, score in enumerate(row):
                if score > best_by_question[q_idx]:
                    best_by_question[q_idx] = score
    return (
        sum(best_by_question) / len(best_by_question) if best_by_question else 0.0,
        {"artifacts": artifacts, "total_retrievers": total_rows},
    )

def _topk_by_average_curve(
    dataset_name,
    manifest,
    num_docs_to_fetch,
    max_k,
    metric,
    expected_questions,
    compute_missing_f1,
    matrix_cache,
    excluded_families=None,
):
    excluded = _normalized_excluded_families(excluded_families)
    topk_members_all = manifest.get("topk_by_average_retriever_baseline") or []
    topk_members = [
        member
        for member in topk_members_all
        if _member_family(member) not in excluded
        and member.get("retriever") not in excluded
    ]
    if len(topk_members) < max_k:
        raise ValueError(
            f"Portfolio union manifest has only {len(topk_members)} top-k-by-average "
            f"members after excluding families={sorted(excluded)}; need max_k={max_k}. "
            "Recompute the manifest with a larger --portfolio-size or include fewer exclusions."
        )

    rows = []
    artifacts = []
    for rank, member in enumerate(topk_members[:max_k], start=1):
        retriever, artifact_embedder_key, local_idx = _member_metric_context(member)
        matrix, artifact = _load_full_pool_metric_matrix(
            dataset_name=dataset_name,
            retriever=retriever,
            artifact_embedder_key=artifact_embedder_key,
            num_docs_to_fetch=num_docs_to_fetch,
            metric=metric,
            expected_questions=expected_questions,
            compute_missing_f1=compute_missing_f1,
            matrix_cache=matrix_cache,
        )
        if local_idx >= len(matrix):
            raise IndexError(
                f"Top-k-by-average member local_idx out of range for dataset={dataset_name} "
                f"metric={metric}: rank={rank}, retriever={retriever}, "
                f"embedder={artifact_embedder_key}, local_idx={local_idx}, rows={len(matrix)}"
            )
        rows.append(matrix[local_idx])
        artifacts.append(
            {
                **artifact,
                "rank": rank,
                "pool_id": member.get("pool_id", member.get("pool_label")),
                "local_idx": local_idx,
                "global_idx": member.get("global_idx"),
                "avg_train_score": member.get("avg_train_score"),
            }
        )

    curve = []
    for k in range(1, max_k + 1):
        total = 0.0
        for q_idx in range(expected_questions):
            total += max(rows[row_idx][q_idx] for row_idx in range(k))
        curve.append(total / expected_questions if expected_questions else 0.0)
    return curve, {"artifacts": artifacts}

def _load_all_pool_support_dataset_curves(
    portfolio_id,
    dataset_name,
    num_docs_to_fetch,
    max_k,
    family_best_families,
    manifest,
    pool_specs,
    excluded_families,
    compute_missing_f1,
):
    questions = _load_test_questions_for_support(dataset_name)
    num_questions = len(questions)
    matrix_cache = {}
    portfolio_included_rows, portfolio_excluded_members = _included_portfolio_rows(
        manifest,
        max_k,
        excluded_families,
    )

    portfolio_recall_path = C.get_portfolio_union_scores_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    portfolio_recall, portfolio_recall_read_path = _load_pickle_artifact(
        portfolio_recall_path,
        f"portfolio union recall scores for dataset={dataset_name}",
    )
    portfolio_recall = _as_score_matrix(
        portfolio_recall,
        f"portfolio union recall scores for dataset={dataset_name}",
        expected_questions=num_questions,
        min_rows=max_k,
    )

    portfolio_f1 = _load_or_compute_portfolio_union_f1(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
        compute_missing_f1,
    )
    portfolio_f1 = _as_score_matrix(
        portfolio_f1,
        f"portfolio union F1 scores for dataset={dataset_name}",
        expected_questions=num_questions,
        min_rows=max_k,
    )

    pool_recall_oracle, pool_recall_paths = _pool_optimal_score(
        dataset_name=dataset_name,
        pool_specs=pool_specs,
        num_docs_to_fetch=num_docs_to_fetch,
        metric="recall",
        expected_questions=num_questions,
        compute_missing_f1=compute_missing_f1,
        matrix_cache=matrix_cache,
    )
    pool_f1_oracle, pool_f1_paths = _pool_optimal_score(
        dataset_name=dataset_name,
        pool_specs=pool_specs,
        num_docs_to_fetch=num_docs_to_fetch,
        metric="f1",
        expected_questions=num_questions,
        compute_missing_f1=compute_missing_f1,
        matrix_cache=matrix_cache,
    )
    topk_recall, topk_recall_paths = _topk_by_average_curve(
        dataset_name=dataset_name,
        manifest=manifest,
        num_docs_to_fetch=num_docs_to_fetch,
        max_k=max_k,
        metric="recall",
        expected_questions=num_questions,
        compute_missing_f1=compute_missing_f1,
        matrix_cache=matrix_cache,
        excluded_families=excluded_families,
    )
    topk_f1, topk_f1_paths = _topk_by_average_curve(
        dataset_name=dataset_name,
        manifest=manifest,
        num_docs_to_fetch=num_docs_to_fetch,
        max_k=max_k,
        metric="f1",
        expected_questions=num_questions,
        compute_missing_f1=compute_missing_f1,
        matrix_cache=matrix_cache,
        excluded_families=excluded_families,
    )

    family_recall = {}
    family_f1 = {}
    family_paths = {}
    for family in family_best_families:
        recall_path = C.get_family_best_scores_test(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
        )
        recall_scores, recall_read_path = _load_pickle_artifact(
            recall_path,
            f"family-best recall scores for dataset={dataset_name} family={family}",
        )
        recall_scores = _as_score_matrix(
            recall_scores,
            f"family-best recall scores for dataset={dataset_name} family={family}",
            expected_questions=num_questions,
            min_rows=max_k,
        )
        f1_scores = _load_or_compute_family_best_f1(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
            compute_missing_f1,
        )
        f1_scores = _as_score_matrix(
            f1_scores,
            f"family-best F1 scores for dataset={dataset_name} family={family}",
            expected_questions=num_questions,
            min_rows=max_k,
        )
        family_recall[family] = _mean_rows_curve(recall_scores, max_k)
        family_f1[family] = _mean_rows_curve(f1_scores, max_k)
        family_paths[family] = {
            "recall_scores_path": recall_path,
            "recall_scores_loaded_path": str(recall_read_path),
            "f1_scores_path": C.get_family_best_scores_test_f1(
                portfolio_id,
                dataset_name,
                family,
                num_docs_to_fetch,
                max_k,
            ),
            "f1_scores_loaded_path": str(_artifact_read_path(C.get_family_best_scores_test_f1(
                portfolio_id,
                dataset_name,
                family,
                num_docs_to_fetch,
                max_k,
            ))),
        }

    return {
        "dataset": dataset_name,
        "num_questions": num_questions,
        "ks": list(range(1, max_k + 1)),
        "recall": {
            "portfolio_prefix": _mean_prefix_oracle_curve(
                portfolio_recall,
                max_k,
                included_rows=portfolio_included_rows,
            ),
            "pool_optimal": pool_recall_oracle,
            "topk_by_average": topk_recall,
            "family_best": family_recall,
        },
        "f1": {
            "portfolio_prefix": _mean_prefix_oracle_curve(
                portfolio_f1,
                max_k,
                included_rows=portfolio_included_rows,
            ),
            "pool_optimal": pool_f1_oracle,
            "topk_by_average": topk_f1,
            "family_best": family_f1,
        },
        "paths": {
            "portfolio_recall_scores_path": portfolio_recall_path,
            "portfolio_recall_scores_loaded_path": str(portfolio_recall_read_path),
            "portfolio_f1_scores_path": C.get_portfolio_union_scores_test_f1(
                portfolio_id,
                dataset_name,
                num_docs_to_fetch,
            ),
            "portfolio_f1_scores_loaded_path": str(_artifact_read_path(C.get_portfolio_union_scores_test_f1(
                portfolio_id,
                dataset_name,
                num_docs_to_fetch,
            ))),
            "pool_optimal": {
                "recall": pool_recall_paths,
                "f1": pool_f1_paths,
            },
            "topk_by_average": {
                "recall": topk_recall_paths,
                "f1": topk_f1_paths,
            },
            "family_best": family_paths,
        },
        "excluded_families": sorted(_normalized_excluded_families(excluded_families)),
        "portfolio_included_rows": portfolio_included_rows,
        "portfolio_excluded_members": portfolio_excluded_members,
    }

def _average_curves(curves):
    if not curves:
        return []
    width = len(curves[0])
    return [
        sum(curve[idx] for curve in curves) / len(curves)
        for idx in range(width)
    ]

def _average_all_pool_support_payload(per_dataset, family_best_families):
    average = {}
    for metric in ["recall", "f1"]:
        average[metric] = {
            "portfolio_prefix": _average_curves([
                payload[metric]["portfolio_prefix"]
                for payload in per_dataset.values()
            ]),
            "pool_optimal": (
                sum(payload[metric]["pool_optimal"] for payload in per_dataset.values())
                / len(per_dataset)
            ),
            "topk_by_average": _average_curves([
                payload[metric]["topk_by_average"]
                for payload in per_dataset.values()
            ]),
            "family_best": {},
        }
        for family in family_best_families:
            average[metric]["family_best"][family] = _average_curves([
                payload[metric]["family_best"][family]
                for payload in per_dataset.values()
            ])
    return average

def _all_pool_support_line_specs(family_best_families):
    specs = [
        (
            "portfolio_prefix",
            "All-pool portfolio",
            {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "topk_by_average",
            "Top-k retrievers by avg score",
            {"color": "#8c564b", "marker": "D", "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "pool_optimal",
            "Pool-optimal (per-query)",
            {"color": "#d62728", "marker": None, "linestyle": "-", "linewidth": 1.8},
        ),
    ]
    family_labels = {
        C.GRAPH_DENSE: "Best GraphDense, k x docs",
        C.DS: "Best DS, k x docs",
        C.VENDI: "Best Vendi, k x docs",
    }
    family_styles = {
        C.DS: {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "linewidth": 1.8},
        C.VENDI: {"color": "#2ca02c", "marker": "x", "linestyle": "-", "linewidth": 1.8},
        C.GRAPH_DENSE: {"color": "#9467bd", "marker": "^", "linestyle": "-", "linewidth": 1.8},
    }
    for family in family_best_families:
        specs.append((f"family:{family}", family_labels[family], family_styles[family]))
    return specs

def _metric_values_for_line(metric_payload, key, ks):
    if key == "portfolio_prefix":
        return metric_payload["portfolio_prefix"]
    if key == "pool_optimal":
        return [metric_payload["pool_optimal"]] * len(ks)
    if key == "topk_by_average":
        return metric_payload["topk_by_average"]
    if key.startswith("family:"):
        family = key.split(":", 1)[1]
        return metric_payload["family_best"][family]
    raise ValueError(f"Unsupported all-pool support line key: {key}")

def _plot_all_pool_support_legend(
    output_dir,
    portfolio_id,
    num_docs_to_fetch,
    max_k,
    family_best_families,
):
    safe_portfolio_id = _safe_plot_name(portfolio_id)
    stem = f"all_pool_support_legend_{safe_portfolio_id}_{num_docs_to_fetch}_k{max_k}"
    png_path = Path(output_dir) / f"{stem}.png"
    handles = []
    labels = []
    for _key, label, style in _all_pool_support_line_specs(family_best_families):
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color"),
                marker=style.get("marker"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.8),
                markersize=7,
            )
        )
        labels.append(label)
    fig_width = max(14.0, 2.7 * len(labels))
    fig = plt.figure(figsize=(fig_width, 1.0))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        columnspacing=1.5,
        handlelength=2.2,
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return {"png": str(png_path)}

def _plot_all_pool_support_metric(
    ks,
    metric_payload,
    metric,
    output_dir,
    portfolio_id,
    num_docs_to_fetch,
    max_k,
    family_best_families,
):
    y_label = "Support F1" if metric == "f1" else "Support Recall"
    safe_portfolio_id = _safe_plot_name(portfolio_id)
    stem = f"all_pool_support_{metric}_{safe_portfolio_id}_{num_docs_to_fetch}_k{max_k}"
    png_path = Path(output_dir) / f"{stem}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.4, 4.0))
    for key, label, style in _all_pool_support_line_specs(family_best_families):
        plt.plot(
            ks,
            _metric_values_for_line(metric_payload, key, ks),
            label=label,
            color=style.get("color"),
            marker=style.get("marker"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.8),
            markersize=7,
        )

    plt.xlabel("Portfolio prefix size k")
    plt.ylabel(y_label)
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    return {"png": str(png_path)}

def plot_all_pool_support_paper(
    portfolio_id=C.POOL_SET_ALL_IMPLEMENTED,
    datasets=C.DATASETS,
    num_docs_to_fetch=4,
    max_k=5,
    compute_missing_f1=True,
    include_vendi_family_best=False,
    strict=False,
    output_dir=None,
):
    """
    Plot paper support recall/F1 curves for the all-pool selected portfolio.

    Curves are computed per dataset first and then macro-averaged across the
    datasets that have a complete set of required artifacts.
    """
    if isinstance(datasets, str):
        datasets = [value.strip() for value in datasets.split(",") if value.strip()]
    else:
        datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets provided for all-pool support plotting.")
    if max_k < 1:
        raise ValueError(f"max_k must be >= 1, got {max_k}")
    if num_docs_to_fetch < 1:
        raise ValueError(f"num_docs_to_fetch must be >= 1, got {num_docs_to_fetch}")
    manifest, manifest_path, manifest_read_path = _load_portfolio_union_manifest_for_support(
        portfolio_id,
        num_docs_to_fetch,
    )
    excluded_families = []
    pool_specs = _filter_pool_specs_for_support(
        _pool_specs_from_manifest(manifest),
        excluded_families,
    )
    family_best_families = [
        C.DS,
        C.GRAPH_DENSE,
    ]
    if include_vendi_family_best:
        family_best_families.insert(1, C.VENDI)
    output_dir = Path(output_dir) if output_dir is not None else Path(C.PLOTS_DIR) / "average"

    per_dataset = {}
    failures = []
    for idx, dataset_name in enumerate(datasets, start=1):
        _progress_bar(idx - 1, len(datasets), "all-pool support: datasets")
        try:
            per_dataset[dataset_name] = _load_all_pool_support_dataset_curves(
                portfolio_id=portfolio_id,
                dataset_name=dataset_name,
                num_docs_to_fetch=num_docs_to_fetch,
                max_k=max_k,
                family_best_families=family_best_families,
                manifest=manifest,
                pool_specs=pool_specs,
                excluded_families=excluded_families,
                compute_missing_f1=compute_missing_f1,
            )
        except (FileNotFoundError, ValueError) as exc:
            message = (
                f"[plot-all-pool-support] skipping dataset={dataset_name}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(message, flush=True)
            failures.append(
                {
                    "dataset": dataset_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if strict:
                raise
        _progress_bar(idx, len(datasets), "all-pool support: datasets")

    if not per_dataset:
        details = " | ".join(
            f"{failure['dataset']}: {failure['error']}"
            for failure in failures
        )
        raise FileNotFoundError(
            "No complete datasets were available for all-pool support plotting. "
            + details
        )

    average = _average_all_pool_support_payload(per_dataset, family_best_families)
    ks = list(range(1, max_k + 1))
    plot_paths = {
        "recall": _plot_all_pool_support_metric(
            ks,
            average["recall"],
            "recall",
            output_dir,
            portfolio_id,
            num_docs_to_fetch,
            max_k,
            family_best_families,
        ),
        "f1": _plot_all_pool_support_metric(
            ks,
            average["f1"],
            "f1",
            output_dir,
            portfolio_id,
            num_docs_to_fetch,
            max_k,
            family_best_families,
        ),
        "legend": _plot_all_pool_support_legend(
            output_dir,
            portfolio_id,
            num_docs_to_fetch,
            max_k,
            family_best_families,
        ),
    }

    safe_portfolio_id = _safe_plot_name(portfolio_id)
    payload_path = (
        Path(output_dir)
        / f"all_pool_support_metrics_{safe_portfolio_id}_{num_docs_to_fetch}_k{max_k}.pickle"
    )
    payload = {
        "schema": "all_pool_support_paper_plots",
        "schema_version": 4,
        "portfolio_id": portfolio_id,
        "datasets_requested": datasets,
        "datasets_plotted": list(per_dataset.keys()),
        "num_docs": int(num_docs_to_fetch),
        "max_k": int(max_k),
        "ks": ks,
        "excluded_families": sorted(_normalized_excluded_families(excluded_families)),
        "pool_specs_plotted": pool_specs,
        "family_best_families": family_best_families,
        "include_vendi_family_best": bool(include_vendi_family_best),
        "compute_missing_f1": bool(compute_missing_f1),
        "plotted_lines": [
            {"key": key, "label": label}
            for key, label, _style in _all_pool_support_line_specs(family_best_families)
        ],
        "manifest_path": manifest_path,
        "manifest_loaded_path": str(manifest_read_path),
        "per_dataset": per_dataset,
        "average": average,
        "failures": failures,
        "plot_paths": plot_paths,
        "payload_path": str(payload_path),
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[plot-all-pool-support] saved plots: recall={plot_paths['recall']['png']} "
        f"f1={plot_paths['f1']['png']} legend={plot_paths['legend']['png']} "
        f"payload={payload_path}",
        flush=True,
    )
    return payload

# Portfolio-Router Ablation Plot

def _parse_router_run_ids(raw):
    if raw is None or str(raw).strip() == "":
        return {}
    mapping = {}
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
        elif ":" in item:
            key, value = item.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --run-ids item {item!r}; expected k2:run_id or 2=run_id."
            )
        key = key.strip().lower()
        if key.startswith("k"):
            key = key[1:]
        k = int(key)
        if k <= 1:
            raise ValueError("--run-ids should only specify router runs for k >= 2.")
        value = value.strip()
        if not value:
            raise ValueError(f"Missing run id for k={k}.")
        mapping[k] = value
    return mapping

def _router_prediction_split_dir(portfolio_id, dataset, num_docs, k, split):
    return (
        C.get_portfolio_union_dir(dataset, portfolio_id, num_docs)
        + f"portfolio_router_predictions/k{int(k)}/{_safe_plot_name(split)}/"
    )

def _available_router_run_ids(portfolio_id, dataset, num_docs, k, split):
    split_dir = _router_prediction_split_dir(portfolio_id, dataset, num_docs, k, split)
    read_dir = _artifact_read_path(split_dir)
    path = Path(read_dir)
    if not path.exists():
        return [], split_dir, read_dir
    run_ids = []
    for child in path.iterdir():
        if child.is_dir() and (child / "predictions.pickle").exists():
            run_ids.append(child.name)
    return sorted(run_ids), split_dir, read_dir

def _discover_router_run_id(portfolio_id, datasets, num_docs, k, split):
    per_dataset = {}
    missing = []
    for dataset in datasets:
        run_ids, expected_dir, checked_dir = _available_router_run_ids(
            portfolio_id,
            dataset,
            num_docs,
            k,
            split,
        )
        if not run_ids:
            missing.append(
                f"{dataset}: expected_dir={expected_dir}, checked_dir={checked_dir}"
            )
            continue
        non_smoke = [run_id for run_id in run_ids if not run_id.startswith("smoke")]
        per_dataset[dataset] = set(non_smoke or run_ids)

    if missing:
        raise FileNotFoundError(
            f"Missing portfolio-router predictions for k={k}: " + " | ".join(missing)
        )
    common = set.intersection(*per_dataset.values()) if per_dataset else set()
    if not common:
        details = ", ".join(
            f"{dataset}={sorted(values)}" for dataset, values in per_dataset.items()
        )
        raise FileNotFoundError(
            f"No common portfolio-router run id for k={k} across datasets. "
            f"Use --run-ids to choose explicitly. Available: {details}"
        )
    best = sorted(run_id for run_id in common if "best" in run_id)
    candidates = best or sorted(common)
    if len(candidates) != 1:
        raise ValueError(
            f"Ambiguous portfolio-router run ids for k={k}: {candidates}. "
            "Use --run-ids k2:...,k3:... to select one."
        )
    return candidates[0]

def _resolve_router_run_ids(portfolio_id, datasets, num_docs, max_k, split, run_ids, run_id_template):
    parsed = _parse_router_run_ids(run_ids)
    resolved = {}
    for k in range(2, int(max_k) + 1):
        if k in parsed:
            resolved[k] = parsed[k]
        elif run_id_template:
            resolved[k] = str(run_id_template).format(k=k, portfolio_size=k)
        else:
            resolved[k] = _discover_router_run_id(
                portfolio_id,
                datasets,
                num_docs,
                k,
                split,
            )
    return resolved

def _load_router_prediction_payload(portfolio_id, dataset, num_docs, k, split, run_id):
    expected_path = C.get_portfolio_router_predictions(
        portfolio_id,
        dataset,
        num_docs,
        k,
        split,
        run_id,
    )
    payload, loaded_path = _load_pickle_artifact(
        expected_path,
        (
            f"portfolio-router predictions for dataset={dataset} "
            f"portfolio_id={portfolio_id} k={k} split={split} run_id={run_id}"
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(
            f"Portfolio-router prediction payload must be a dict: path={expected_path}"
        )
    top_indices = np.asarray(payload.get("top_indices"), dtype=np.int64)
    if top_indices.ndim != 2 or top_indices.shape[1] < int(k):
        raise ValueError(
            f"top_indices must have shape [Q, >=k]: path={expected_path}, "
            f"shape={top_indices.shape}, k={k}"
        )
    top_indices = top_indices[:, : int(k)]
    if top_indices.size and (top_indices.min() < 0 or top_indices.max() >= int(k)):
        raise ValueError(
            f"top_indices contains out-of-range member indices: path={expected_path}, "
            f"min={int(top_indices.min())}, max={int(top_indices.max())}, k={k}"
        )
    question_indices = np.asarray(payload.get("question_indices"), dtype=np.int64)
    if question_indices.ndim != 1 or question_indices.shape[0] != top_indices.shape[0]:
        raise ValueError(
            f"question_indices must align with top_indices: path={expected_path}, "
            f"question_indices_shape={question_indices.shape}, top_indices_shape={top_indices.shape}"
        )
    return payload, top_indices, question_indices, expected_path, loaded_path

def _score_matrix_from_payload(payload, label):
    if isinstance(payload, dict):
        for key in ("scores", "recalls", "recall_matrix"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError(f"{label} dict payload must contain scores/recalls/recall_matrix.")
    matrix = np.asarray(payload, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be 2D [K, Q], got shape={matrix.shape}")
    return matrix

def _align_kq_to_question_indices(values_kq, question_indices, label):
    if question_indices.size == 0:
        return values_kq[:, []]
    max_idx = int(question_indices.max())
    if max_idx < values_kq.shape[1]:
        return values_kq[:, question_indices]
    if values_kq.shape[1] == question_indices.shape[0]:
        return values_kq
    raise ValueError(
        f"Cannot align {label} with question_indices: "
        f"values_shape={values_kq.shape}, question_indices_shape={question_indices.shape}"
    )

def _load_router_ablation_recall_matrix(portfolio_id, dataset, num_docs, k, question_indices):
    scores_path = C.get_portfolio_union_scores_test(portfolio_id, dataset, num_docs)
    payload, loaded_path = _load_pickle_artifact(
        scores_path,
        f"portfolio union recall scores for dataset={dataset}",
    )
    scores = _score_matrix_from_payload(payload, f"recall scores for dataset={dataset}")
    if scores.shape[0] < int(k):
        raise ValueError(
            f"Recall score matrix has fewer rows than k={k}: "
            f"path={scores_path}, shape={scores.shape}"
        )
    return (
        _align_kq_to_question_indices(
            scores[: int(k), :],
            question_indices,
            f"recall scores for dataset={dataset}",
        ),
        scores_path,
        loaded_path,
        scores.shape[1],
    )

def _router_ablation_values(values_kq, top_indices_qk):
    values = np.asarray(values_kq, dtype=np.float32)
    top_indices = np.asarray(top_indices_qk, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError(f"values_kq must be 2D [K, Q], got shape={values.shape}")
    k, q_count = values.shape
    if top_indices.ndim != 2 or top_indices.shape[0] != q_count:
        raise ValueError(
            f"top_indices must be [Q, K] and align with values: "
            f"values_shape={values.shape}, top_indices_shape={top_indices.shape}"
        )
    values_qk = values.T
    rows = np.arange(q_count)

    def mean(values_1d):
        if values_1d.size == 0 or np.all(np.isnan(values_1d)):
            return 0.0
        return float(np.nanmean(values_1d))

    def topn(n):
        if q_count == 0 or k == 0:
            return 0.0
        n = min(int(n), k)
        selected = values_qk[rows[:, None], top_indices[:, :n]]
        return mean(np.nanmax(selected, axis=1))

    return {
        "portfolio_max": mean(np.nanmax(values_qk, axis=1)) if k > 0 else 0.0,
        "router_top1": topn(1),
        "router_top2": topn(2),
        "router_top3": topn(3),
        "random_member": mean(values.reshape(-1)),
        "first_member": mean(values[0, :]) if k > 0 else 0.0,
    }

def _question_path_for_router_ablation(dataset, split):
    if split == "test":
        return C.get_questions_test(dataset)
    if split == "train":
        return C.get_questions_train(dataset)
    raise ValueError(f"split must be train or test, got {split!r}")

def _load_gold_answers_for_router_ablation(dataset, split, question_indices):
    questions_path = _question_path_for_router_ablation(dataset, split)
    questions_dataset, _loaded_path = _load_pickle_artifact(
        questions_path,
        f"questions for dataset={dataset} split={split}",
    )
    questions = getattr(questions_dataset, "questions", None)
    if not isinstance(questions, list):
        raise ValueError(f"Questions payload has no .questions list: path={questions_path}")
    if question_indices.size and int(question_indices.max()) >= len(questions):
        raise IndexError(
            f"Question index out of range for dataset={dataset}: "
            f"max_idx={int(question_indices.max())}, questions={len(questions)}"
        )
    return [questions[int(qidx)].get("answer") for qidx in question_indices]

def _extract_router_ablation_response(cell):
    if cell is None:
        return None
    if isinstance(cell, dict):
        for key in ("response", "answer", "prediction", "text", "output"):
            value = cell.get(key)
            if value is not None:
                return str(value)
        return None
    return str(cell)

def _router_ablation_rank_from_record(record, k):
    for key in ("retriever_idx", "portfolio_rank", "portfolio_member_idx", "member_idx", "ridx", "rank"):
        if key not in record or record[key] is None:
            continue
        raw = int(record[key])
        if key == "rank":
            raw -= 1
        if 0 <= raw < int(k):
            return raw
    return None

def _router_ablation_question_row_from_record(record, original_to_row, q_count):
    for key in ("question_idx", "question_id", "question_index", "q_idx"):
        if key not in record or record[key] is None:
            continue
        raw = int(record[key])
        if raw in original_to_row:
            return original_to_row[raw]
        if 0 <= raw < q_count:
            return raw
    return None

def _router_ablation_matrix_answers_from_rows(raw_answers, question_indices, k):
    q_count = int(question_indices.shape[0])
    if q_count == 0:
        return []
    if len(raw_answers) > int(question_indices.max()):
        selected_rows = [raw_answers[int(qidx)] for qidx in question_indices]
    elif len(raw_answers) == q_count:
        selected_rows = list(raw_answers)
    else:
        raise ValueError(
            f"Answer matrix row count cannot align with question_indices: "
            f"rows={len(raw_answers)}, Q={q_count}, max_question_idx={int(question_indices.max())}"
        )

    matrix = []
    for row in selected_rows:
        if isinstance(row, dict):
            matrix.append([row.get(rank, row.get(str(rank))) for rank in range(k)])
        elif isinstance(row, (list, tuple)):
            matrix.append([row[rank] if rank < len(row) else None for rank in range(k)])
        else:
            matrix.append([None for _ in range(k)])
    return matrix

def _router_ablation_answers_payload_to_matrix(payload, question_indices, k):
    raw_answers = payload.get("answers", payload) if isinstance(payload, dict) else payload
    q_count = int(question_indices.shape[0])
    matrix = [[None for _ in range(k)] for _ in range(q_count)]
    if raw_answers is None:
        return matrix

    if isinstance(raw_answers, dict):
        original_to_row = {int(qidx): row for row, qidx in enumerate(question_indices)}
        for raw_qidx, per_rank in raw_answers.items():
            qidx = int(raw_qidx)
            row = original_to_row.get(qidx, qidx if 0 <= qidx < q_count else None)
            if row is None:
                continue
            if isinstance(per_rank, dict):
                for rank in range(k):
                    matrix[row][rank] = per_rank.get(rank, per_rank.get(str(rank)))
            elif isinstance(per_rank, (list, tuple)):
                for rank in range(min(k, len(per_rank))):
                    matrix[row][rank] = per_rank[rank]
        return matrix

    if not isinstance(raw_answers, list):
        raise ValueError(f"Unsupported answers payload type: {type(raw_answers).__name__}")
    first = next((item for item in raw_answers if item is not None), None)
    if isinstance(first, (list, tuple)):
        return _router_ablation_matrix_answers_from_rows(raw_answers, question_indices, k)

    original_to_row = {int(qidx): row for row, qidx in enumerate(question_indices)}
    for record in raw_answers:
        if not isinstance(record, dict):
            continue
        row = _router_ablation_question_row_from_record(record, original_to_row, q_count)
        rank = _router_ablation_rank_from_record(record, k)
        if row is None or rank is None:
            continue
        matrix[row][rank] = record
    return matrix

def _compute_router_ablation_em_matrix(answer_payload, dataset, split, question_indices, k):
    gold_answers = _load_gold_answers_for_router_ablation(dataset, split, question_indices)
    answer_matrix = _router_ablation_answers_payload_to_matrix(
        answer_payload,
        question_indices,
        int(k),
    )
    em = np.zeros((int(k), len(question_indices)), dtype=np.float32)
    for qrow, gold in enumerate(gold_answers):
        if gold is None:
            em[:, qrow] = np.nan
            continue
        for rank in range(int(k)):
            response = _extract_router_ablation_response(answer_matrix[qrow][rank])
            if not response:
                continue
            parsed = extract_tagged_answer(response, tag="answer")
            prediction = parsed if parsed is not None else response
            em[rank, qrow] = 1.0 if exact_match_score(prediction, gold) else 0.0
    return em

def _router_ablation_answer_path(portfolio_id, dataset, num_docs, answer_llm, answers_path_template=None):
    if answers_path_template:
        return answers_path_template.format(
            portfolio_id=portfolio_id,
            dataset=dataset,
            num_docs=int(num_docs),
            answer_llm=answer_llm,
        )
    return C.get_portfolio_union_answers_all(portfolio_id, dataset, answer_llm, num_docs)

def _trivial_top_indices(q_count, k):
    return np.tile(np.arange(int(k), dtype=np.int64), (int(q_count), 1))

def _load_router_ablation_dataset_curves(
    portfolio_id,
    dataset,
    num_docs,
    max_k,
    split,
    run_ids_by_k,
    answer_llm,
    answers_path_template,
):
    answer_path = _router_ablation_answer_path(
        portfolio_id,
        dataset,
        num_docs,
        answer_llm,
        answers_path_template=answers_path_template,
    )
    answer_payload, answer_loaded_path = _load_pickle_artifact(
        answer_path,
        f"portfolio union answers for dataset={dataset} answer_llm={answer_llm}",
    )

    curves = {
        "recall": {key: [] for key in _portfolio_router_ablation_line_keys()},
        "em": {key: [] for key in _portfolio_router_ablation_line_keys()},
    }
    per_k = {}
    prediction_paths = {}
    score_paths = {}

    full_recall_payload, _full_scores_loaded_path = _load_pickle_artifact(
        C.get_portfolio_union_scores_test(portfolio_id, dataset, num_docs),
        f"portfolio union recall scores for dataset={dataset}",
    )
    full_recall = _score_matrix_from_payload(
        full_recall_payload,
        f"portfolio union recall scores for dataset={dataset}",
    )
    full_question_count = int(full_recall.shape[1])

    for k in range(1, int(max_k) + 1):
        if k == 1:
            question_indices = np.arange(full_question_count, dtype=np.int64)
            top_indices = _trivial_top_indices(full_question_count, k)
            prediction_path = None
            prediction_loaded_path = None
            run_id = None
        else:
            run_id = run_ids_by_k[int(k)]
            _payload, top_indices, question_indices, prediction_path, prediction_loaded_path = (
                _load_router_prediction_payload(
                    portfolio_id,
                    dataset,
                    num_docs,
                    k,
                    split,
                    run_id,
                )
            )

        recall_kq, recall_path, recall_loaded_path, _ = _load_router_ablation_recall_matrix(
            portfolio_id,
            dataset,
            num_docs,
            k,
            question_indices,
        )
        em_kq = _compute_router_ablation_em_matrix(
            answer_payload,
            dataset,
            split,
            question_indices,
            k,
        )
        recall_metrics = _router_ablation_values(recall_kq, top_indices)
        em_metrics = _router_ablation_values(em_kq, top_indices)

        for key, value in recall_metrics.items():
            curves["recall"][key].append(value)
        for key, value in em_metrics.items():
            curves["em"][key].append(value)

        per_k[int(k)] = {
            "run_id": run_id,
            "num_questions": int(question_indices.shape[0]),
            "recall": recall_metrics,
            "em": em_metrics,
            "prediction_path": prediction_path,
            "prediction_loaded_path": str(prediction_loaded_path) if prediction_loaded_path else None,
            "recall_scores_path": recall_path,
            "recall_scores_loaded_path": str(recall_loaded_path),
        }
        if prediction_path:
            prediction_paths[int(k)] = {
                "path": prediction_path,
                "loaded_path": str(prediction_loaded_path),
                "run_id": run_id,
            }
        score_paths[int(k)] = {
            "path": recall_path,
            "loaded_path": str(recall_loaded_path),
        }

    return {
        "dataset": dataset,
        "ks": list(range(1, int(max_k) + 1)),
        "curves": curves,
        "per_k": per_k,
        "paths": {
            "answer_path": answer_path,
            "answer_loaded_path": str(answer_loaded_path),
            "prediction_paths": prediction_paths,
            "recall_score_paths": score_paths,
        },
    }

def _portfolio_router_ablation_line_keys():
    return [
        "portfolio_max",
        "router_top1",
        "router_top2",
        "router_top3",
        "random_member",
        "first_member",
    ]

def _average_router_ablation_payload(per_dataset):
    average = {
        "recall": {},
        "em": {},
    }
    for metric in ["recall", "em"]:
        for key in _portfolio_router_ablation_line_keys():
            average[metric][key] = _average_curves([
                payload["curves"][metric][key]
                for payload in per_dataset.values()
            ])
    return average

def _portfolio_router_ablation_line_specs():
    return [
        (
            "portfolio_max",
            "Max over portfolio members",
            {"color": "#d62728", "marker": None, "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "router_top1",
            "Router top-1",
            {"color": "#1f77b4", "marker": "o", "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "router_top2",
            "Max of router top-2",
            {"color": "#ff7f0e", "marker": "s", "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "router_top3",
            "Max of router top-3",
            {"color": "#2ca02c", "marker": "x", "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "random_member",
            "Random portfolio member",
            {"color": "#8c564b", "marker": "D", "linestyle": "-", "linewidth": 1.8},
        ),
        (
            "first_member",
            "First portfolio member",
            {"color": "#9467bd", "marker": "^", "linestyle": "-", "linewidth": 1.8},
        ),
    ]

def _plot_portfolio_router_ablation_legend(output_dir, stem):
    png_path = Path(output_dir) / f"{stem}_legend.png"
    pdf_path = Path(output_dir) / f"{stem}_legend.pdf"
    handles = []
    labels = []
    for _key, label, style in _portfolio_router_ablation_line_specs():
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.get("color"),
                marker=style.get("marker"),
                linestyle=style.get("linestyle", "-"),
                linewidth=style.get("linewidth", 1.8),
                markersize=7,
            )
        )
        labels.append(label)
    fig_width = max(14.0, 2.5 * len(labels))
    fig = plt.figure(figsize=(fig_width, 1.0))
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        columnspacing=1.4,
        handlelength=2.2,
    )
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}

def _plot_portfolio_router_ablation_metric(
    ks,
    metric_payload,
    metric,
    output_dir,
    stem,
    num_docs_to_fetch,
):
    if metric == "recall":
        y_label = "Support Recall"
    else:
        y_label = f"Exact match@{int(num_docs_to_fetch)} docs/member"
    png_path = Path(output_dir) / f"{stem}_{metric}.png"
    pdf_path = Path(output_dir) / f"{stem}_{metric}.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6.4, 4.0))
    for key, _label, style in _portfolio_router_ablation_line_specs():
        plt.plot(
            ks,
            metric_payload[key],
            color=style.get("color"),
            marker=style.get("marker"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.8),
            markersize=7,
        )
    plt.xlabel("Portfolio prefix size k")
    plt.ylabel(y_label)
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    return {"png": str(png_path), "pdf": str(pdf_path)}

def plot_portfolio_router_ablations(
    portfolio_id=C.POOL_SET_ALL_IMPLEMENTED,
    datasets=C.DATASETS,
    num_docs_to_fetch=4,
    max_k=5,
    split="test",
    run_ids=None,
    run_id_template=None,
    answer_llm=C.GEMMA27B,
    answers_path_template=None,
    strict=False,
    output_dir=None,
    plot_id=None,
):
    if isinstance(datasets, str):
        datasets = [value.strip() for value in datasets.split(",") if value.strip()]
    else:
        datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets provided for portfolio-router ablation plotting.")
    if split != "test":
        raise ValueError(f"Only split='test' is supported for router ablation plots, got {split!r}.")
    if max_k < 1:
        raise ValueError(f"max_k must be >= 1, got {max_k}")
    if num_docs_to_fetch < 1:
        raise ValueError(f"num_docs_to_fetch must be >= 1, got {num_docs_to_fetch}")
    if not answer_llm:
        raise ValueError("answer_llm is required for the EM plot.")

    run_ids_by_k = _resolve_router_run_ids(
        portfolio_id,
        datasets,
        num_docs_to_fetch,
        max_k,
        split,
        run_ids,
        run_id_template,
    )
    output_dir = Path(output_dir) if output_dir is not None else Path(C.PLOTS_DIR) / "average"

    per_dataset = {}
    failures = []
    for idx, dataset in enumerate(datasets, start=1):
        _progress_bar(idx - 1, len(datasets), "portfolio-router ablations: datasets")
        try:
            per_dataset[dataset] = _load_router_ablation_dataset_curves(
                portfolio_id=portfolio_id,
                dataset=dataset,
                num_docs=num_docs_to_fetch,
                max_k=max_k,
                split=split,
                run_ids_by_k=run_ids_by_k,
                answer_llm=answer_llm,
                answers_path_template=answers_path_template,
            )
        except (FileNotFoundError, ValueError, IndexError) as exc:
            message = (
                f"[plot-portfolio-router-ablations] skipping dataset={dataset}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(message, flush=True)
            failures.append(
                {
                    "dataset": dataset,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if strict:
                raise
        _progress_bar(idx, len(datasets), "portfolio-router ablations: datasets")

    if not per_dataset:
        details = " | ".join(
            f"{failure['dataset']}: {failure['error']}" for failure in failures
        )
        raise FileNotFoundError(
            "No complete datasets were available for portfolio-router ablation plotting. "
            + details
        )

    average = _average_router_ablation_payload(per_dataset)
    ks = list(range(1, int(max_k) + 1))
    safe_portfolio_id = _safe_plot_name(portfolio_id)
    safe_answer_llm = _safe_plot_name(answer_llm)
    suffix = _safe_plot_name(plot_id) if plot_id else f"{safe_portfolio_id}_{num_docs_to_fetch}_k{max_k}_{safe_answer_llm}"
    stem = f"portfolio_router_ablation_{suffix}"
    plot_paths = {
        "recall": _plot_portfolio_router_ablation_metric(
            ks,
            average["recall"],
            "recall",
            output_dir,
            stem,
            num_docs_to_fetch,
        ),
        "em": _plot_portfolio_router_ablation_metric(
            ks,
            average["em"],
            "em",
            output_dir,
            stem,
            num_docs_to_fetch,
        ),
        "legend": _plot_portfolio_router_ablation_legend(output_dir, stem),
    }
    payload_path = Path(output_dir) / f"{stem}_metrics.pickle"
    payload = {
        "schema": "portfolio_router_ablation_plots",
        "schema_version": 1,
        "portfolio_id": portfolio_id,
        "datasets_requested": datasets,
        "datasets_plotted": list(per_dataset.keys()),
        "num_docs": int(num_docs_to_fetch),
        "max_k": int(max_k),
        "ks": ks,
        "split": split,
        "run_ids_by_k": run_ids_by_k,
        "answer_llm": answer_llm,
        "answers_path_template": answers_path_template,
        "plotted_lines": [
            {"key": key, "label": label}
            for key, label, _style in _portfolio_router_ablation_line_specs()
        ],
        "per_dataset": per_dataset,
        "average": average,
        "failures": failures,
        "plot_paths": plot_paths,
        "payload_path": str(payload_path),
    }
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[plot-portfolio-router-ablations] saved plots: "
        f"recall={plot_paths['recall']['png']} em={plot_paths['em']['png']} "
        f"legend={plot_paths['legend']['png']} payload={payload_path}",
        flush=True,
    )
    return payload

# Vendi Resource-vs-Accuracy Plot

def _build_tokenizer(model_name: str):
    model_path = C.LLM_DIR.get(model_name, model_name)
    return AutoTokenizer.from_pretrained(model_path)

def _count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return len(encoded["input_ids"])

def _load_gold_answers(dataset_name: str):
    with open(C.get_questions_test(dataset_name), "rb") as f:
        questions_dataset = pickle.load(f)
    return questions_dataset.questions

def _load_answers_all(dataset_name: str, retriever: str, model_name: str, num_docs: int):
    answers_path = Path(C.get_answers_all(dataset_name, retriever, model_name, num_docs))
    if not answers_path.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_path}")
    with open(answers_path, "rb") as f:
        payload = pickle.load(f)
    return answers_path, payload.get("answers", [])

def _index_answers_by_question(answers):
    index = {}
    for entry in answers:
        if not entry:
            continue
        qidx = entry.get("question_idx")
        ridx = entry.get("retriever_idx")
        if qidx is None or ridx is None:
            continue
        index.setdefault(qidx, {})[ridx] = entry
    return index

def _response_em(response: str, gold: str) -> float:
    parsed = extract_tagged_answer(response, tag="answer")
    pred = parsed if parsed is not None else response
    return 1.0 if exact_match_score(pred, gold) else 0.0

def _response_f1(response: str, gold: str) -> float:
    parsed = extract_tagged_answer(response, tag="answer")
    pred = parsed if parsed is not None else response
    f1, _prec, _rec = f1_score(pred, gold)
    return f1

def _response_tagged_em(response: str, gold: str, tag: str) -> float:
    parsed = extract_tagged_answer(response, tag=tag)
    pred = parsed if parsed is not None else response
    return 1.0 if exact_match_score(pred, gold) else 0.0

def _response_tagged_f1(response: str, gold: str, tag: str) -> float:
    parsed = extract_tagged_answer(response, tag=tag)
    pred = parsed if parsed is not None else response
    f1, _prec, _rec = f1_score(pred, gold)
    return f1

def _select_router_best_points(router_points):
    if not router_points:
        return []
    by_k = {}
    for point in router_points:
        k = int(point["k"])
        by_k.setdefault(k, []).append(point)
    best_points = []
    for k in sorted(by_k.keys()):
        best = None
        for point in by_k[k]:
            if best is None:
                best = point
            elif point["avg_em"] > best["avg_em"]:
                best = point
            elif point["avg_em"] == best["avg_em"] and point["avg_tokens"] < best["avg_tokens"]:
                best = point
        if best is not None:
            best_points.append(best)
    return best_points

def _compute_baseline_point(entries, questions, tokenizer):
    per_q_tokens = []
    per_q_em = []
    per_q_f1 = []
    for entry in entries or []:
        if not entry:
            continue
        qidx = entry.get("question_idx")
        if qidx is None or qidx >= len(questions):
            continue
        gold = questions[qidx].get("answer")
        if gold is None:
            continue
        response = entry.get("response") or ""
        system_prompt = entry.get("system_prompt") or ""
        user_prompt = entry.get("user_prompt") or ""
        prompt_text = system_prompt + "\n" + user_prompt

        total = _count_tokens(tokenizer, prompt_text) + _count_tokens(tokenizer, response)
        per_q_tokens.append(total)
        per_q_em.append(_response_em(response, gold))
        per_q_f1.append(_response_f1(response, gold))

    if not per_q_tokens:
        return None
    return {
        "avg_tokens": sum(per_q_tokens) / len(per_q_tokens),
        "avg_em": sum(per_q_em) / len(per_q_em),
        "avg_f1": sum(per_q_f1) / len(per_q_f1),
        "count": len(per_q_tokens),
    }

def _get_efficiency_cache_path(dataset_name, retriever, num_docs_to_fetch, model_name):
    return (
        Path(C.RESULTS_DIR)
        / dataset_name
        / f"{retriever}_{num_docs_to_fetch}"
        / "plots"
        / f"tokens_vs_quality_{model_name}.pickle"
    )

def compute_tokens_vs_quality_data(
    dataset_name,
    model_name,
    retriever,
    num_docs_to_fetch,
    force=False,
):
    cache_path = _get_efficiency_cache_path(
        dataset_name, retriever, num_docs_to_fetch, model_name
    )
    if cache_path.exists() and not force:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        cache_version = payload.get("schema_version")
        if cache_version == 7:
            payload["cache_path"] = str(cache_path)
            return payload

    print(
        f"[tokens_vs_quality] dataset={dataset_name} retriever={retriever} "
        f"model={model_name} num_docs={num_docs_to_fetch}",
        flush=True,
    )
    print("[tokens_vs_quality] computing first portfolio-member answer tokens/EM...", flush=True)
    tokenizer = _build_tokenizer(model_name)
    questions = _load_gold_answers(dataset_name)

    answers_all_path, answers_all = _load_answers_all(
        dataset_name, retriever, model_name, num_docs_to_fetch
    )
    answers_index = _index_answers_by_question(answers_all)

    total_tokens = []
    em_scores = []
    f1_scores = []

    for qidx, q in enumerate(questions):
        gold = q.get("answer")
        if gold is None:
            continue
        entry = answers_index.get(qidx, {}).get(0)
        if entry is None or not entry.get("response"):
            continue

        response = entry.get("response") or ""
        system_prompt = entry.get("system_prompt") or ""
        user_prompt = entry.get("user_prompt") or ""
        prompt_text = system_prompt + "\n" + user_prompt

        prompt_tokens = _count_tokens(tokenizer, prompt_text)
        response_tokens = _count_tokens(tokenizer, response)
        total_tokens.append(prompt_tokens + response_tokens)

        em_scores.append(_response_em(response, gold))
        f1_scores.append(_response_f1(response, gold))

    avg_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0.0
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    print("[tokens_vs_quality] loading answers and selector outputs...", flush=True)
    baseline_points = []
    baseline_path = Path(C.get_answers_baseline(dataset_name, num_docs_to_fetch, model_name))
    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            baseline_payload = pickle.load(f) or {}
        no_entries = baseline_payload.get("no_retrieval_answers", [])
        naive_entries = baseline_payload.get("naive_retrieval_answers", [])
        no_point = _compute_baseline_point(no_entries, questions, tokenizer)
        if no_point:
            baseline_points.append({"label": "No retrieval", **no_point})
        naive_point = _compute_baseline_point(naive_entries, questions, tokenizer)
        if naive_point:
            baseline_points.append({"label": "Naive retrieval", **naive_point})
        print(
            f"[tokens_vs_quality] baseline points: {baseline_points}",
            flush=True,
        )
    else:
        print(
            f"[tokens_vs_quality] baseline answers missing: {baseline_path}",
            flush=True,
        )

    router_points = []
    # Single-point baseline: first retriever in the portfolio (k=1).
    per_q_tokens = []
    per_q_em = []
    per_q_f1 = []
    for qidx, q in enumerate(questions):
        gold = q.get("answer")
        if gold is None:
            continue
        entry = answers_index.get(qidx, {}).get(0)
        if entry is None or not entry.get("response"):
            continue
        prompt_text = (entry.get("system_prompt") or "") + "\n" + (entry.get("user_prompt") or "")
        response = entry.get("response") or ""
        total = _count_tokens(tokenizer, prompt_text) + _count_tokens(tokenizer, response)
        per_q_tokens.append(total)
        per_q_em.append(_response_em(response, gold))
        per_q_f1.append(_response_f1(response, gold))
    if per_q_tokens:
        avg_tok = sum(per_q_tokens) / len(per_q_tokens)
        avg_em = sum(per_q_em) / len(per_q_em)
        avg_f1 = sum(per_q_f1) / len(per_q_f1)
        router_points.append(
            {
                "k": 1,
                "ell": 1,
                "avg_tokens": avg_tok,
                "avg_em": avg_em,
                "avg_f1": avg_f1,
            }
        )

    selector_prompts_path = Path(
        C.get_selector_prompts(dataset_name, retriever, model_name, num_docs_to_fetch)
    )
    selector_answers_path = Path(
        C.get_answers_llm_selector(dataset_name, retriever, model_name, num_docs_to_fetch)
    )
    selector_lookup = {}
    if selector_prompts_path.exists() and selector_answers_path.exists():
        print(
            f"[tokens_vs_quality] selector prompts: {selector_prompts_path}",
            flush=True,
        )
        print(
            f"[tokens_vs_quality] selector answers: {selector_answers_path}",
            flush=True,
        )
        with open(selector_prompts_path, "rb") as f:
            selector_prompts_payload = pickle.load(f)
        with open(selector_answers_path, "rb") as f:
            selector_answers_payload = pickle.load(f)
        selector_prompts = selector_prompts_payload.get("selector_prompts", [])
        selector_answers = selector_answers_payload.get("answers", [])
        for idx, prompt in enumerate(selector_prompts):
            if idx >= len(selector_answers):
                break
            answer = selector_answers[idx]
            if not answer or not answer.get("response"):
                continue
            qidx = prompt.get("question_idx")
            subset_mask = prompt.get("subset_mask")
            if qidx is None or subset_mask is None:
                continue
            selector_lookup[(qidx, int(subset_mask))] = {
                "prompt": prompt,
                "answer": answer,
            }
        print(
            f"[tokens_vs_quality] selector cache size: {len(selector_lookup)}",
            flush=True,
        )
    else:
        print(
            "[tokens_vs_quality] selector prompts/answers missing; selector points will be empty.",
            flush=True,
        )

    selector_groups = {}
    for (qidx, mask), item in selector_lookup.items():
        if qidx >= len(questions):
            continue
        gold = questions[qidx].get("answer")
        if gold is None:
            continue
        prompt = item.get("prompt") or {}
        answer = item.get("answer") or {}
        response = answer.get("response") or ""
        if not response:
            continue

        subset_retrievers = prompt.get("subset_retrievers")
        if subset_retrievers is None:
            subset_retrievers = [
                ridx for ridx in range(mask.bit_length()) if mask & (1 << ridx)
            ]
        subset_retrievers = [int(ridx) for ridx in subset_retrievers]
        if not subset_retrievers:
            continue

        candidate_tokens = 0
        missing_candidate = False
        for ridx in subset_retrievers:
            entry = answers_index.get(qidx, {}).get(ridx)
            if entry is None or not entry.get("response"):
                missing_candidate = True
                break
            candidate_prompt = (entry.get("system_prompt") or "") + "\n" + (
                entry.get("user_prompt") or ""
            )
            candidate_tokens += _count_tokens(tokenizer, candidate_prompt)
            candidate_tokens += _count_tokens(tokenizer, entry.get("response") or "")
        if missing_candidate:
            continue

        selector_prompt_text = (prompt.get("system_prompt") or "") + "\n" + (
            prompt.get("user_prompt") or ""
        )
        selector_tokens = _count_tokens(tokenizer, selector_prompt_text) + _count_tokens(
            tokenizer,
            response,
        )
        k = max(subset_retrievers) + 1
        ell = len(subset_retrievers)
        group = selector_groups.setdefault(
            (k, ell),
            {"tokens": [], "em": [], "f1": []},
        )
        group["tokens"].append(candidate_tokens + selector_tokens)
        group["em"].append(_response_tagged_em(response, gold, tag="judge"))
        group["f1"].append(_response_tagged_f1(response, gold, tag="judge"))

    for (k, ell), group in sorted(selector_groups.items()):
        if not group["tokens"]:
            continue
        router_points.append(
            {
                "k": int(k),
                "ell": int(ell),
                "avg_tokens": sum(group["tokens"]) / len(group["tokens"]),
                "avg_em": sum(group["em"]) / len(group["em"]),
                "avg_f1": sum(group["f1"]) / len(group["f1"]),
                "count": len(group["tokens"]),
            }
        )

    router_best_points = []
    if router_points:
        router_best_points = _select_router_best_points(router_points)
        print(
            f"[tokens_vs_quality] best router points: {router_best_points}",
            flush=True,
        )

    vendi_points = []
    vendi_steps = []
    if retriever == C.VENDI:
        print("[tokens_vs_quality] computing VendiRAG early stopping points...", flush=True)
        max_steps_list = [2, 5, 10, 15, 20]
        summaries = evaluate_vendirag_early_stopping(
            dataset=dataset_name,
            num_docs=num_docs_to_fetch,
            model_key=model_name,
            max_steps_list=max_steps_list,
        )
        xs = [summaries[m]["avg_tokens"] for m in max_steps_list]
        ys = [summaries[m]["avg_em"] for m in max_steps_list]
        vendi_points = list(zip(xs, ys))
        vendi_steps = list(max_steps_list)

    vendi_pareto_points = []
    vendi_pareto_steps = []
    if vendi_points:
        indexed = [
            (step, point[0], point[1])
            for step, point in zip(vendi_steps, vendi_points)
        ]
        indexed.sort(key=lambda x: x[1])
        best_em = None
        for step, tok, em in indexed:
            if best_em is None or em > best_em:
                vendi_pareto_points.append((tok, em))
                vendi_pareto_steps.append(step)
                best_em = em

    payload = {
        "schema_version": 7,
        "answers_path": str(answers_all_path),
        "answers_all_path": str(answers_all_path),
        "avg_tokens": avg_tokens,
        "avg_em": avg_em,
        "avg_f1": avg_f1,
        "router_points": router_points,
        "router_best_points": router_best_points,
        "baseline_points": baseline_points,
        "vendi_points": vendi_points,
        "vendi_steps": vendi_steps,
        "vendi_pareto_points": vendi_pareto_points,
        "vendi_pareto_steps": vendi_pareto_steps,
        "cache_path": str(cache_path),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[tokens_vs_quality] saving cache: {cache_path}", flush=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return payload

def plot_tokens_vs_quality(model_name, retriever, num_docs_to_fetch, force=False):
    """
    Scatter plot of avg total tokens (x) vs avg EM (y). For Vendi, overlays
    early-stopping points from Vendi-RAG traces.
    """

    def _plot_from_payload(dataset_label, payload, title):
        avg_tokens = payload.get("avg_tokens", 0.0)
        avg_em = payload.get("avg_em", 0.0)
        router_points = payload.get("router_points", [])
        router_best_points = payload.get("router_best_points", [])
        baseline_points = payload.get("baseline_points", [])
        vendi_points = payload.get("vendi_points", [])

        plt.figure(figsize=(6, 4))
        pareto_points = []
        if router_points:
            plot_points = router_best_points or router_points
            sorted_by_tokens = sorted(plot_points, key=lambda p: p["avg_tokens"])
            best_em = None
            for point in sorted_by_tokens:
                if best_em is None or point["avg_em"] > best_em:
                    pareto_points.append(point)
                    best_em = point["avg_em"]
            xs = [p["avg_tokens"] for p in pareto_points]
            ys = [p["avg_em"] for p in pareto_points]
            plt.plot(
                xs,
                ys,
                label="Vendi-Portfolio",
                marker="*",
                linestyle="-",
                markersize=12,
            )

        if vendi_points:
            indexed = list(vendi_points)
            indexed.sort(key=lambda p: p[0])
            vendi_pareto = []
            best_em = None
            for tok, em in indexed:
                if best_em is None or em > best_em:
                    vendi_pareto.append((tok, em))
                    best_em = em
            xs = [p[0] for p in vendi_pareto]
            ys = [p[1] for p in vendi_pareto]
            plt.plot(xs, ys, label="Vendi-RAG Adaptive", marker="s", linestyle="-")

        if baseline_points:
            marker_map = {
                "No Retrieval": "o",
                "Nearest Neighbor Retrieval": "D",
            }
            color_map = {
                "No Retrieval": "#7f7f7f",
                "Nearest Neighbor Retrieval": "#9467bd",
            }
            for point in baseline_points:
                label = point.get("label", "Baseline")
                if label == "Naive retrieval":
                    label = "Nearest Neighbor Retrieval"
                if label == "No retrieval":
                    label = "No Retrieval"
                marker = marker_map.get(label, "o")
                color = color_map.get(label)
                plt.scatter(
                    [point["avg_tokens"]],
                    [point["avg_em"]],
                    label=label,
                    marker=marker,
                    s=70,
                    edgecolors="black",
                    linewidths=0.4,
                    color=color,
                )

        plt.xlabel("Average total tokens")
        plt.ylabel("Average EM")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.3)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.tight_layout()

        if dataset_label == "average":
            out_path = (
                Path(C.PLOTS_DIR)
                / "average"
                / f"tokens_vs_quality_average_{retriever}_{num_docs_to_fetch}_{model_name}.png"
            )
            legend_path = (
                Path(C.PLOTS_DIR)
                / "average"
                / f"tokens_vs_quality_average_{retriever}_{num_docs_to_fetch}_{model_name}_legend.png"
            )
        elif dataset_label in C.DATASETS:
            out_path = (
                Path(C.PLOTS_DIR)
                / dataset_label
                / f"tokens_vs_quality_{dataset_label}_{retriever}_{num_docs_to_fetch}_{model_name}.png"
            )
            legend_path = None
        else:
            out_path = Path(
                C.get_plot_path(
                    dataset_label,
                    "tokens_vs_quality",
                    retriever,
                    num_docs_to_fetch,
                    model_name,
                )
            )
            legend_path = None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        if legend_path and handles:
            legend_fig = plt.figure(figsize=(14, 1.2))
            legend_fig.legend(
                handles,
                labels,
                loc="center",
                ncol=len(labels),
                frameon=False,
            )
            legend_fig.tight_layout()
            legend_fig.savefig(legend_path, bbox_inches="tight")
            plt.close(legend_fig)
        plt.close()

        return {
            "answers_path": payload.get("answers_path"),
            "answers_all_path": payload.get("answers_all_path"),
            "avg_tokens": avg_tokens,
            "avg_em": avg_em,
            "router_points": router_points,
            "router_best_points": router_best_points,
            "baseline_points": baseline_points,
            "vendi_points": vendi_points,
            "cache_path": payload.get("cache_path"),
            "plot_path": str(out_path),
            "legend_path": str(legend_path) if legend_path else None,
        }

    def _average_payload(payloads):
        avg_tokens_vals = []
        avg_em_vals = []
        router_agg = {}
        baseline_agg = {}
        vendi_agg = {}
        for payload in payloads:
            avg_tokens_vals.append(float(payload.get("avg_tokens", 0.0)))
            avg_em_vals.append(float(payload.get("avg_em", 0.0)))

            for point in payload.get("router_points", []) or []:
                key = (int(point["k"]), int(point["ell"]))
                agg = router_agg.setdefault(key, {"tokens": 0.0, "em": 0.0, "count": 0})
                agg["tokens"] += float(point["avg_tokens"])
                agg["em"] += float(point["avg_em"])
                agg["count"] += 1

            for point in payload.get("baseline_points", []) or []:
                label = point.get("label", "Baseline")
                agg = baseline_agg.setdefault(label, {"tokens": 0.0, "em": 0.0, "count": 0})
                agg["tokens"] += float(point["avg_tokens"])
                agg["em"] += float(point["avg_em"])
                agg["count"] += 1

            vendi_steps = payload.get("vendi_steps", []) or []
            vendi_points = payload.get("vendi_points", []) or []
            for step, point in zip(vendi_steps, vendi_points):
                tok, em = point
                agg = vendi_agg.setdefault(int(step), {"tokens": 0.0, "em": 0.0, "count": 0})
                agg["tokens"] += float(tok)
                agg["em"] += float(em)
                agg["count"] += 1

        router_points = []
        for (k, ell), agg in sorted(router_agg.items()):
            count = agg["count"]
            if count <= 0:
                continue
            router_points.append(
                {
                    "k": k,
                    "ell": ell,
                    "avg_tokens": agg["tokens"] / count,
                    "avg_em": agg["em"] / count,
                    "count": count,
                }
            )

        baseline_points = []
        for label, agg in baseline_agg.items():
            count = agg["count"]
            if count <= 0:
                continue
            baseline_points.append(
                {
                    "label": label,
                    "avg_tokens": agg["tokens"] / count,
                    "avg_em": agg["em"] / count,
                    "count": count,
                }
            )

        vendi_points = []
        vendi_steps = []
        for step in sorted(vendi_agg.keys()):
            agg = vendi_agg[step]
            count = agg["count"]
            if count <= 0:
                continue
            vendi_steps.append(step)
            vendi_points.append((agg["tokens"] / count, agg["em"] / count))

        avg_tokens = sum(avg_tokens_vals) / len(avg_tokens_vals) if avg_tokens_vals else 0.0
        avg_em = sum(avg_em_vals) / len(avg_em_vals) if avg_em_vals else 0.0

        return {
            "avg_tokens": avg_tokens,
            "avg_em": avg_em,
            "router_points": router_points,
            "router_best_points": _select_router_best_points(router_points),
            "baseline_points": baseline_points,
            "vendi_points": vendi_points,
            "vendi_steps": vendi_steps,
        }

    payloads = []
    for name in C.DATASETS:
        payload = compute_tokens_vs_quality_data(
            dataset_name=name,
            model_name=model_name,
            retriever=retriever,
            num_docs_to_fetch=num_docs_to_fetch,
            force=force,
        )
        payloads.append(payload)

    avg_payload = _average_payload(payloads)
    avg_plot = _plot_from_payload(
        "average",
        avg_payload,
        f"Tokens vs EM: Average over datasets / {model_name}",
    )
    return {"average": avg_plot}

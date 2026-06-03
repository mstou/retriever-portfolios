import argparse
import os
import pickle
from pathlib import Path

import constants as C
from experiment_utils import (
    answer_portfolio_union_prompts_with_llm,
    answer_family_best_prompts_with_llm,
    answer_prompts_with_llm,
    answer_selector_prompts_with_llm,
    answer_baseline_prompts_with_llm,
    build_answer_prompts,
    build_baseline_answer_prompts,
    build_family_best_answer_prompts,
    build_portfolio_union_answer_prompts,
    build_portfolio_router_judge_prompts,
    build_selector_prompts,
    build_graph_query_entity_cache_from_extraction_results,
    answer_portfolio_router_judge_prompts_with_llm,
    audit_pool_artifacts,
    compute_full_pool_recalls_to_file,
    compute_recalls_to_file,
    compute_retrievals_train,
    compute_retrievals_test,
    compute_family_best_test_retrievals,
    compute_portfolio_retrievals_test,
    compute_single_retriever_retrievals,
    compute_universal_portfolio,
    compute_universal_portfolio_union,
    index_corpus,
    materialize_portfolio_test,
    questions_train_test_split,
    save_embeddings,
    save_prefilters,
    select_family_best_retrievers,
)
from graph_index import GraphIndex
from models import OpenAI_LLM
from prompts import answer_prompt
from text_processing import Embedder
from train_portfolio_router import (
    train_portfolio_router,
    write_portfolio_router_test_predictions_from_checkpoint,
)
from vector_db import FaissVectorDB
from vendi_rag import VendiRAGAdaptive


def _parse_list(values, allowed, label):
    if not values:
        return list(allowed)
    selected = []
    for raw in values.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in allowed:
            raise SystemExit(f"Unknown {label}: {name}. Allowed: {', '.join(allowed)}")
        selected.append(name)
    return selected


def _parse_splits(values):
    return _parse_list(values, ["train", "test"], "split")


def _parse_int_list(values, default, label):
    if values is None or str(values).strip().lower() in {"", "all"}:
        return list(default)
    selected = []
    for raw in str(values).split(","):
        value = raw.strip()
        if not value:
            continue
        try:
            selected.append(int(value))
        except ValueError as exc:
            raise SystemExit(f"Invalid {label}: {value!r}. Expected comma-separated integers.") from exc
    if not selected:
        raise SystemExit(f"No {label} values provided.")
    return selected


def _parse_llm_list(values, default=None):
    allowed = [C.GEMMA27B, C.LLAMA70B]
    if default is None:
        default = allowed
    return _parse_list(values, allowed, "LLM") if values else list(default)


def _parse_pool_ids(values):
    return _parse_list(values, sorted(C.POOL_CATALOG), "pool")


def _parse_families(values):
    return _parse_list(values, [C.DS, C.VENDI, C.GRAPH_DENSE], "family")


def _resolve_pool_specs(pool_set, pool_ids):
    if pool_ids:
        return C.get_pool_specs(_parse_pool_ids(pool_ids))
    return C.get_pool_specs_for_set(pool_set)


def _get_api_key(explicit_key):
    return explicit_key or os.environ.get("OPENAI_API_KEY", "EMPTY")


def _build_llm(llm_key, api_key):
    model_path = C.LLM_DIR[llm_key]
    base_url = C.LLM_BASE_URL[llm_key]
    return OpenAI_LLM(model_name=model_path, api_key=api_key, base_url=base_url)


def _translated_artifact_path(path):
    return Path(path)


def _artifact_read_path(path):
    expected = Path(path)
    if expected.exists():
        return expected
    translated = _translated_artifact_path(expected)
    if translated != expected and translated.exists():
        return translated
    return expected


def _portfolio_router_prediction_parent(portfolio_id, dataset_name, num_docs, portfolio_size, split="test"):
    return (
        C.get_portfolio_union_dir(dataset_name, portfolio_id, num_docs)
        + f"portfolio_router_predictions/k{int(portfolio_size)}/{split}/"
    )


def _available_portfolio_router_prediction_runs(
    portfolio_id,
    dataset_name,
    num_docs,
    portfolio_size,
    split="test",
):
    parent = _artifact_read_path(
        _portfolio_router_prediction_parent(
            portfolio_id,
            dataset_name,
            num_docs,
            portfolio_size,
            split=split,
        )
    )
    if not parent.exists() or not parent.is_dir():
        return {}
    runs = {}
    for child in parent.iterdir():
        if not child.is_dir():
            continue
        prediction_file = child / "predictions.pickle"
        if prediction_file.exists():
            runs[child.name] = prediction_file.stat().st_mtime
    return runs


def _discover_common_portfolio_router_run_id(
    *,
    portfolio_id,
    datasets,
    num_docs,
    portfolio_size,
    split="test",
):
    available_by_dataset = {
        dataset_name: _available_portfolio_router_prediction_runs(
            portfolio_id,
            dataset_name,
            num_docs,
            portfolio_size,
            split=split,
        )
        for dataset_name in datasets
    }
    common = None
    for runs in available_by_dataset.values():
        run_ids = set(runs)
        common = run_ids if common is None else common & run_ids
    common = common or set()
    non_smoke = {run_id for run_id in common if "smoke" not in run_id.lower()}
    candidates = non_smoke or common
    if not candidates:
        details = []
        for dataset_name, runs in available_by_dataset.items():
            values = ", ".join(sorted(runs)) if runs else "<none>"
            details.append(f"{dataset_name}: {values}")
        raise SystemExit(
            "Could not auto-discover a common portfolio-router prediction run id "
            f"for portfolio_id={portfolio_id}, k={portfolio_size}, split={split}. "
            "Available runs by dataset: " + " | ".join(details)
        )

    def sort_key(run_id):
        mtimes = [available_by_dataset[dataset_name][run_id] for dataset_name in datasets]
        return (
            1 if "best" in run_id.lower() else 0,
            min(mtimes),
            run_id,
        )

    chosen = sorted(candidates, key=sort_key, reverse=True)[0]
    if len(candidates) > 1:
        print(
            f"[MAIN] Auto-discovered multiple common run_ids for k={portfolio_size}; "
            f"using {chosen}. candidates={sorted(candidates)}",
            flush=True,
        )
    else:
        print(
            f"[MAIN] Auto-discovered run_id for k={portfolio_size}: {chosen}",
            flush=True,
        )
    return chosen


def _parse_run_id_map(raw):
    if not raw:
        return {}
    mapping = {}
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"Invalid --run-id-map entry {part!r}. Expected format like 2:run_for_k2,3:run_for_k3."
            )
        k_raw, run_id = part.split(":", 1)
        try:
            k = int(k_raw.strip())
        except ValueError as exc:
            raise SystemExit(f"Invalid k in --run-id-map entry {part!r}.") from exc
        run_id = run_id.strip()
        if not run_id:
            raise SystemExit(f"Missing run_id in --run-id-map entry {part!r}.")
        mapping[k] = run_id
    return mapping


def _resolve_ell_values_for_k(k, explicit_ells, min_ell, max_ell, include_ell1):
    if explicit_ells:
        values = list(explicit_ells)
    else:
        start = 1 if include_ell1 else int(min_ell)
        stop = int(max_ell) if max_ell is not None else int(k)
        values = list(range(start, stop + 1))
    valid = []
    for ell in values:
        if ell < 1 or ell > k:
            raise SystemExit(f"Invalid ell={ell} for k={k}. Expected 1 <= ell <= k.")
        if ell == 1 and not include_ell1 and not explicit_ells:
            continue
        valid.append(ell)
    if not valid:
        raise SystemExit(f"No valid ell values resolved for k={k}.")
    return valid


def run_prep(datasets, device, prefilter_num, embedder):
    for dataset_name in datasets:
        print(f"[MAIN] Dataset prep: {dataset_name}, embedder={embedder}", flush=True)
        index_corpus(dataset_name, device, embedder=embedder)
        questions_train_test_split(dataset_name)
        save_embeddings(dataset_name, device, embedder=embedder)
        save_prefilters(dataset_name, prefilter_size=prefilter_num, embedder=embedder)


def run_build_graph_index(datasets, max_workers, checkpoint_every):
    for dataset_name in datasets:
        print(
            f"[MAIN] Build graph index: dataset={dataset_name}, "
            f"max_workers={max_workers}, checkpoint_every={checkpoint_every}",
            flush=True,
        )
        graph_index = GraphIndex()
        graph_index.build(
            dataset_name,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
        )
        output_path = C.get_graph_index_path(dataset_name)
        graph_index.save(output_path)
        print(
            f"[MAIN] Saved graph index: dataset={dataset_name}, "
            f"chunks={graph_index.num_chunks}, entities={len(graph_index.entity_to_chunk_keys)}, "
            f"path={output_path}",
            flush=True,
        )


def run_build_graph_query_entity_cache(datasets, splits, overwrite):
    for dataset_name in datasets:
        for split in splits:
            print(
                f"[MAIN] Build graph query entity cache: dataset={dataset_name}, "
                f"split={split}, overwrite={overwrite}",
                flush=True,
            )
            output_path = build_graph_query_entity_cache_from_extraction_results(
                dataset_name=dataset_name,
                split=split,
                overwrite=overwrite,
            )
            print(
                f"[MAIN] Saved graph query entity cache: dataset={dataset_name}, "
                f"split={split}, path={output_path}",
                flush=True,
            )


def run_train_retrievals(datasets, retrievers, num_docs, device, embedder, prefilter_num):
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Train retrievals: dataset={dataset_name}, retriever={retriever}, "
                f"embedder={embedder}, prefilter_num={prefilter_num}",
                flush=True,
            )
            compute_retrievals_train(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                device=device,
                embedder=embedder,
                prefilter_num=prefilter_num,
            )
            compute_recalls_to_file(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                split="train",
                embedder=embedder,
            )


def run_full_pool_recalls(datasets, retrievers, num_docs, splits, embedder):
    for dataset_name in datasets:
        for split in splits:
            for retriever in retrievers:
                print(
                    f"[MAIN] Full-pool recalls: dataset={dataset_name}, split={split}, "
                    f"retriever={retriever}, embedder={embedder}",
                    flush=True,
                )
                compute_full_pool_recalls_to_file(
                    dataset_name=dataset_name,
                    retriever=retriever,
                    num_docs_to_fetch=num_docs,
                    split=split,
                    embedder=embedder,
                )


def run_universal_portfolio(retrievers, num_docs, portfolio_size, device, embedder):
    for retriever in retrievers:
        print(f"[MAIN] Universal portfolio: retriever={retriever}, embedder={embedder}", flush=True)
        compute_universal_portfolio(
            retriever=retriever,
            num_docs_to_fetch=num_docs,
            portfolio_size=portfolio_size,
            device=device,
            embedder=embedder,
        )


def run_pool_artifact_audit(datasets, num_docs, pool_set, pools, strict):
    pool_specs = _resolve_pool_specs(pool_set, pools)
    return audit_pool_artifacts(
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        pool_specs=pool_specs,
        strict=strict,
    )


def run_universal_portfolio_union(datasets, num_docs, portfolio_size, device, pool_set, pools, portfolio_id):
    pool_specs = _resolve_pool_specs(pool_set, pools)
    resolved_portfolio_id = portfolio_id or (pool_set if not pools else "custom")
    print(
        f"[MAIN] Universal portfolio union: id={resolved_portfolio_id}, "
        f"datasets={','.join(datasets)}, pools={','.join(spec['pool_id'] for spec in pool_specs)}",
        flush=True,
    )
    output_path = compute_universal_portfolio_union(
        pool_specs=pool_specs,
        num_docs_to_fetch=num_docs,
        portfolio_size=portfolio_size,
        device=device,
        datasets=datasets,
        union_name=resolved_portfolio_id,
    )
    print(f"[MAIN] Universal portfolio union saved: {output_path}", flush=True)
    return output_path


def run_materialize_portfolio_test(datasets, num_docs, portfolio_id, portfolio_path, strict):
    print(
        f"[MAIN] Materialize portfolio test: id={portfolio_id} "
        f"path={portfolio_path or '-'} datasets={','.join(datasets)} "
        f"num_docs={num_docs} strict={strict}",
        flush=True,
    )
    summary = materialize_portfolio_test(
        portfolio_path=portfolio_path,
        portfolio_id=portfolio_id,
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        strict=strict,
    )
    print(
        f"[MAIN] Materialized datasets: "
        f"{','.join(summary['datasets_materialized']) or '-'}",
        flush=True,
    )
    if summary["failures"]:
        print(f"[MAIN] Materialization failures: {len(summary['failures'])}", flush=True)
        for failure in summary["failures"]:
            print(
                f"  dataset={failure['dataset']} "
                f"error_type={failure['error_type']} error={failure['error']}",
                flush=True,
        )
    return summary


def run_select_family_best_baselines(datasets, portfolio_id, num_docs, families, device):
    print(
        f"[MAIN] Select family-best baselines: portfolio_id={portfolio_id}, "
        f"datasets={','.join(datasets)}, families={','.join(families)}, "
        f"num_docs={num_docs}",
        flush=True,
    )
    output_path = select_family_best_retrievers(
        portfolio_id=portfolio_id,
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        families=families,
        device=device,
    )
    print(f"[MAIN] Family-best manifest saved: {output_path}", flush=True)
    return output_path


def run_compute_family_best_baselines_test(
    datasets,
    portfolio_id,
    num_docs,
    max_k,
    families,
    device,
    prefilter_num,
    save_every,
):
    print(
        f"[MAIN] Compute family-best baseline test artifacts: "
        f"portfolio_id={portfolio_id}, datasets={','.join(datasets)}, "
        f"families={','.join(families)}, num_docs={num_docs}, "
        f"max_k={max_k}, device={device}, prefilter_num={prefilter_num}, "
        f"save_every={save_every}",
        flush=True,
    )
    summary = compute_family_best_test_retrievals(
        portfolio_id=portfolio_id,
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        max_k=max_k,
        families=families,
        device=device,
        prefilter_num=prefilter_num,
        save_every=save_every,
    )
    print(
        f"[MAIN] Family-best baseline outputs written: {len(summary['outputs'])}",
        flush=True,
    )
    return summary


def run_build_family_best_answer_prompts(
    datasets,
    portfolio_id,
    families,
    num_docs,
    max_k,
    max_questions,
):
    for dataset_name in datasets:
        for family in families:
            print(
                f"[MAIN] Build family-best answer prompts: dataset={dataset_name}, "
                f"portfolio_id={portfolio_id}, family={family}, num_docs={num_docs}, "
                f"max_k={max_k}, max_questions={max_questions}",
                flush=True,
            )
            out_path = build_family_best_answer_prompts(
                portfolio_id=portfolio_id,
                dataset_name=dataset_name,
                family=family,
                num_docs_to_fetch=num_docs,
                max_k=max_k,
                max_questions=max_questions,
            )
            print(f"[MAIN] Family-best prompts saved: {out_path}", flush=True)


def run_answer_family_best_prompts(
    datasets,
    portfolio_id,
    families,
    num_docs,
    max_k,
    llm_key,
    max_workers=16,
    checkpoint_every=1000,
):
    api_key = _get_api_key(None)
    llm = _build_llm(llm_key, api_key)
    for dataset_name in datasets:
        for family in families:
            print(
                f"[MAIN] Answer family-best prompts: dataset={dataset_name}, "
                f"portfolio_id={portfolio_id}, family={family}, num_docs={num_docs}, "
                f"max_k={max_k}, llm={llm_key}, max_workers={max_workers}, "
                f"checkpoint_every={checkpoint_every}",
                flush=True,
            )
            out_path = answer_family_best_prompts_with_llm(
                portfolio_id=portfolio_id,
                dataset_name=dataset_name,
                family=family,
                num_docs_to_fetch=num_docs,
                max_k=max_k,
                llm=llm,
                llm_name=llm_key,
                max_workers=max_workers,
                checkpoint_every=checkpoint_every,
            )
            print(f"[MAIN] Family-best answers saved: {out_path}", flush=True)


def run_test_retrievals(datasets, retrievers, num_docs, portfolio_size, device, prefilter_num, embedder):
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Universal test retrievals: dataset={dataset_name}, retriever={retriever}, embedder={embedder}",
                flush=True,
            )
            compute_portfolio_retrievals_test(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                device=device,
                prefilter_num=prefilter_num,
                universal=True,
                embedder=embedder,
            )
            compute_single_retriever_retrievals(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                portfolio_size=portfolio_size,
                split="test",
                device=device,
                prefilter_num=prefilter_num,
                universal=True,
                embedder=embedder,
            )


def run_test_retrievals_pool(datasets, retrievers, num_docs, device, prefilter_num, embedder, compute_recalls=False):
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Test retrievals (pool): dataset={dataset_name}, retriever={retriever}, embedder={embedder}",
                flush=True,
            )
            compute_retrievals_test(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                device=device,
                prefilter_num=prefilter_num,
                embedder=embedder,
            )
            if compute_recalls:
                compute_full_pool_recalls_to_file(
                    dataset_name=dataset_name,
                    retriever=retriever,
                    num_docs_to_fetch=num_docs,
                    split="test",
                    embedder=embedder,
                )


def run_single_retriever_test(datasets, retrievers, num_docs, portfolio_size, device, prefilter_num, universal, embedder):
    for dataset_name in datasets:
        for retriever in retrievers:
            scope = "universal" if universal else "dataset"
            print(
                f"[MAIN] Single retriever test: dataset={dataset_name}, retriever={retriever}, scope={scope}, embedder={embedder}",
                flush=True,
            )
            compute_single_retriever_retrievals(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                portfolio_size=portfolio_size,
                split="test",
                device=device,
                prefilter_num=prefilter_num,
                universal=universal,
                embedder=embedder,
            )


def run_build_answer_prompts(datasets, retrievers, num_docs, portfolio_size, embedder):
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Build answer prompts: dataset={dataset_name}, retriever={retriever}, embedder={embedder}",
                flush=True,
            )
            build_answer_prompts(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                portfolio_size=portfolio_size,
                universal=True,
                embedder=embedder,
            )


def run_build_baseline_prompts(datasets, num_docs, device, prefilter_num):
    for dataset_name in datasets:
        print(
            f"[MAIN] Build baseline prompts: dataset={dataset_name}, num_docs={num_docs}",
            flush=True,
        )
        out_path = build_baseline_answer_prompts(
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs,
            device=device,
            pre_filter=prefilter_num,
        )
        print(f"[MAIN] Baseline prompts saved: {out_path}", flush=True)


def run_answer_prompts(datasets, retrievers, num_docs, llm_key, embedder, max_workers=16, checkpoint_every=1000):
    api_key = _get_api_key(None)
    llm = _build_llm(llm_key, api_key)
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Answer prompts: dataset={dataset_name}, retriever={retriever}, llm={llm_key}, embedder={embedder}",
                flush=True,
            )
            answer_prompts_with_llm(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                llm=llm,
                llm_name=llm_key,
                max_workers=max_workers,
                checkpoint_every=checkpoint_every,
                embedder=embedder,
            )


def run_build_portfolio_union_answer_prompts(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_size,
    max_questions,
):
    for dataset_name in datasets:
        print(
            f"[MAIN] Build all-pool portfolio answer prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, num_docs={num_docs}, "
            f"portfolio_size={portfolio_size}, max_questions={max_questions}",
            flush=True,
        )
        out_path = build_portfolio_union_answer_prompts(
            portfolio_id=portfolio_id,
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs,
            portfolio_size=portfolio_size,
            max_questions=max_questions,
        )
        print(f"[MAIN] All-pool portfolio prompts saved: {out_path}", flush=True)


def run_answer_portfolio_union_prompts(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_size,
    llm_key,
    max_workers=16,
    checkpoint_every=1000,
):
    api_key = _get_api_key(None)
    llm = _build_llm(llm_key, api_key)
    for dataset_name in datasets:
        print(
            f"[MAIN] Answer all-pool portfolio prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, num_docs={num_docs}, "
            f"portfolio_size={portfolio_size}, llm={llm_key}, "
            f"max_workers={max_workers}, checkpoint_every={checkpoint_every}",
            flush=True,
        )
        out_path = answer_portfolio_union_prompts_with_llm(
            portfolio_id=portfolio_id,
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs,
            llm=llm,
            llm_name=llm_key,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
            portfolio_size=portfolio_size,
        )
        print(f"[MAIN] All-pool portfolio answers saved: {out_path}", flush=True)


def run_train_portfolio_router(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_size,
    output_dir,
    device,
    batch_size,
    max_length,
    epochs,
    lr,
    weight_decay,
    dev_ratio,
    seed,
    grad_clip,
    grad_accum_steps,
    max_train_questions,
    max_dev_questions,
    max_test_questions,
    load_test,
    t5_model_name_or_path,
    tokenizer_name_or_path,
    use_wandb,
    wandb_run_name,
    resume_from,
    auto_resume,
):
    train_portfolio_router(
        portfolio_id=portfolio_id,
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        portfolio_size=portfolio_size,
        output_dir=output_dir,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        dev_ratio=dev_ratio,
        seed=seed,
        grad_clip=grad_clip,
        grad_accum_steps=grad_accum_steps,
        max_train_questions=max_train_questions,
        max_dev_questions=max_dev_questions,
        max_test_questions=max_test_questions,
        load_test=load_test,
        t5_model_name_or_path=t5_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        use_wandb=use_wandb,
        wandb_run_name=wandb_run_name,
        resume_from=resume_from,
        auto_resume=auto_resume,
    )


def run_portfolio_router_predict_test(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_size,
    run_id,
    checkpoint_path,
    device,
    batch_size,
    max_questions,
    strict,
    t5_model_name_or_path,
):
    write_portfolio_router_test_predictions_from_checkpoint(
        portfolio_id=portfolio_id,
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        portfolio_size=portfolio_size,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        device=device,
        batch_size=batch_size,
        max_questions=max_questions,
        strict=strict,
        t5_model_name_or_path=t5_model_name_or_path,
    )


def run_build_portfolio_router_judge_prompts(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_size,
    ell,
    run_id,
    answer_llm,
    max_questions,
    strict_answers,
):
    for dataset_name in datasets:
        print(
            f"[MAIN] Build portfolio-router judge prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, num_docs={num_docs}, k={portfolio_size}, "
            f"ell={ell}, run_id={run_id}, answer_llm={answer_llm}, "
            f"max_questions={max_questions}, strict_answers={strict_answers}",
            flush=True,
        )
        out_path = build_portfolio_router_judge_prompts(
            portfolio_id=portfolio_id,
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs,
            portfolio_size=portfolio_size,
            ell=ell,
            run_id=run_id,
            answer_llm=answer_llm,
            max_questions=max_questions,
            strict_answers=strict_answers,
        )
        print(f"[MAIN] Portfolio-router judge prompts saved: {out_path}", flush=True)


def run_build_all_portfolio_router_judge_prompts(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_sizes,
    ell_values,
    min_ell,
    max_ell,
    include_ell1,
    run_id_map,
    answer_llms,
    max_questions,
    strict_answers,
    dry_run,
):
    run_id_by_k = _parse_run_id_map(run_id_map)
    explicit_ells = _parse_int_list(ell_values, [], "ell") if ell_values else None
    total = 0

    for k in portfolio_sizes:
        if k <= 0:
            raise SystemExit(f"portfolio sizes must be positive; got {k}.")
        run_id = run_id_by_k.get(k)
        if run_id is None:
            run_id = _discover_common_portfolio_router_run_id(
                portfolio_id=portfolio_id,
                datasets=datasets,
                num_docs=num_docs,
                portfolio_size=k,
                split="test",
            )
        else:
            print(f"[MAIN] Using explicit run_id for k={k}: {run_id}", flush=True)

        ells = _resolve_ell_values_for_k(
            k,
            explicit_ells,
            min_ell,
            max_ell,
            include_ell1,
        )
        for answer_llm in answer_llms:
            for ell in ells:
                total += 1
                print(
                    f"[MAIN] Build-all portfolio-router judge prompts: "
                    f"k={k}, ell={ell}, run_id={run_id}, answer_llm={answer_llm}, "
                    f"datasets={','.join(datasets)}, dry_run={dry_run}",
                    flush=True,
                )
                if dry_run:
                    continue
                run_build_portfolio_router_judge_prompts(
                    datasets=datasets,
                    portfolio_id=portfolio_id,
                    num_docs=num_docs,
                    portfolio_size=k,
                    ell=ell,
                    run_id=run_id,
                    answer_llm=answer_llm,
                    max_questions=max_questions,
                    strict_answers=strict_answers,
                )

    print(f"[MAIN] Build-all portfolio-router judge prompt runs completed: {total}", flush=True)


def run_answer_portfolio_router_judge_prompts(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_size,
    ell,
    run_id,
    answer_llm,
    judge_llm_key,
    max_workers=16,
    checkpoint_every=1000,
):
    api_key = _get_api_key(None)
    judge_llm = _build_llm(judge_llm_key, api_key)
    for dataset_name in datasets:
        print(
            f"[MAIN] Answer portfolio-router judge prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, num_docs={num_docs}, k={portfolio_size}, "
            f"ell={ell}, run_id={run_id}, answer_llm={answer_llm}, "
            f"judge_llm={judge_llm_key}, max_workers={max_workers}, "
            f"checkpoint_every={checkpoint_every}",
            flush=True,
        )
        out_path = answer_portfolio_router_judge_prompts_with_llm(
            portfolio_id=portfolio_id,
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs,
            portfolio_size=portfolio_size,
            ell=ell,
            run_id=run_id,
            answer_llm=answer_llm,
            judge_llm_name=judge_llm_key,
            judge_llm=judge_llm,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
        )
        print(f"[MAIN] Portfolio-router judge answers saved: {out_path}", flush=True)


def run_answer_all_portfolio_router_judge_prompts(
    datasets,
    portfolio_id,
    num_docs,
    portfolio_sizes,
    ell_values,
    min_ell,
    max_ell,
    run_id_map,
    answer_llms,
    judge_llm_key,
    max_workers,
    checkpoint_every,
    dry_run,
):
    run_id_by_k = _parse_run_id_map(run_id_map)
    explicit_ells = _parse_int_list(ell_values, [], "ell") if ell_values else None
    total = 0

    for k in portfolio_sizes:
        if k <= 0:
            raise SystemExit(f"portfolio sizes must be positive; got {k}.")
        run_id = run_id_by_k.get(k)
        if run_id is None:
            run_id = _discover_common_portfolio_router_run_id(
                portfolio_id=portfolio_id,
                datasets=datasets,
                num_docs=num_docs,
                portfolio_size=k,
                split="test",
            )
        else:
            print(f"[MAIN] Using explicit run_id for k={k}: {run_id}", flush=True)

        ells = _resolve_ell_values_for_k(
            k,
            explicit_ells,
            min_ell,
            max_ell,
            include_ell1=False,
        )
        for answer_llm in answer_llms:
            for ell in ells:
                total += 1
                print(
                    f"[MAIN] Answer-all portfolio-router judge prompts: "
                    f"k={k}, ell={ell}, run_id={run_id}, answer_llm={answer_llm}, "
                    f"judge_llm={judge_llm_key}, datasets={','.join(datasets)}, "
                    f"max_workers={max_workers}, checkpoint_every={checkpoint_every}, "
                    f"dry_run={dry_run}",
                    flush=True,
                )
                if dry_run:
                    continue
                run_answer_portfolio_router_judge_prompts(
                    datasets=datasets,
                    portfolio_id=portfolio_id,
                    num_docs=num_docs,
                    portfolio_size=k,
                    ell=ell,
                    run_id=run_id,
                    answer_llm=answer_llm,
                    judge_llm_key=judge_llm_key,
                    max_workers=max_workers,
                    checkpoint_every=checkpoint_every,
                )

    print(f"[MAIN] Answer-all portfolio-router judge prompt runs completed: {total}", flush=True)


def run_answer_baseline_prompts(datasets, num_docs, llm_key, max_workers=64, checkpoint_every=1000):
    api_key = _get_api_key(None)
    llm = _build_llm(llm_key, api_key)
    for dataset_name in datasets:
        print(
            f"[MAIN] Answer baseline prompts: dataset={dataset_name}, llm={llm_key}",
            flush=True,
        )
        answer_baseline_prompts_with_llm(
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs,
            llm=llm,
            llm_name=llm_key,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
        )


def run_build_selector_prompts(datasets, retrievers, num_docs, portfolio_size, llm_key, embedder):
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Build selector prompts: dataset={dataset_name}, retriever={retriever}, llm={llm_key}, embedder={embedder}",
                flush=True,
            )
            build_selector_prompts(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                portfolio_size=portfolio_size,
                llm_name=llm_key,
                embedder=embedder,
            )


def run_answer_selector_prompts(datasets, retrievers, num_docs, llm_key, embedder, max_workers=16, checkpoint_every=1000):
    api_key = _get_api_key(None)
    llm = _build_llm(llm_key, api_key)
    for dataset_name in datasets:
        for retriever in retrievers:
            print(
                f"[MAIN] Answer selector prompts: dataset={dataset_name}, retriever={retriever}, llm={llm_key}, embedder={embedder}",
                flush=True,
            )
            answer_selector_prompts_with_llm(
                dataset_name=dataset_name,
                retriever=retriever,
                num_docs_to_fetch=num_docs,
                llm=llm,
                llm_name=llm_key,
                max_workers=max_workers,
                checkpoint_every=checkpoint_every,
                embedder=embedder,
            )


def run_vendi_rag(
    datasets,
    num_docs,
    llm_key,
    device,
    num_steps,
    initial_s,
    max_questions,
    start_idx,
    max_workers,
    checkpoint_every,
    embedder,
):
    api_key = _get_api_key(None)
    llm = _build_llm(llm_key, api_key)

    for dataset_name in datasets:
        print(
            f"[MAIN] VendiRAG: dataset={dataset_name}, docs={num_docs}, llm={llm_key}, embedder={embedder}",
            flush=True,
        )
        vector_db = FaissVectorDB.load(C.get_vector_db_dir(dataset_name, embedder=embedder))
        with open(C.get_questions_test(dataset_name), "rb") as f:
            questions_dataset = pickle.load(f)

        vendi_runner = VendiRAGAdaptive(
            embedder=Embedder(device, embedder=embedder),
            vector_db=vector_db,
            answer_llm=llm,
            judge_llm=llm,
            answer_prompt_fn=answer_prompt,
            num_steps=num_steps,
            k_docs=num_docs,
            device=device,
            questions_dataset=questions_dataset,
        )

        out_path = C.get_vendirag_results(dataset_name, num_docs, llm_key, embedder=embedder)
        vendi_runner.run_dataset(
            initial_s=initial_s,
            max_questions=max_questions,
            start_idx=start_idx,
            max_workers=max_workers,
            output_file=out_path,
            checkpoint_every=checkpoint_every,
        )
        print(
            f"[MAIN] Saved VendiRAG results for dataset={dataset_name} to {out_path}",
            flush=True,
        )


def run_plot_all_pool_support(
    datasets,
    portfolio_id,
    num_docs,
    max_k,
    compute_missing_f1,
    include_vendi_family_best,
    strict,
    output_dir,
):
    from plot_utils import plot_all_pool_support_paper

    return plot_all_pool_support_paper(
        portfolio_id=portfolio_id,
        datasets=datasets,
        num_docs_to_fetch=num_docs,
        max_k=max_k,
        compute_missing_f1=compute_missing_f1,
        include_vendi_family_best=include_vendi_family_best,
        strict=strict,
        output_dir=output_dir,
    )


def run_plot_portfolio_router_ablations(
    datasets,
    portfolio_id,
    num_docs,
    max_k,
    split,
    run_ids,
    run_id_template,
    answer_llms,
    answers_path_template,
    strict,
    output_dir,
    plot_id,
):
    from plot_utils import plot_portfolio_router_ablations

    payloads = {}
    for answer_llm in answer_llms:
        print(
            f"[MAIN] Plot portfolio-router ablations: answer_llm={answer_llm}",
            flush=True,
        )
        model_plot_id = None
        if plot_id:
            model_plot_id = f"{plot_id}_{answer_llm}" if len(answer_llms) > 1 else plot_id
        payloads[answer_llm] = plot_portfolio_router_ablations(
            portfolio_id=portfolio_id,
            datasets=datasets,
            num_docs_to_fetch=num_docs,
            max_k=max_k,
            split=split,
            run_ids=run_ids,
            run_id_template=run_id_template,
            answer_llm=answer_llm,
            answers_path_template=answers_path_template,
            strict=strict,
            output_dir=output_dir,
            plot_id=model_plot_id,
        )
    return payloads


def run_plot_resources_vs_accuracy(models, retriever, num_docs, force):
    from plot_utils import plot_tokens_vs_quality

    payloads = {}
    for model_name in models:
        print(
            f"[MAIN] Plot resources vs accuracy: model={model_name}, retriever={retriever}",
            flush=True,
        )
        payloads[model_name] = plot_tokens_vs_quality(
            model_name=model_name,
            retriever=retriever,
            num_docs_to_fetch=num_docs,
            force=force,
        )
    return payloads


def run_full(
    datasets,
    retrievers,
    num_docs,
    portfolio_size,
    device,
    llm_key,
    embedder,
    prefilter_num=1000,
):
    run_prep(datasets, device, prefilter_num, embedder)
    run_train_retrievals(datasets, retrievers, num_docs, device, embedder, prefilter_num)
    run_universal_portfolio(retrievers, num_docs, portfolio_size, device, embedder)
    run_test_retrievals(datasets, retrievers, num_docs, portfolio_size, device, prefilter_num, embedder)
    run_build_answer_prompts(datasets, retrievers, num_docs, portfolio_size, embedder)
    run_answer_prompts(datasets, retrievers, num_docs, llm_key, embedder)
    run_build_selector_prompts(datasets, retrievers, num_docs, portfolio_size, llm_key, embedder)
    run_answer_selector_prompts(datasets, retrievers, num_docs, llm_key, embedder)


def _add_common_args(parser, include_llm=False, include_embedder=False):
    parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    parser.add_argument(
        "--retrievers",
        default=",".join([C.VENDI]),
        help="Comma-separated retriever pools (default: vendi).",
    )
    parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    parser.add_argument(
        "--portfolio-size",
        type=int,
        default=C.PORTFOLIO_SIZE,
        help="Universal portfolio size.",
    )
    parser.add_argument("--device", default="cpu", help="Device for embedding/retrieval.")
    parser.add_argument(
        "--prefilter-num",
        type=int,
        default=1000,
        help="Prefilter size for retrievals.",
    )
    if include_embedder:
        parser.add_argument(
            "--embedder",
            default=C.DEFAULT_EMBEDDER_KEY,
            choices=C.SUPPORTED_DENSE_EMBEDDER_KEYS,
            help="Dense embedding backbone for indexing and retrieval artifacts.",
        )
    if include_llm:
        parser.add_argument(
            "--llm",
            default=C.GEMMA27B,
            choices=[C.GEMMA27B, C.LLAMA70B],
            help="LLM key for answering and selector prompts.",
        )


def _add_baseline_args(parser):
    parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    parser.add_argument("--device", default="cpu", help="Device for embedding/retrieval.")
    parser.add_argument(
        "--prefilter-num",
        type=int,
        default=1000,
        help="Prefilter size for retrievals.",
    )


def _add_baseline_answer_args(parser):
    parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    parser.add_argument(
        "--llm",
        default=C.GEMMA27B,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key for answering baseline prompts.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="Number of worker threads for answering.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Checkpoint frequency for baseline answering.",
    )


def main():
    parser = argparse.ArgumentParser(description="Orchestrate the RAG pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("prep", help="Index + split + embeddings + prefilters.")
    _add_common_args(prep_parser, include_embedder=True)

    graph_index_parser = subparsers.add_parser(
        "build-graph-index",
        help="Build and save the dataset-level graph sidecar index.",
    )
    graph_index_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    graph_index_parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum concurrent LLM calls for graph-index entity extraction.",
    )
    graph_index_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Save graph-index entity-extraction checkpoints every N completed prompts.",
    )

    graph_query_entity_cache_parser = subparsers.add_parser(
        "build-graph-query-entity-cache",
        help="Build graph_dense query-entity caches from saved question entity extraction results.",
    )
    graph_query_entity_cache_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    graph_query_entity_cache_parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated question splits (default: train,test).",
    )
    graph_query_entity_cache_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing graph query entity cache files.",
    )

    train_parser = subparsers.add_parser("train-retrievals", help="Train retrievals + recall scores.")
    _add_common_args(train_parser, include_embedder=True)

    recalls_parser = subparsers.add_parser(
        "full-pool-recalls",
        help="Compute full-pool recall scores from retrievals_{split}.pickle.",
    )
    _add_common_args(recalls_parser, include_embedder=True)
    recalls_parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma-separated question splits (default: train,test).",
    )

    portfolio_parser = subparsers.add_parser("universal-portfolio", help="Compute universal portfolio.")
    _add_common_args(portfolio_parser, include_embedder=True)

    audit_pool_parser = subparsers.add_parser(
        "audit-pool-artifacts",
        help="Read-only audit for train/test artifacts across cataloged retriever pools.",
    )
    audit_pool_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    audit_pool_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    audit_pool_parser.add_argument(
        "--pool-set",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        choices=sorted(C.POOL_SETS),
        help="Named pool set to audit.",
    )
    audit_pool_parser.add_argument(
        "--pools",
        default=None,
        help="Comma-separated pool ids; overrides --pool-set.",
    )
    audit_pool_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected artifact is missing.",
    )

    portfolio_union_parser = subparsers.add_parser(
        "universal-portfolio-union",
        help="Compute a train-score universal portfolio over a cataloged pool union.",
    )
    portfolio_union_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    portfolio_union_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    portfolio_union_parser.add_argument(
        "--portfolio-size",
        type=int,
        default=C.PORTFOLIO_SIZE,
        help="Requested portfolio size.",
    )
    portfolio_union_parser.add_argument("--device", default="cpu", help="Torch device for score tensors.")
    portfolio_union_parser.add_argument(
        "--pool-set",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        choices=sorted(C.POOL_SETS),
        help="Named pool set to include.",
    )
    portfolio_union_parser.add_argument(
        "--pools",
        default=None,
        help="Comma-separated pool ids; overrides --pool-set.",
    )
    portfolio_union_parser.add_argument(
        "--portfolio-id",
        default=None,
        help="Stable id/name for the saved manifest; defaults to --pool-set.",
    )

    materialize_union_parser = subparsers.add_parser(
        "materialize-portfolio-test",
        help="Materialize a saved all-pool portfolio manifest on the test split.",
    )
    materialize_union_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    materialize_union_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    materialize_union_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="Portfolio id used by constants.get_universal_portfolio_union_manifest.",
    )
    materialize_union_parser.add_argument(
        "--portfolio-path",
        default=None,
        help="Explicit portfolio manifest pickle path; overrides --portfolio-id path lookup.",
    )
    materialize_union_parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Report per-dataset materialization failures instead of raising immediately.",
    )
    materialize_union_parser.set_defaults(strict=True)

    select_family_best_parser = subparsers.add_parser(
        "select-family-best-baselines",
        help="Select the train-best single retriever per family for paper baselines.",
    )
    select_family_best_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="Family-best baseline id; also used as a pool-set name when it matches one.",
    )
    select_family_best_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    select_family_best_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    select_family_best_parser.add_argument(
        "--families",
        default=f"{C.DS},{C.VENDI},{C.GRAPH_DENSE}",
        help="Comma-separated families to select (default: ds,vendi,graph_dense).",
    )
    select_family_best_parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for loading/averaging train score tensors.",
    )

    family_best_test_parser = subparsers.add_parser(
        "compute-family-best-baselines-test",
        help="Compute test retrievals plus recall/F1 scores for family-best baselines.",
    )
    family_best_test_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="Family-best baseline id used by constants.get_family_best_baseline_manifest.",
    )
    family_best_test_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    family_best_test_parser.add_argument("--num-docs", type=int, default=4, help="Docs per k unit.")
    family_best_test_parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Maximum k multiplier; retrieves max_k * num_docs documents.",
    )
    family_best_test_parser.add_argument(
        "--families",
        default=f"{C.DS},{C.VENDI},{C.GRAPH_DENSE}",
        help="Comma-separated families to compute (default: ds,vendi,graph_dense).",
    )
    family_best_test_parser.add_argument("--device", default="cuda", help="Device for retrieval.")
    family_best_test_parser.add_argument(
        "--prefilter-num",
        type=int,
        default=1000,
        help="Dense prefilter size for DS/Vendi retrieval.",
    )
    family_best_test_parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save a resumable retrieval checkpoint every N newly completed questions; 0 disables periodic checkpoints.",
    )

    family_best_prompts_parser = subparsers.add_parser(
        "build-family-best-answer-prompts",
        help="Build answer prompts for family-best baselines.",
    )
    family_best_prompts_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="Family-best baseline id used by constants.get_family_best_* paths.",
    )
    family_best_prompts_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    family_best_prompts_parser.add_argument(
        "--families",
        default=f"{C.DS},{C.VENDI},{C.GRAPH_DENSE}",
        help="Comma-separated families to build (default: ds,vendi,graph_dense).",
    )
    family_best_prompts_parser.add_argument("--num-docs", type=int, default=4, help="Docs per answer prompt.")
    family_best_prompts_parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Family-best retrieval artifact max_k.",
    )
    family_best_prompts_parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for smoke-testing prompt construction.",
    )

    family_best_answer_parser = subparsers.add_parser(
        "answer-family-best-prompts",
        help="Answer saved family-best baseline prompts with an LLM.",
    )
    family_best_answer_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="Family-best baseline id used by constants.get_family_best_* paths.",
    )
    family_best_answer_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    family_best_answer_parser.add_argument(
        "--families",
        default=f"{C.DS},{C.VENDI},{C.GRAPH_DENSE}",
        help="Comma-separated families to answer (default: ds,vendi,graph_dense).",
    )
    family_best_answer_parser.add_argument("--num-docs", type=int, default=4, help="Docs per answer prompt.")
    family_best_answer_parser.add_argument(
        "--max-k",
        type=int,
        default=5,
        help="Family-best retrieval artifact max_k.",
    )
    family_best_answer_parser.add_argument(
        "--llm",
        default=C.GEMMA27B,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key for answering family-best prompts.",
    )
    family_best_answer_parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of worker threads for answering.",
    )
    family_best_answer_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Checkpoint frequency for family-best answering.",
    )

    test_retrievals_parser = subparsers.add_parser(
        "test-retrievals", help="Universal-portfolio retrievals on test."
    )
    _add_common_args(test_retrievals_parser, include_embedder=True)

    test_retrievals_pool_parser = subparsers.add_parser(
        "test-retrievals-pool", help="Full retriever-pool retrievals on test."
    )
    _add_common_args(test_retrievals_pool_parser, include_embedder=True)
    test_retrievals_pool_parser.add_argument(
        "--compute-recalls",
        action="store_true",
        help="Compute full-pool scores_test.pickle after test retrievals.",
    )

    single_retriever_test_parser = subparsers.add_parser(
        "single-retriever-test",
        help="Single-retriever retrievals on test (portfolio first retriever).",
    )
    _add_common_args(single_retriever_test_parser, include_embedder=True)
    single_retriever_test_parser.add_argument(
        "--universal",
        action="store_true",
        help="Use the universal portfolio instead of dataset-specific portfolio.",
    )

    build_answer_parser = subparsers.add_parser(
        "build-answer-prompts", help="Build answer prompts from test retrievals."
    )
    _add_common_args(build_answer_parser, include_embedder=True)

    build_union_answer_parser = subparsers.add_parser(
        "build-portfolio-union-answer-prompts",
        help="Build answer prompts from materialized all-pool portfolio test retrievals.",
    )
    build_union_answer_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    build_union_answer_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used by constants.get_portfolio_union_retrievals_test.",
    )
    build_union_answer_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    build_union_answer_parser.add_argument(
        "--portfolio-size",
        type=int,
        default=None,
        help="Use the first K materialized portfolio members; default uses all materialized members.",
    )
    build_union_answer_parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for smoke-testing prompt construction.",
    )

    baseline_parser = subparsers.add_parser(
        "build-baseline-prompts",
        help="Build baseline prompts (no retrieval + DS(0,1)).",
    )
    _add_baseline_args(baseline_parser)

    baseline_answer_parser = subparsers.add_parser(
        "answer-baseline-prompts",
        help="Answer baseline prompts with an LLM.",
    )
    _add_baseline_answer_args(baseline_answer_parser)

    answer_prompts_parser = subparsers.add_parser(
        "answer-prompts", help="Answer saved prompts with an LLM."
    )
    _add_common_args(answer_prompts_parser, include_llm=True, include_embedder=True)

    answer_union_parser = subparsers.add_parser(
        "answer-portfolio-union-prompts",
        help="Answer materialized all-pool portfolio prompts with an LLM.",
    )
    answer_union_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    answer_union_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used by constants.get_portfolio_union_answer_prompts_test.",
    )
    answer_union_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    answer_union_parser.add_argument(
        "--portfolio-size",
        type=int,
        default=None,
        help="Optional prompt portfolio-size validation.",
    )
    answer_union_parser.add_argument(
        "--llm",
        default=C.GEMMA27B,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key for answering all-pool portfolio prompts.",
    )
    answer_union_parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of worker threads for answering.",
    )
    answer_union_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Checkpoint frequency for all-pool answering.",
    )

    router_train_parser = subparsers.add_parser(
        "portfolio-router-train",
        help="Train the final all-pool portfolio router.",
    )
    router_train_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    router_train_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used for router data.",
    )
    router_train_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    router_train_parser.add_argument(
        "--portfolio-size",
        "-k",
        type=int,
        default=C.PORTFOLIO_SIZE,
        help="Router portfolio size.",
    )
    router_train_parser.add_argument("--output-dir", default=None)
    router_train_parser.add_argument("--device", default="cuda")
    router_train_parser.add_argument("--batch-size", type=int, default=64)
    router_train_parser.add_argument("--max-length", type=int, default=256)
    router_train_parser.add_argument("--epochs", type=int, default=10)
    router_train_parser.add_argument("--lr", type=float, default=3e-4)
    router_train_parser.add_argument("--weight-decay", type=float, default=0.0)
    router_train_parser.add_argument("--dev-ratio", type=float, default=0.1)
    router_train_parser.add_argument("--seed", type=int, default=0)
    router_train_parser.add_argument("--grad-clip", type=float, default=1.0)
    router_train_parser.add_argument("--grad-accum-steps", type=int, default=1)
    router_train_parser.add_argument("--max-train-questions", type=int, default=None)
    router_train_parser.add_argument("--max-dev-questions", type=int, default=None)
    router_train_parser.add_argument("--max-test-questions", type=int, default=None)
    router_train_parser.add_argument("--no-test", action="store_true", help="Skip test-set evaluation during training.")
    router_train_parser.add_argument("--t5-model-name-or-path", default=None)
    router_train_parser.add_argument("--tokenizer-name-or-path", default=None)
    router_train_parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    router_train_parser.add_argument("--wandb-run-name", default=None)
    router_train_parser.add_argument("--resume-from", default=None)
    router_train_parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not auto-resume from portfolio_router_last.pt in the output dir.",
    )

    router_predict_parser = subparsers.add_parser(
        "portfolio-router-predict-test",
        help="Write portfolio-router test predictions from a trained checkpoint.",
    )
    router_predict_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    router_predict_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used for router data.",
    )
    router_predict_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    router_predict_parser.add_argument(
        "--portfolio-size",
        "-k",
        type=int,
        default=C.PORTFOLIO_SIZE,
        help="Router portfolio size.",
    )
    router_predict_parser.add_argument("--run-id", required=True, help="Prediction run id to write.")
    router_predict_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to MODELS_DIR/portfolio_router/<portfolio-id>/k<K>/portfolio_router_best.pt.",
    )
    router_predict_parser.add_argument("--device", default="cuda")
    router_predict_parser.add_argument("--batch-size", type=int, default=64)
    router_predict_parser.add_argument("--max-questions", type=int, default=None)
    router_predict_parser.add_argument("--t5-model-name-or-path", default=None)
    router_predict_parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow missing optional router-data artifacts where supported.",
    )
    router_predict_parser.set_defaults(strict=True)

    build_router_judge_parser = subparsers.add_parser(
        "build-portfolio-router-judge-prompts",
        help="Build judge prompts from portfolio-router top-ell all-pool portfolio answers.",
    )
    build_router_judge_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    build_router_judge_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used by constants.get_portfolio_router_predictions.",
    )
    build_router_judge_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    build_router_judge_parser.add_argument(
        "--portfolio-size",
        "-k",
        type=int,
        default=C.PORTFOLIO_SIZE,
        help="Router portfolio size.",
    )
    build_router_judge_parser.add_argument(
        "--ell",
        type=int,
        required=True,
        help="Number of router-ranked candidate answers to show to the judge.",
    )
    build_router_judge_parser.add_argument("--run-id", required=True, help="Router prediction run id.")
    build_router_judge_parser.add_argument(
        "--answer-llm",
        required=True,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key used for the saved all-pool per-member answers.",
    )
    build_router_judge_parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for smoke-testing judge prompt construction.",
    )
    build_router_judge_parser.add_argument(
        "--no-strict-answers",
        dest="strict_answers",
        action="store_false",
        help="Skip questions with missing selected candidate answers instead of failing.",
    )
    build_router_judge_parser.set_defaults(strict_answers=True)

    build_all_router_judge_parser = subparsers.add_parser(
        "build-all-portfolio-router-judge-prompts",
        help=(
            "Build judge prompts for multiple router k/ell values and answer LLMs, "
            "auto-discovering prediction run ids per k."
        ),
    )
    build_all_router_judge_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    build_all_router_judge_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used by constants.get_portfolio_router_predictions.",
    )
    build_all_router_judge_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    build_all_router_judge_parser.add_argument(
        "--portfolio-sizes",
        default="2,3,4,5",
        help="Comma-separated router portfolio sizes to process.",
    )
    build_all_router_judge_parser.add_argument(
        "--ells",
        default=None,
        help=(
            "Optional comma-separated ell values to use for every k. "
            "Default uses all valid selector ell values, 2..k."
        ),
    )
    build_all_router_judge_parser.add_argument(
        "--min-ell",
        type=int,
        default=2,
        help="Minimum ell when --ells is not provided. Default skips ell=1 because no judge prompt is needed.",
    )
    build_all_router_judge_parser.add_argument(
        "--max-ell",
        type=int,
        default=None,
        help="Maximum ell when --ells is not provided. Default is k.",
    )
    build_all_router_judge_parser.add_argument(
        "--include-ell1",
        action="store_true",
        help="Also write direct-top1 metadata payloads for ell=1. No judge prompts are built for ell=1.",
    )
    build_all_router_judge_parser.add_argument(
        "--run-id-map",
        default=None,
        help=(
            "Optional comma-separated k:run_id overrides, e.g. "
            "2:k2_lr0p0001_wd0_seed0_best,3:k3_lr0p0003_wd0_seed0_best. "
            "Missing k values are auto-discovered from prediction artifacts."
        ),
    )
    build_all_router_judge_parser.add_argument(
        "--answer-llms",
        default=f"{C.LLAMA70B},{C.GEMMA27B}",
        help="Comma-separated answer LLM keys whose per-member answers should be used.",
    )
    build_all_router_judge_parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for smoke-testing prompt construction.",
    )
    build_all_router_judge_parser.add_argument(
        "--no-strict-answers",
        dest="strict_answers",
        action="store_false",
        help="Skip questions with missing selected candidate answers instead of failing.",
    )
    build_all_router_judge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved k/ell/run_id/answer_llm combinations without writing prompts.",
    )
    build_all_router_judge_parser.set_defaults(strict_answers=True)

    answer_router_judge_parser = subparsers.add_parser(
        "answer-portfolio-router-judge-prompts",
        help="Answer router top-ell judge prompts with an LLM.",
    )
    answer_router_judge_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    answer_router_judge_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used by constants.get_portfolio_router_judge_prompts.",
    )
    answer_router_judge_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    answer_router_judge_parser.add_argument(
        "--portfolio-size",
        "-k",
        type=int,
        default=C.PORTFOLIO_SIZE,
        help="Router portfolio size.",
    )
    answer_router_judge_parser.add_argument(
        "--ell",
        type=int,
        required=True,
        help="Number of router-ranked candidate answers in each judge prompt.",
    )
    answer_router_judge_parser.add_argument("--run-id", required=True, help="Router prediction run id.")
    answer_router_judge_parser.add_argument(
        "--answer-llm",
        required=True,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key used for the saved all-pool per-member answers.",
    )
    answer_router_judge_parser.add_argument(
        "--judge-llm",
        required=True,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key used to answer the judge prompts.",
    )
    answer_router_judge_parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of worker threads for judge answering.",
    )
    answer_router_judge_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Checkpoint frequency for judge answering.",
    )

    answer_all_router_judge_parser = subparsers.add_parser(
        "answer-all-portfolio-router-judge-prompts",
        help=(
            "Answer judge prompts for multiple router k/ell values and answer LLMs, "
            "auto-discovering prediction run ids per k."
        ),
    )
    answer_all_router_judge_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    answer_all_router_judge_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id used by constants.get_portfolio_router_judge_prompts.",
    )
    answer_all_router_judge_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    answer_all_router_judge_parser.add_argument(
        "--portfolio-sizes",
        default="2,3,4,5",
        help="Comma-separated router portfolio sizes to process.",
    )
    answer_all_router_judge_parser.add_argument(
        "--ells",
        default=None,
        help=(
            "Optional comma-separated ell values to use for every k. "
            "Default answers all valid selector ell values, 2..k."
        ),
    )
    answer_all_router_judge_parser.add_argument(
        "--min-ell",
        type=int,
        default=2,
        help="Minimum ell when --ells is not provided. Default skips ell=1 because no judge prompt exists.",
    )
    answer_all_router_judge_parser.add_argument(
        "--max-ell",
        type=int,
        default=None,
        help="Maximum ell when --ells is not provided. Default is k.",
    )
    answer_all_router_judge_parser.add_argument(
        "--run-id-map",
        default=None,
        help=(
            "Optional comma-separated k:run_id overrides, e.g. "
            "2:k2_lr0p0001_wd0_seed0_best,3:k3_lr0p0003_wd0_seed0_best. "
            "Missing k values are auto-discovered from prediction artifacts."
        ),
    )
    answer_all_router_judge_parser.add_argument(
        "--answer-llms",
        default=f"{C.LLAMA70B},{C.GEMMA27B}",
        help="Comma-separated answer LLM keys whose per-member answers were used to build judge prompts.",
    )
    answer_all_router_judge_parser.add_argument(
        "--judge-llm",
        required=True,
        choices=[C.GEMMA27B, C.LLAMA70B],
        help="LLM key to answer the judge prompts.",
    )
    answer_all_router_judge_parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of worker threads for judge answering.",
    )
    answer_all_router_judge_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1000,
        help="Checkpoint frequency for judge answering.",
    )
    answer_all_router_judge_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved k/ell/run_id/answer_llm combinations without answering prompts.",
    )

    build_selector_parser = subparsers.add_parser(
        "build-selector-prompts", help="Build selector prompts for a given LLM."
    )
    _add_common_args(build_selector_parser, include_llm=True, include_embedder=True)

    answer_selector_parser = subparsers.add_parser(
        "answer-selector-prompts", help="Answer selector prompts with an LLM."
    )
    _add_common_args(answer_selector_parser, include_llm=True, include_embedder=True)

    vendirag_parser = subparsers.add_parser(
        "vendi-rag",
        help="Run VendiRAGAdaptive on test questions.",
    )
    _add_common_args(vendirag_parser, include_llm=True, include_embedder=True)
    vendirag_parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of adaptive steps per question.",
    )
    vendirag_parser.add_argument(
        "--initial-s",
        type=float,
        default=0.8,
        help="Initial s value for the adaptive loop.",
    )
    vendirag_parser.add_argument(
        "--max-questions",
        type=int,
        default=1000,
        help="Maximum number of test questions to process.",
    )
    vendirag_parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index into the test questions.",
    )
    vendirag_parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Number of worker threads for answering/judging.",
    )
    vendirag_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Checkpoint frequency for VendiRAG runs.",
    )

    plot_support_parser = subparsers.add_parser(
        "plot-all-pool-support",
        help="Generate the paper all-pool portfolio support recall/F1 plots.",
    )
    plot_support_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    plot_support_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id.",
    )
    plot_support_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    plot_support_parser.add_argument("--max-k", type=int, default=5, help="Maximum portfolio prefix size.")
    plot_support_parser.add_argument(
        "--no-compute-missing-f1",
        dest="compute_missing_f1",
        action="store_false",
        help="Require cached F1 scores instead of computing missing ones from retrievals.",
    )
    plot_support_parser.add_argument(
        "--include-vendi-family-best",
        action="store_true",
        help="Also plot the Vendi family-best baseline.",
    )
    plot_support_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first missing dataset artifact.",
    )
    plot_support_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plot outputs. Default: PLOTS_DIR/average.",
    )
    plot_support_parser.set_defaults(compute_missing_f1=True)

    plot_router_parser = subparsers.add_parser(
        "plot-portfolio-router-ablations",
        help="Generate the paper portfolio-router ablation plots.",
    )
    plot_router_parser.add_argument(
        "--datasets",
        default=",".join(C.DATASETS),
        help="Comma-separated dataset names (default: all).",
    )
    plot_router_parser.add_argument(
        "--portfolio-id",
        default=C.POOL_SET_ALL_IMPLEMENTED,
        help="All-pool portfolio id.",
    )
    plot_router_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    plot_router_parser.add_argument("--max-k", type=int, default=5, help="Maximum portfolio prefix size.")
    plot_router_parser.add_argument("--split", default="test", choices=["test"], help="Question split to plot.")
    plot_router_parser.add_argument(
        "--run-ids",
        default=None,
        help="Comma-separated k:run_id overrides, e.g. 2:run_k2,3:run_k3.",
    )
    plot_router_parser.add_argument(
        "--run-id-template",
        default=None,
        help="Format string for run ids, e.g. k{k}_best.",
    )
    plot_router_parser.add_argument(
        "--answer-llms",
        default=f"{C.GEMMA27B},{C.LLAMA70B}",
        help="Comma-separated answer LLMs for EM plots.",
    )
    plot_router_parser.add_argument(
        "--answers-path-template",
        default=None,
        help="Optional format string overriding portfolio answer artifact paths.",
    )
    plot_router_parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on the first missing dataset artifact.",
    )
    plot_router_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plot outputs. Default: PLOTS_DIR/average.",
    )
    plot_router_parser.add_argument(
        "--plot-id",
        default=None,
        help="Optional filename suffix for generated plots.",
    )

    plot_resources_parser = subparsers.add_parser(
        "plot-resources-vs-accuracy",
        help="Generate the paper Vendi-Portfolio/Vendi-RAG resource-vs-accuracy plots.",
    )
    plot_resources_parser.add_argument(
        "--models",
        default=f"{C.GEMMA27B},{C.LLAMA70B}",
        help="Comma-separated answer models to plot.",
    )
    plot_resources_parser.add_argument(
        "--retriever",
        default=C.VENDI,
        choices=[C.VENDI],
        help="Retriever family for the portfolio/resource plot.",
    )
    plot_resources_parser.add_argument("--num-docs", type=int, default=4, help="Docs per retriever.")
    plot_resources_parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached token/EM summaries before plotting.",
    )

    full_parser = subparsers.add_parser("full", help="Run the full pipeline.")
    _add_common_args(full_parser, include_llm=True, include_embedder=True)

    args = parser.parse_args()

    datasets = _parse_list(args.datasets, C.DATASETS, "dataset") if hasattr(args, "datasets") else list(C.DATASETS)
    splits = _parse_splits(args.splits) if hasattr(args, "splits") else None
    retrievers = (
        _parse_list(args.retrievers, [C.VENDI, C.DS, C.GRAPH_DENSE], "retriever")
        if hasattr(args, "retrievers")
        else None
    )

    if args.command == "prep":
        run_prep(datasets, args.device, args.prefilter_num, args.embedder)
    elif args.command == "build-graph-index":
        run_build_graph_index(datasets, args.max_workers, args.checkpoint_every)
    elif args.command == "build-graph-query-entity-cache":
        run_build_graph_query_entity_cache(datasets, splits, args.overwrite)
    elif args.command == "train-retrievals":
        run_train_retrievals(
            datasets,
            retrievers,
            args.num_docs,
            args.device,
            args.embedder,
            args.prefilter_num,
        )
    elif args.command == "full-pool-recalls":
        run_full_pool_recalls(datasets, retrievers, args.num_docs, splits, args.embedder)
    elif args.command == "universal-portfolio":
        run_universal_portfolio(retrievers, args.num_docs, args.portfolio_size, args.device, args.embedder)
    elif args.command == "audit-pool-artifacts":
        run_pool_artifact_audit(
            datasets,
            args.num_docs,
            args.pool_set,
            args.pools,
            args.strict,
        )
    elif args.command == "universal-portfolio-union":
        run_universal_portfolio_union(
            datasets,
            args.num_docs,
            args.portfolio_size,
            args.device,
            args.pool_set,
            args.pools,
            args.portfolio_id,
        )
    elif args.command == "materialize-portfolio-test":
        run_materialize_portfolio_test(
            datasets,
            args.num_docs,
            args.portfolio_id,
            args.portfolio_path,
            args.strict,
        )
    elif args.command == "select-family-best-baselines":
        run_select_family_best_baselines(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            families=_parse_families(args.families),
            device=args.device,
        )
    elif args.command == "compute-family-best-baselines-test":
        run_compute_family_best_baselines_test(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            max_k=args.max_k,
            families=_parse_families(args.families),
            device=args.device,
            prefilter_num=args.prefilter_num,
            save_every=args.save_every,
        )
    elif args.command == "build-family-best-answer-prompts":
        run_build_family_best_answer_prompts(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            families=_parse_families(args.families),
            num_docs=args.num_docs,
            max_k=args.max_k,
            max_questions=args.max_questions,
        )
    elif args.command == "answer-family-best-prompts":
        run_answer_family_best_prompts(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            families=_parse_families(args.families),
            num_docs=args.num_docs,
            max_k=args.max_k,
            llm_key=args.llm,
            max_workers=args.max_workers,
            checkpoint_every=args.checkpoint_every,
        )
    elif args.command == "test-retrievals":
        run_test_retrievals(
            datasets,
            retrievers,
            args.num_docs,
            args.portfolio_size,
            args.device,
            args.prefilter_num,
            args.embedder,
        )
    elif args.command == "test-retrievals-pool":
        run_test_retrievals_pool(
            datasets,
            retrievers,
            args.num_docs,
            args.device,
            args.prefilter_num,
            args.embedder,
            args.compute_recalls,
        )
    elif args.command == "single-retriever-test":
        run_single_retriever_test(
            datasets,
            retrievers,
            args.num_docs,
            args.portfolio_size,
            args.device,
            args.prefilter_num,
            args.universal,
            args.embedder,
        )
    elif args.command == "build-answer-prompts":
        run_build_answer_prompts(datasets, retrievers, args.num_docs, args.portfolio_size, args.embedder)
    elif args.command == "build-portfolio-union-answer-prompts":
        run_build_portfolio_union_answer_prompts(
            datasets,
            args.portfolio_id,
            args.num_docs,
            args.portfolio_size,
            args.max_questions,
        )
    elif args.command == "build-baseline-prompts":
        run_build_baseline_prompts(
            datasets,
            args.num_docs,
            args.device,
            args.prefilter_num,
        )
    elif args.command == "answer-baseline-prompts":
        run_answer_baseline_prompts(
            datasets,
            args.num_docs,
            args.llm,
            args.max_workers,
            args.checkpoint_every,
        )
    elif args.command == "answer-prompts":
        run_answer_prompts(datasets, retrievers, args.num_docs, args.llm, args.embedder)
    elif args.command == "answer-portfolio-union-prompts":
        run_answer_portfolio_union_prompts(
            datasets,
            args.portfolio_id,
            args.num_docs,
            args.portfolio_size,
            args.llm,
            args.max_workers,
            args.checkpoint_every,
        )
    elif args.command == "portfolio-router-train":
        run_train_portfolio_router(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            portfolio_size=args.portfolio_size,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            max_train_questions=args.max_train_questions,
            max_dev_questions=args.max_dev_questions,
            max_test_questions=args.max_test_questions,
            load_test=not args.no_test,
            t5_model_name_or_path=args.t5_model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_wandb=not args.no_wandb,
            wandb_run_name=args.wandb_run_name,
            resume_from=args.resume_from,
            auto_resume=not args.no_auto_resume,
        )
    elif args.command == "portfolio-router-predict-test":
        run_portfolio_router_predict_test(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            portfolio_size=args.portfolio_size,
            run_id=args.run_id,
            checkpoint_path=args.checkpoint,
            device=args.device,
            batch_size=args.batch_size,
            max_questions=args.max_questions,
            strict=args.strict,
            t5_model_name_or_path=args.t5_model_name_or_path,
        )
    elif args.command == "build-portfolio-router-judge-prompts":
        run_build_portfolio_router_judge_prompts(
            datasets,
            args.portfolio_id,
            args.num_docs,
            args.portfolio_size,
            args.ell,
            args.run_id,
            args.answer_llm,
            args.max_questions,
            args.strict_answers,
        )
    elif args.command == "build-all-portfolio-router-judge-prompts":
        run_build_all_portfolio_router_judge_prompts(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            portfolio_sizes=_parse_int_list(args.portfolio_sizes, [2, 3, 4, 5], "portfolio size"),
            ell_values=args.ells,
            min_ell=args.min_ell,
            max_ell=args.max_ell,
            include_ell1=args.include_ell1,
            run_id_map=args.run_id_map,
            answer_llms=_parse_llm_list(args.answer_llms, default=[C.LLAMA70B, C.GEMMA27B]),
            max_questions=args.max_questions,
            strict_answers=args.strict_answers,
            dry_run=args.dry_run,
        )
    elif args.command == "answer-portfolio-router-judge-prompts":
        run_answer_portfolio_router_judge_prompts(
            datasets,
            args.portfolio_id,
            args.num_docs,
            args.portfolio_size,
            args.ell,
            args.run_id,
            args.answer_llm,
            args.judge_llm,
            args.max_workers,
            args.checkpoint_every,
        )
    elif args.command == "answer-all-portfolio-router-judge-prompts":
        run_answer_all_portfolio_router_judge_prompts(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            portfolio_sizes=_parse_int_list(args.portfolio_sizes, [2, 3, 4, 5], "portfolio size"),
            ell_values=args.ells,
            min_ell=args.min_ell,
            max_ell=args.max_ell,
            run_id_map=args.run_id_map,
            answer_llms=_parse_llm_list(args.answer_llms, default=[C.LLAMA70B, C.GEMMA27B]),
            judge_llm_key=args.judge_llm,
            max_workers=args.max_workers,
            checkpoint_every=args.checkpoint_every,
            dry_run=args.dry_run,
        )
    elif args.command == "build-selector-prompts":
        run_build_selector_prompts(
            datasets, retrievers, args.num_docs, args.portfolio_size, args.llm, args.embedder
        )
    elif args.command == "answer-selector-prompts":
        run_answer_selector_prompts(datasets, retrievers, args.num_docs, args.llm, args.embedder)
    elif args.command == "vendi-rag":
        run_vendi_rag(
            datasets,
            args.num_docs,
            args.llm,
            args.device,
            args.num_steps,
            args.initial_s,
            args.max_questions,
            args.start_idx,
            args.max_workers,
            args.checkpoint_every,
            args.embedder,
        )
    elif args.command == "plot-all-pool-support":
        run_plot_all_pool_support(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            max_k=args.max_k,
            compute_missing_f1=args.compute_missing_f1,
            include_vendi_family_best=args.include_vendi_family_best,
            strict=args.strict,
            output_dir=args.output_dir,
        )
    elif args.command == "plot-portfolio-router-ablations":
        run_plot_portfolio_router_ablations(
            datasets=datasets,
            portfolio_id=args.portfolio_id,
            num_docs=args.num_docs,
            max_k=args.max_k,
            split=args.split,
            run_ids=args.run_ids,
            run_id_template=args.run_id_template,
            answer_llms=_parse_llm_list(
                args.answer_llms,
                default=[C.GEMMA27B, C.LLAMA70B],
            ),
            answers_path_template=args.answers_path_template,
            strict=args.strict,
            output_dir=args.output_dir,
            plot_id=args.plot_id,
        )
    elif args.command == "plot-resources-vs-accuracy":
        run_plot_resources_vs_accuracy(
            models=_parse_llm_list(args.models, default=[C.GEMMA27B, C.LLAMA70B]),
            retriever=args.retriever,
            num_docs=args.num_docs,
            force=args.force,
        )
    elif args.command == "full":
        run_full(
            datasets,
            retrievers,
            args.num_docs,
            args.portfolio_size,
            args.device,
            args.llm,
            args.embedder,
            args.prefilter_num,
        )
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

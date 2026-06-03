import os

# Directories
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def _env_path(name, default):
    value = os.environ.get(name, default)
    return value if value.endswith(os.sep) else value + os.sep

CACHE_DIR = _env_path('CACHE_DIR', os.path.join(_PROJECT_ROOT, 'cache'))
MODELS_DIR = _env_path('MODELS_DIR', os.path.join(_PROJECT_ROOT, 'models'))
RESULTS_DIR = _env_path('RESULTS_DIR', os.path.join(_PROJECT_ROOT, 'results'))
PLOTS_DIR = os.environ.get('PLOTS_DIR', 'plots/')

# Datasets
HotpotQA = 'HotpotQA'
MUSIQUE = 'MusiQue'
TRIVIAQA = 'TriviaQA'
TWOWIKI = '2WikiMultiHopQA'
DATASETS = [HotpotQA, MUSIQUE, TRIVIAQA, TWOWIKI]

DATASET_LOCATION = {}
DATASET_LOCATION[HotpotQA] = os.environ.get('HOTPOTQA_DIR', 'datasets/HotpotQA-BEIR/')
DATASET_LOCATION[MUSIQUE] = os.environ.get('MUSIQUE_DIR', 'datasets/MuSiQue/')
DATASET_LOCATION[TRIVIAQA] = os.environ.get('TRIVIAQA_DIR', 'datasets/TriviaQA/')
DATASET_LOCATION[TWOWIKI] = os.environ.get('TWOWIKI_DIR', 'datasets/2WikiMultiHopQA/')

# Caches
get_vector_db_dir = lambda dataset, embedder=None: CACHE_DIR + f'{dataset}/faiss_vector_db{_embedder_tag(embedder)}/'  # FaissVectorDB
get_chunk_cache_dir = lambda dataset: CACHE_DIR + f'{dataset}/chunk_cache/'  # Chunked corpus cache
get_chunk_cache_metadata_path = lambda dataset: get_chunk_cache_dir(dataset) + 'metadata.json'
get_chunk_cache_documents_path = lambda dataset: get_chunk_cache_dir(dataset) + 'documents.pkl'
get_graph_index_path = lambda dataset: CACHE_DIR + f'{dataset}/graph_index.pickle'  # Dataset-level graph sidecar index
get_graph_entity_extraction_prompts_path = lambda dataset: CACHE_DIR + f'{dataset}/graph_entity_extraction_prompts.pickle'
get_graph_entity_extraction_results_path = lambda dataset: CACHE_DIR + f'{dataset}/graph_entity_extraction_results.pickle'
get_graph_question_entity_extraction_prompts_path = lambda dataset, split: CACHE_DIR + f'{dataset}/graph_question_entity_extraction_prompts_{split}.pickle'
get_graph_question_entity_extraction_results_path = lambda dataset, split: CACHE_DIR + f'{dataset}/graph_question_entity_extraction_results_{split}.pickle'
get_graph_query_entities_path = lambda dataset, split: CACHE_DIR + f'{dataset}/graph_query_entities_{split}.pickle'
get_questions_train = lambda dataset: CACHE_DIR + f'{dataset}/questions_train.pickle'   # Questions train split saved on disk
get_questions_test  = lambda dataset: CACHE_DIR + f'{dataset}/questions_test.pickle'    # Questions test split saved on disk
get_prefilters_train = lambda dataset, embedder=None: CACHE_DIR + f'{dataset}/prefilters_train{_embedder_tag(embedder)}.pickle' # Prefilter pools (train)
get_prefilters_test  = lambda dataset, embedder=None: CACHE_DIR + f'{dataset}/prefilters_test{_embedder_tag(embedder)}.pickle'  # Prefilter pools (test)
get_embeddings_train = lambda dataset, embedder=None: CACHE_DIR + f'{dataset}/embeddings_train{_embedder_tag(embedder)}.pickle' # Question embeddings (train)
get_embeddings_test  = lambda dataset, embedder=None: CACHE_DIR + f'{dataset}/embeddings_test{_embedder_tag(embedder)}.pickle'  # Question embeddings (test)

# Models
GEMMA27B = 'Gemma27B'
LLAMA70B = 'Llama70B'

LLM_DIR = {}
LLM_DIR[GEMMA27B] = os.environ.get('GEMMA27B_MODEL_DIR', MODELS_DIR + 'gemma-3-27b-it')
LLM_DIR[LLAMA70B] = os.environ.get('LLAMA70B_MODEL_DIR', MODELS_DIR + 'llama-3.1-70b-instruct')

# Router models
T5_LARGE = 'flan-t5-large'
ROUTER_T5_DIR = os.environ.get('ROUTER_T5_DIR', MODELS_DIR + T5_LARGE)

LLM_BASE_URL = {}
LLM_BASE_URL[GEMMA27B] = 'http://localhost:5000/v1'
LLM_BASE_URL[LLAMA70B] = 'http://localhost:8000/v1'

# Retrievers
VENDI = 'vendi'
DS = 'ds'
GRAPH_DENSE = 'graph_dense'
GRAPH_DENSE_MIXED_EMBEDDER_KEY = "mixed"

VENDI_POOL_PARAMETERS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

GRAPH_DENSE_POOL_PARAMETERS = [
    {
        "name": f"{embedder}_h{max_hops}_df{max_entity_df}_c{max_candidates}",
        "embedder": embedder,
        "max_hops": max_hops,
        "max_entity_df": max_entity_df,
        "max_candidates": max_candidates,
    }
    for embedder in ["mpnet", "e5"]
    for max_hops in [1, 3, 5]
    for max_entity_df in [100, 300, 500]
    for max_candidates in [1000, 2000]
]

gammas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 4, 6, 8, 10]
rs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DS_POOL_PARAMETERS = []

for gamma in gammas:
    for r in rs:
        DS_POOL_PARAMETERS.append((gamma,r))

DS_POOL_PARAMETERS.append((0,1)) # Naive retriever, no discount

# Embedder backbones
DEFAULT_EMBEDDER_KEY = "mpnet"
E5_EMBEDDER_KEY = "e5"
EMBEDDER_REGISTRY = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    E5_EMBEDDER_KEY: "intfloat/e5-large-v2",
}
SUPPORTED_DENSE_EMBEDDER_KEYS = tuple(EMBEDDER_REGISTRY.keys())
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

def normalize_embedder_key(embedder):
    if embedder is None:
        return DEFAULT_EMBEDDER_KEY
    if embedder == GRAPH_DENSE_MIXED_EMBEDDER_KEY:
        return GRAPH_DENSE_MIXED_EMBEDDER_KEY
    if embedder in EMBEDDER_REGISTRY:
        return embedder
    # Allow passing full model name
    for key, name in EMBEDDER_REGISTRY.items():
        if embedder == name:
            return key
    raise ValueError(
        f"Unsupported dense embedder: {embedder}. "
        f"Supported: {', '.join(SUPPORTED_DENSE_EMBEDDER_KEYS)}"
    )

def resolve_embedder_name(embedder=None):
    key = normalize_embedder_key(embedder)
    return EMBEDDER_REGISTRY.get(key, key)

def _embedder_tag(embedder=None):
    key = normalize_embedder_key(embedder)
    return "" if key == DEFAULT_EMBEDDER_KEY else f"_{key}"

def _retriever_dir(retriever, num_docs, embedder=None):
    return f"{retriever}_{num_docs}{_embedder_tag(embedder)}"

def _safe_artifact_name(value):
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("@", "_at_")
        .replace("=", "")
        .replace(",", "_")
    )

def _pool_size_for_retriever(retriever):
    if retriever == DS:
        return len(DS_POOL_PARAMETERS)
    if retriever == VENDI:
        return len(VENDI_POOL_PARAMETERS)
    if retriever == GRAPH_DENSE:
        return len(GRAPH_DENSE_POOL_PARAMETERS)
    raise ValueError(f"Unsupported retriever pool: {retriever}")

def _pool_spec(retriever, artifact_embedder_key, display_label):
    artifact_embedder_key = normalize_embedder_key(artifact_embedder_key)
    pool_id = f"{retriever}@{artifact_embedder_key}"
    return {
        "pool_id": pool_id,
        "label": pool_id,
        "display_label": display_label,
        "retriever": retriever,
        "family": retriever,
        "artifact_embedder_key": artifact_embedder_key,
        "embedder": artifact_embedder_key,
        "pool_size": _pool_size_for_retriever(retriever),
    }

POOL_DS_MPNET = f"{DS}@{DEFAULT_EMBEDDER_KEY}"
POOL_DS_E5 = f"{DS}@{E5_EMBEDDER_KEY}"
POOL_VENDI_MPNET = f"{VENDI}@{DEFAULT_EMBEDDER_KEY}"
POOL_VENDI_E5 = f"{VENDI}@{E5_EMBEDDER_KEY}"
POOL_GRAPH_DENSE_MIXED = f"{GRAPH_DENSE}@{GRAPH_DENSE_MIXED_EMBEDDER_KEY}"

POOL_CATALOG = {
    POOL_DS_MPNET: _pool_spec(DS, DEFAULT_EMBEDDER_KEY, "DS / MPNet"),
    POOL_DS_E5: _pool_spec(DS, E5_EMBEDDER_KEY, "DS / E5"),
    POOL_VENDI_MPNET: _pool_spec(VENDI, DEFAULT_EMBEDDER_KEY, "Vendi / MPNet"),
    POOL_VENDI_E5: _pool_spec(VENDI, E5_EMBEDDER_KEY, "Vendi / E5"),
    POOL_GRAPH_DENSE_MIXED: _pool_spec(
        GRAPH_DENSE,
        GRAPH_DENSE_MIXED_EMBEDDER_KEY,
        "Graph dense / mixed",
    ),
}

POOL_SET_ALL_IMPLEMENTED = "all_implemented"
POOL_SETS = {
    POOL_SET_ALL_IMPLEMENTED: [
        POOL_DS_MPNET,
        POOL_DS_E5,
        POOL_VENDI_MPNET,
        POOL_VENDI_E5,
        POOL_GRAPH_DENSE_MIXED,
    ],
    "all": [
        POOL_DS_MPNET,
        POOL_DS_E5,
        POOL_VENDI_MPNET,
        POOL_VENDI_E5,
        POOL_GRAPH_DENSE_MIXED,
    ],
    "dense_all": [
        POOL_DS_MPNET,
        POOL_DS_E5,
        POOL_VENDI_MPNET,
        POOL_VENDI_E5,
    ],
    "ds_all": [
        POOL_DS_MPNET,
        POOL_DS_E5,
    ],
    "vendi_all": [
        POOL_VENDI_MPNET,
        POOL_VENDI_E5,
    ],
    "graph_dense_only": [
        POOL_GRAPH_DENSE_MIXED,
    ],
}

def get_pool_catalog():
    return {pool_id: dict(spec) for pool_id, spec in POOL_CATALOG.items()}

def get_pool_spec(pool_id):
    if pool_id not in POOL_CATALOG:
        raise ValueError(
            f"Unknown pool id: {pool_id}. "
            f"Allowed: {', '.join(sorted(POOL_CATALOG))}"
        )
    return dict(POOL_CATALOG[pool_id])

def get_pool_specs(pool_ids):
    return [get_pool_spec(pool_id) for pool_id in pool_ids]

def get_pool_specs_for_set(pool_set=POOL_SET_ALL_IMPLEMENTED):
    if pool_set not in POOL_SETS:
        raise ValueError(
            f"Unknown pool set: {pool_set}. "
            f"Allowed: {', '.join(sorted(POOL_SETS))}"
        )
    return get_pool_specs(POOL_SETS[pool_set])

def normalize_pool_spec(pool_spec):
    if isinstance(pool_spec, str):
        return get_pool_spec(pool_spec)

    spec = dict(pool_spec)
    retriever = spec["retriever"]
    artifact_embedder_key = spec.get("artifact_embedder_key", spec.get("embedder"))
    if retriever == GRAPH_DENSE:
        artifact_embedder_key = GRAPH_DENSE_MIXED_EMBEDDER_KEY
    artifact_embedder_key = normalize_embedder_key(artifact_embedder_key)
    pool_id = spec.get("pool_id") or f"{retriever}@{artifact_embedder_key}"
    display_label = spec.get("display_label") or spec.get("label") or pool_id
    return {
        **spec,
        "pool_id": pool_id,
        "label": pool_id,
        "display_label": display_label,
        "retriever": retriever,
        "family": spec.get("family", retriever),
        "artifact_embedder_key": artifact_embedder_key,
        "embedder": artifact_embedder_key,
        "pool_size": int(spec.get("pool_size", _pool_size_for_retriever(retriever))),
    }

def describe_pool_member(pool_spec, local_idx):
    spec = normalize_pool_spec(pool_spec)
    local_idx = int(local_idx)
    if local_idx < 0 or local_idx >= spec["pool_size"]:
        raise IndexError(
            f"local_idx={local_idx} out of range for {spec['pool_id']} "
            f"with pool_size={spec['pool_size']}"
        )

    metadata = {
        "pool_id": spec["pool_id"],
        "pool_label": spec["label"],
        "display_label": spec["display_label"],
        "retriever": spec["retriever"],
        "family": spec["family"],
        "artifact_embedder_key": spec["artifact_embedder_key"],
        "local_idx": local_idx,
    }

    if spec["retriever"] == DS:
        gamma, r = DS_POOL_PARAMETERS[local_idx]
        metadata.update({"gamma": gamma, "r": r})
    elif spec["retriever"] == VENDI:
        metadata.update({"s": VENDI_POOL_PARAMETERS[local_idx]})
    elif spec["retriever"] == GRAPH_DENSE:
        params = dict(GRAPH_DENSE_POOL_PARAMETERS[local_idx])
        metadata["parameters"] = params
        metadata.update(params)
    else:
        raise ValueError(f"Unsupported retriever pool: {spec['retriever']}")

    return metadata

def build_retriever_map_for_pools(pool_specs, pool_sizes=None):
    normalized_specs = [normalize_pool_spec(spec) for spec in pool_specs]
    if pool_sizes is None:
        pool_sizes = [spec["pool_size"] for spec in normalized_specs]
    if len(pool_sizes) != len(normalized_specs):
        raise ValueError(
            f"pool_sizes length mismatch: expected {len(normalized_specs)}, got {len(pool_sizes)}"
        )

    retriever_map = []
    for pool_idx, (spec, pool_size) in enumerate(zip(normalized_specs, pool_sizes)):
        expected_size = spec["pool_size"]
        pool_size = int(pool_size)
        if pool_size != expected_size:
            raise ValueError(
                f"Pool size mismatch for {spec['pool_id']}: "
                f"catalog={expected_size}, observed={pool_size}"
            )
        for local_idx in range(pool_size):
            member = describe_pool_member(spec, local_idx)
            member.update({
                "global_idx": len(retriever_map),
                "pool_idx": pool_idx,
            })
            retriever_map.append(member)
    return retriever_map

# Params
EMBEDDER = EMBEDDER_REGISTRY[DEFAULT_EMBEDDER_KEY]
CHUNKING_VERSION = "v1"
CHUNKING_TOKENIZER_MODEL = EMBEDDER
CHUNK_SIZE = 512
OVERLAP = 50
PORTFOLIO_SIZE = 5


get_retrievals_train = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/retrievals_train.pickle'
get_retrievals_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/retrievals_test.pickle'
get_portfolio_retrievals_train = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/portfolio_retrievals_train.pickle'
get_portfolio_retrievals_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/portfolio_retrievals_test.pickle'
get_retriever_scores_train = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/scores_train.pickle'
get_retriever_scores_test = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/scores_test.pickle'
get_retriever_scores_test_f1 = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/scores_test_f1.pickle'
get_portfolio_scores_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/portfolio_scores_test.pickle'
get_retriever_portfolio = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/portfolio.pickle'
get_answer_prompts_train = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/prompts_train.pickle'
get_answer_prompts_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/prompts_test.pickle'
get_selector_prompts = lambda dataset, retriever, model, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/answers/{model}/selector_prompts.pickle'
get_baseline_answer_prompts = lambda dataset: RESULTS_DIR + f'{dataset}/baseline_answer_prompts.pickle'

get_universal_portfolio = lambda retriever, num_docs, embedder=None: RESULTS_DIR + f'portfolio_{_retriever_dir(retriever, num_docs, embedder)}.pickle'
get_universal_portfolio_retrievals_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/universal_portfolio_retrievals_test.pickle'
get_universal_portfolio_scores_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/universal_portfolio_scores_test.pickle'
get_universal_portfolio_union_manifest = lambda portfolio_id, num_docs: RESULTS_DIR + f'portfolio_union_{portfolio_id}_{num_docs}.pickle'
get_universal_portfolio_union = get_universal_portfolio_union_manifest
get_portfolio_union_dir = lambda dataset, portfolio_id, num_docs: RESULTS_DIR + f'{dataset}/portfolio_union_{portfolio_id}_{num_docs}/'
get_portfolio_union_retrievals_test = lambda portfolio_id, dataset, num_docs: get_portfolio_union_dir(dataset, portfolio_id, num_docs) + 'retrievals_test.pickle'
get_portfolio_union_scores_test = lambda portfolio_id, dataset, num_docs: get_portfolio_union_dir(dataset, portfolio_id, num_docs) + 'scores_test.pickle'
get_portfolio_union_scores_test_f1 = lambda portfolio_id, dataset, num_docs: get_portfolio_union_dir(dataset, portfolio_id, num_docs) + 'scores_test_f1.pickle'
get_portfolio_union_materialization_metadata = lambda portfolio_id, dataset, num_docs: get_portfolio_union_dir(dataset, portfolio_id, num_docs) + 'materialization_metadata.pickle'
get_portfolio_union_answer_prompts_test = lambda portfolio_id, dataset, num_docs: get_portfolio_union_dir(dataset, portfolio_id, num_docs) + 'prompts_test.pickle'
get_portfolio_union_answers_all = lambda portfolio_id, dataset, model, num_docs: get_portfolio_union_dir(dataset, portfolio_id, num_docs) + f'answers/{model}/answers.pickle'

get_family_best_baseline_manifest = lambda portfolio_id, num_docs: RESULTS_DIR + f'family_best_baselines_{_safe_artifact_name(portfolio_id)}_{num_docs}.pickle'
get_family_best_dir = lambda portfolio_id, dataset, family, num_docs, max_k: RESULTS_DIR + f'{dataset}/family_best_{_safe_artifact_name(portfolio_id)}_{num_docs}_k{int(max_k)}/{_safe_artifact_name(family)}/'
get_family_best_retrievals_test = lambda portfolio_id, dataset, family, num_docs, max_k: get_family_best_dir(portfolio_id, dataset, family, num_docs, max_k) + 'retrievals_test.pickle'
get_family_best_scores_test = lambda portfolio_id, dataset, family, num_docs, max_k: get_family_best_dir(portfolio_id, dataset, family, num_docs, max_k) + 'scores_test.pickle'
get_family_best_scores_test_f1 = lambda portfolio_id, dataset, family, num_docs, max_k: get_family_best_dir(portfolio_id, dataset, family, num_docs, max_k) + 'scores_test_f1.pickle'
get_family_best_answer_prompts_test = lambda portfolio_id, dataset, family, num_docs, max_k: get_family_best_dir(portfolio_id, dataset, family, num_docs, max_k) + 'prompts_test.pickle'
get_family_best_answers_test = lambda portfolio_id, dataset, family, model, num_docs, max_k: get_family_best_dir(portfolio_id, dataset, family, num_docs, max_k) + f'answers/{model}/answers.pickle'

def get_portfolio_router_prediction_dir(portfolio_id, dataset, num_docs, k, split, run_id):
    if run_id is None or str(run_id).strip() == "":
        raise ValueError("run_id is required for final all-pool portfolio-router predictions.")
    return (
        get_portfolio_union_dir(dataset, portfolio_id, num_docs)
        + f"portfolio_router_predictions/k{int(k)}/"
        + f"{_safe_artifact_name(split)}/{_safe_artifact_name(run_id)}/"
    )

def get_portfolio_router_predictions(portfolio_id, dataset, num_docs, k, split, run_id):
    return get_portfolio_router_prediction_dir(
        portfolio_id,
        dataset,
        num_docs,
        k,
        split,
        run_id,
    ) + "predictions.pickle"

def get_portfolio_router_prediction_metadata(portfolio_id, dataset, num_docs, k, split, run_id):
    return get_portfolio_router_prediction_dir(
        portfolio_id,
        dataset,
        num_docs,
        k,
        split,
        run_id,
    ) + "metadata.json"


def get_portfolio_router_judge_dir(portfolio_id, dataset, num_docs, k, ell, split, run_id, answer_llm):
    if run_id is None or str(run_id).strip() == "":
        raise ValueError("run_id is required for final all-pool portfolio-router judge artifacts.")
    if answer_llm is None or str(answer_llm).strip() == "":
        raise ValueError("answer_llm is required for final all-pool portfolio-router judge artifacts.")
    return (
        get_portfolio_union_dir(dataset, portfolio_id, num_docs)
        + f"portfolio_router_judge/k{int(k)}/"
        + f"ell{int(ell)}/"
        + f"{_safe_artifact_name(split)}/{_safe_artifact_name(run_id)}/"
        + f"answers_{_safe_artifact_name(answer_llm)}/"
    )


def get_portfolio_router_judge_prompts(portfolio_id, dataset, num_docs, k, ell, split, run_id, answer_llm):
    return get_portfolio_router_judge_dir(
        portfolio_id,
        dataset,
        num_docs,
        k,
        ell,
        split,
        run_id,
        answer_llm,
    ) + "prompts.pickle"


def get_portfolio_router_judge_answers(
    portfolio_id,
    dataset,
    num_docs,
    k,
    ell,
    split,
    run_id,
    answer_llm,
    judge_llm,
):
    if judge_llm is None or str(judge_llm).strip() == "":
        raise ValueError("judge_llm is required for final all-pool portfolio-router judge answers.")
    return (
        get_portfolio_router_judge_dir(
            portfolio_id,
            dataset,
            num_docs,
            k,
            ell,
            split,
            run_id,
            answer_llm,
        )
        + f"judge_{_safe_artifact_name(judge_llm)}/answers.pickle"
    )


def get_portfolio_router_judge_scores(
    portfolio_id,
    dataset,
    num_docs,
    k,
    ell,
    split,
    run_id,
    answer_llm,
    judge_llm,
):
    if judge_llm is None or str(judge_llm).strip() == "":
        raise ValueError("judge_llm is required for final all-pool portfolio-router judge scores.")
    return (
        get_portfolio_router_judge_dir(
            portfolio_id,
            dataset,
            num_docs,
            k,
            ell,
            split,
            run_id,
            answer_llm,
        )
        + f"judge_{_safe_artifact_name(judge_llm)}/scores.pickle"
    )


get_single_retriever_retrievals_train = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/single_retrievals_train.pickle'
get_single_retriever_retrievals_test  = lambda dataset, retriever, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/single_retrievals_test.pickle'

# Results
get_answers_all = lambda dataset, retriever, model, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/answers/{model}/answers.pickle'
get_answers_llm_selector = lambda dataset, retriever, model, num_docs, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(retriever, num_docs, embedder)}/answers/{model}/answers_llm_selector.pickle'
get_answers_baseline = lambda dataset, num_docs, model: RESULTS_DIR + f'{dataset}/answers_baseline_{num_docs}_{model}.pickle'

# Vendi-RAG experiment logs (full results + timing summaries)
get_vendirag_results = lambda dataset, num_docs, model, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(VENDI, num_docs, embedder)}/vendirag_{model}.pickle'
get_vendirag_timing =  lambda dataset, num_docs, model, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(VENDI, num_docs, embedder)}/vendirag_timing_{model}.pickle'
get_vendirag_parsed =  lambda dataset, num_docs, model, embedder=None: RESULTS_DIR + f'{dataset}/{_retriever_dir(VENDI, num_docs, embedder)}/vendirag_parsed_{model}.pickle'

# Plots
get_plot_path = (
    lambda dataset, plot_name, retriever, num_docs, model: (
        PLOTS_DIR + f"{dataset}/{plot_name}_{retriever}_{num_docs}_{model}.png"
    )
)

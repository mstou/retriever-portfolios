import os
import pickle
import inspect
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

import constants as C
from vector_db import FaissVectorDB
from data_classes import Questions
from dataset_loaders import (
    HotpotQA_DataLoader,
    HotpotQA_QuestionsLoader,
    MusiQue_DataLoader,
    MusiQue_QuestionsLoader,
    TriviaQA_DataLoader,
    TriviaQA_QuestionsLoader,
    TwoWikiMultiHopQA_DataLoader,
    TwoWikiMultiHopQA_QuestionsLoader,
    get_document_loader,
)
from text_processing import ChunkedCorpusCache, Embedder, TextProcessor
from retrievers import VendiRetriever, BatchVendiRetriever, DiscountedSimilarity, BatchDiscountedSimilarity
from graph_index import GraphIndex, normalize_entity
from graph_retrievers import BatchGraphDenseRetriever
from portfolios import select_portfolio
from prompts import answer_prompt, selector_prompt
from utils import f1_support
from models import OpenAI_LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Corpus Preprocessing and Indexing
# -----------------------------------------------------------------------------

# Functions: _load_corpus_documents, _faiss_index_exists, load_chunk_cache, _save_chunk_cache
#            create_chunk_cache_from_documents, ensure_chunk_cache, backfill_chunk_cache_from_faiss
#            build_faiss_index_from_chunk_cache, index_corpus, questions_train_test_split, save_embeddings
#            save_prefilters

def _load_corpus_documents(dataset_name):
    return get_document_loader(dataset_name).load_documents()

def _faiss_index_exists(folder: str | Path) -> bool:
    folder = Path(folder)
    return (
        (folder / "index.faiss").exists()
        and (folder / "sidecar.json").exists()
        and (folder / "text_units.pkl").exists()
    )

def load_chunk_cache(dataset_name):
    return ChunkedCorpusCache.load(C.get_chunk_cache_dir(dataset_name))

def _save_chunk_cache(dataset, creation_source, text_processor=None):
    if text_processor is not None:
        chunking_metadata = text_processor.get_chunking_metadata()
    else:
        chunking_metadata = {
            "chunking_version": C.CHUNKING_VERSION,
            "chunk_size": C.CHUNK_SIZE,
            "overlap": C.OVERLAP,
            "tokenizer_model": C.CHUNKING_TOKENIZER_MODEL,
        }

    metadata = ChunkedCorpusCache.build_metadata(
        dataset,
        chunking_version=chunking_metadata["chunking_version"],
        chunk_size=chunking_metadata["chunk_size"],
        overlap=chunking_metadata["overlap"],
        tokenizer_model=chunking_metadata["tokenizer_model"],
        creation_source=creation_source,
    )
    ChunkedCorpusCache.save(C.get_chunk_cache_dir(dataset.dataset_name), dataset, metadata)
    return dataset, metadata

def create_chunk_cache_from_documents(dataset_name, creation_source="fresh_chunking"):
    dataset = _load_corpus_documents(dataset_name)
    text_processor = TextProcessor(
        chunk_size=C.CHUNK_SIZE,
        overlap=C.OVERLAP,
        tokenizer_name=C.CHUNKING_TOKENIZER_MODEL,
    )
    dataset.tokenize_and_chunk_documents(text_processor)
    dataset.gather_all_text_units()
    _save_chunk_cache(dataset, creation_source=creation_source, text_processor=text_processor)
    return dataset

def ensure_chunk_cache(dataset_name):
    chunk_cache_dir = C.get_chunk_cache_dir(dataset_name)
    if ChunkedCorpusCache.exists(chunk_cache_dir):
        dataset, _ = load_chunk_cache(dataset_name)
        return dataset
    return create_chunk_cache_from_documents(dataset_name, creation_source="fresh_chunking")

def backfill_chunk_cache_from_faiss(dataset_name, overwrite=False):
    chunk_cache_dir = C.get_chunk_cache_dir(dataset_name)
    if ChunkedCorpusCache.exists(chunk_cache_dir) and not overwrite:
        dataset, metadata = load_chunk_cache(dataset_name)
        return chunk_cache_dir, metadata, dataset

    faiss_dir = C.get_vector_db_dir(dataset_name)
    if not os.path.exists(faiss_dir):
        raise FileNotFoundError(f"Missing FAISS index for dataset {dataset_name}: {faiss_dir}")

    text_units = FaissVectorDB.load_text_units(faiss_dir)
    dataset = _load_corpus_documents(dataset_name)
    dataset.attach_text_units(text_units, strict=True)
    dataset, metadata = _save_chunk_cache(dataset, creation_source="faiss_backfill")
    return chunk_cache_dir, metadata, dataset

def build_faiss_index_from_chunk_cache(dataset_name, device="cpu", embedder=None):
    embedder_key = C.normalize_embedder_key(embedder)
    output_dir = C.get_vector_db_dir(dataset_name, embedder=embedder_key)
    if _faiss_index_exists(output_dir):
        tqdm.write(
            f"[index] skipping existing FAISS index for dataset={dataset_name} "
            f"embedder={embedder_key} at {output_dir}"
        )
        return output_dir

    tqdm.write(
        f"[index] starting dataset={dataset_name} embedder={embedder_key} device={device}"
    )
    tqdm.write("[index] loading chunk cache")
    dataset = ensure_chunk_cache(dataset_name)
    dataset.gather_all_text_units()

    tqdm.write(
        f"[index] dataset={dataset_name} embedder={embedder_key} "
        f"chunks={len(dataset.all_text_units)}"
    )
    tqdm.write("[index] loading embedder model")
    embedder_model = Embedder(device, embedder=embedder_key)
    tqdm.write("[index] embedding passage chunks")
    embedder_model.batch_embed(dataset.all_text_units, role="passage")

    tqdm.write("[index] building FAISS index in memory")
    vector_db = FaissVectorDB(metric="dot", dim=embedder_model.get_embedding_dim())
    vector_db.add_text_units(dataset.all_text_units)

    tqdm.write(f"[index] saving FAISS index to {output_dir}")
    vector_db.save(output_dir)
    tqdm.write(f"[index] saved FAISS index to {output_dir}")
    return output_dir

def index_corpus(dataset_name, device, embedder=None, reuse_chunks_from_default: bool = True):
    del reuse_chunks_from_default
    return build_faiss_index_from_chunk_cache(
        dataset_name,
        device=device,
        embedder=embedder,
    )

def questions_train_test_split(dataset_name, split=0.9):
    questions_train_path = C.get_questions_train(dataset_name)
    questions_test_path = C.get_questions_test(dataset_name)
    if os.path.exists(questions_train_path) and os.path.exists(questions_test_path):
        tqdm.write(
            f"[questions] skipping existing train/test split for dataset={dataset_name}"
        )
        return questions_train_path, questions_test_path

    if dataset_name == C.HotpotQA:
        loader = HotpotQA_DataLoader(C.DATASET_LOCATION[C.HotpotQA])
        _ = loader.load_documents()
        title_to_id = loader.get_title_to_id()
        questions_loader = HotpotQA_QuestionsLoader(
            C.DATASET_LOCATION[C.HotpotQA],
            title_to_id,
            split,
        )
    elif dataset_name == C.MUSIQUE:
        loader = MusiQue_DataLoader(C.DATASET_LOCATION[C.MUSIQUE])
        _ = loader.load_documents()
        title_to_id = loader.get_title_to_id()
        questions_loader = MusiQue_QuestionsLoader(
            C.DATASET_LOCATION[C.MUSIQUE],
            title_to_id=title_to_id,
        )
    elif dataset_name == C.TRIVIAQA:
        loader = TriviaQA_DataLoader(C.DATASET_LOCATION[C.TRIVIAQA])
        _ = loader.load_documents()
        key_to_id = loader.get_key_to_id()
        questions_loader = TriviaQA_QuestionsLoader(
            C.DATASET_LOCATION[C.TRIVIAQA],
            key_to_id=key_to_id,
        )
    elif dataset_name == C.TWOWIKI:
        loader = TwoWikiMultiHopQA_DataLoader(C.DATASET_LOCATION[C.TWOWIKI])
        _ = loader.load_documents()
        title_to_id = loader.get_title_to_id()
        questions_loader = TwoWikiMultiHopQA_QuestionsLoader(
            C.DATASET_LOCATION[C.TWOWIKI],
            title_to_id=title_to_id,
        )
    else:
        raise ValueError(f'Invalid dataset provided: {dataset_name}')

    q_train, q_test = questions_loader.load_questions()

    with open(questions_train_path, 'wb') as f:
        pickle.dump(q_train, f)
    
    with open(questions_test_path, 'wb') as f:
        pickle.dump(q_test, f)

    tqdm.write(
        f"[questions] saved train/test split for dataset={dataset_name}"
    )
    return questions_train_path, questions_test_path

def save_embeddings(dataset_name, device, embedder=None):
    embedder_key = C.normalize_embedder_key(embedder)
    embeddings_train_path = C.get_embeddings_train(dataset_name, embedder=embedder_key)
    embeddings_test_path = C.get_embeddings_test(dataset_name, embedder=embedder_key)
    if os.path.exists(embeddings_train_path) and os.path.exists(embeddings_test_path):
        tqdm.write(
            f"[embeddings] skipping existing question embeddings for dataset={dataset_name} "
            f"embedder={embedder_key}"
        )
        return embeddings_train_path, embeddings_test_path

    with open(C.get_questions_train(dataset_name), 'rb') as f:
        q_train = pickle.load(f)
        
    with open(C.get_questions_test(dataset_name), 'rb') as f:
        q_test = pickle.load(f)
    
    embedder = Embedder(device, embedder=embedder_key)

    queries_train = [q['question'] for q in q_train.questions]
    queries_test  = [q['question'] for q in q_test.questions]

    q_embeddings_train = embedder.embed(queries_train, leave_on_device=False, role="query")
    q_embeddings_test  = embedder.embed(queries_test, leave_on_device=False, role="query")

    with open(embeddings_train_path, 'wb') as f:
        pickle.dump({
            "queries": queries_train,
            "embeddings": q_embeddings_train
        }, f)

    with open(embeddings_test_path, 'wb') as f:
        pickle.dump({
            "queries": queries_test,
            "embeddings": q_embeddings_test
        }, f)

    tqdm.write(
        f"[embeddings] saved question embeddings for dataset={dataset_name} "
        f"embedder={embedder_key}"
    )
    return embeddings_train_path, embeddings_test_path

def save_prefilters(dataset_name, batch_size=1000, prefilter_size=1000, embedder=None):
    embedder_key = C.normalize_embedder_key(embedder)
    prefilters_train_path = C.get_prefilters_train(dataset_name, embedder=embedder_key)
    prefilters_test_path = C.get_prefilters_test(dataset_name, embedder=embedder_key)
    if os.path.exists(prefilters_train_path) and os.path.exists(prefilters_test_path):
        tqdm.write(
            f"[prefilters] skipping existing prefilters for dataset={dataset_name} "
            f"embedder={embedder_key}"
        )
        return prefilters_train_path, prefilters_test_path
    
    print('Loading embeddings..')
    
    with open(C.get_embeddings_train(dataset_name, embedder=embedder_key), 'rb') as f:
        payload_train = pickle.load(f)
        
    with open(C.get_embeddings_test(dataset_name, embedder=embedder_key), 'rb') as f:
        payload_test = pickle.load(f)
    
    print('Loading vector database..')

    vector_db = FaissVectorDB.load(C.get_vector_db_dir(dataset_name, embedder=embedder_key))

    queries_train = payload_train['queries']
    q_embeddings_train = payload_train['embeddings']

    queries_test  = payload_test['queries']
    q_embeddings_test = payload_test['embeddings']

    prefiltered_tus_train = []
    for i in tqdm(range(0, len(q_embeddings_train), batch_size)):
        batch = q_embeddings_train[i:i+batch_size]
        res = vector_db.batch_search(batch, k=prefilter_size)
        prefiltered_tus_train.extend([[tu for (tu, _) in r] for r in res])
    
    prefiltered_tus_test = []
    for i in tqdm(range(0, len(q_embeddings_test), batch_size)):
        batch = q_embeddings_test[i:i+batch_size]
        res = vector_db.batch_search(batch, k=prefilter_size)
        prefiltered_tus_test.extend([[tu for (tu, _) in r] for r in res])
    
    with open(prefilters_train_path, 'wb') as f:
        pickle.dump({
            "queries": queries_train,
            "candidates": prefiltered_tus_train,
        }, f)

    with open(prefilters_test_path, 'wb') as f:
        pickle.dump({
            "queries": queries_test,
            "candidates": prefiltered_tus_test,
        }, f)
    tqdm.write(
        f"[prefilters] saved prefilters for dataset={dataset_name} "
        f"embedder={embedder_key} prefilter_size={prefilter_size}"
    )
    return prefilters_train_path, prefilters_test_path

# -----------------------------------------------------------------------------
# Shared Artifact and Question Helpers
# -----------------------------------------------------------------------------

# Functions: _questions_file_for_split, _load_questions_for_split, _artifact_embedder_for_retriever
#            _translated_artifact_path, _artifact_exists_for_audit, _artifact_read_path, _artifact_write_path

def _questions_file_for_split(dataset_name, split):
    split = split.lower()
    if split == "train":
        return C.get_questions_train(dataset_name)
    if split == "test":
        return C.get_questions_test(dataset_name)
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")

def _load_questions_for_split(dataset_name, split):
    questions_path = _questions_file_for_split(dataset_name, split)
    questions_read_path = _artifact_read_path(questions_path)
    if not os.path.exists(questions_read_path):
        raise FileNotFoundError(
            f"Missing questions file for dataset={dataset_name} split={split}: "
            f"expected_path={questions_path}, checked_path={questions_read_path}"
        )
    with open(questions_read_path, "rb") as f:
        return pickle.load(f)

def _artifact_embedder_for_retriever(retriever, embedder=None):
    if retriever == C.GRAPH_DENSE:
        return C.GRAPH_DENSE_MIXED_EMBEDDER_KEY
    return C.normalize_embedder_key(embedder)

def _translated_artifact_path(path):
    return Path(path)

def _artifact_exists_for_audit(path):
    expected_path = Path(path)
    if expected_path.exists():
        return True, str(expected_path)
    translated_path = _translated_artifact_path(expected_path)
    if translated_path != expected_path and translated_path.exists():
        return True, str(translated_path)
    checked_path = translated_path if translated_path != expected_path else expected_path
    return False, str(checked_path)

def _artifact_read_path(path):
    expected_path = Path(path)
    if expected_path.exists():
        return str(expected_path)
    translated_path = _translated_artifact_path(expected_path)
    if translated_path != expected_path and translated_path.exists():
        return str(translated_path)
    return str(expected_path)

def _artifact_write_path(path):
    expected_path = Path(path)
    if expected_path.parent.exists():
        return str(expected_path)
    return str(expected_path)

# -----------------------------------------------------------------------------
# Graph-Dense Retrieval Utilities
# -----------------------------------------------------------------------------

# Functions: load_query_entity_cache, build_graph_query_entity_cache_from_extraction_results
#            _load_graph_dense_batch_retriever, _load_q_embeddings_if_available
#            _load_q_embeddings_by_embedder_if_available, _compute_retrievals_graph_dense
#            compute_retrievals_train_graph_dense, compute_retrievals_test_graph_dense

def load_query_entity_cache(dataset_name: str, split: str) -> dict:
    split = split.lower()
    cache_path = Path(C.get_graph_query_entities_path(dataset_name, split))
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing graph query entity cache: {cache_path}. "
            "Build it from saved question entity extraction results before running "
            "graph_dense retrieval."
        )

    with cache_path.open("rb") as f:
        payload = pickle.load(f)

    if payload.get("dataset") != dataset_name:
        raise ValueError(
            f"Graph query entity cache dataset mismatch at {cache_path}: "
            f"expected {dataset_name}, got {payload.get('dataset')}"
        )
    if payload.get("split") != split:
        raise ValueError(
            f"Graph query entity cache split mismatch at {cache_path}: "
            f"expected {split}, got {payload.get('split')}"
        )

    queries = payload.get("queries")
    entities = payload.get("entities")
    if not isinstance(queries, list) or not isinstance(entities, list):
        raise ValueError(f"Graph query entity cache at {cache_path} must contain list queries/entities")
    if len(queries) != len(entities):
        raise ValueError(
            f"Graph query entity cache length mismatch at {cache_path}: "
            f"queries={len(queries)}, entities={len(entities)}"
        )

    questions_path = Path(_questions_file_for_split(dataset_name, split))
    if questions_path.exists():
        questions_dataset = _load_questions_for_split(dataset_name, split)
        active_queries = [q["question"] for q in questions_dataset.questions]
        if len(active_queries) != len(queries):
            raise ValueError(
                f"Graph query entity cache length does not match {questions_path}: "
                f"cache={len(queries)}, questions={len(active_queries)}"
            )
        mismatches = [
            idx for idx, (cached, active) in enumerate(zip(queries, active_queries))
            if cached != active
        ]
        if mismatches:
            first_idx = mismatches[0]
            raise ValueError(
                f"Graph query entity cache query strings do not match {questions_path}; "
                f"first mismatch index={first_idx}, cache={queries[first_idx]!r}, "
                f"questions={active_queries[first_idx]!r}"
            )

    normalized_entities = []
    for entity_list in entities:
        normalized_entities.append([
            normalized
            for normalized in (normalize_entity(entity) for entity in entity_list)
            if normalized
        ])
    payload["entities"] = normalized_entities
    payload["path"] = str(cache_path)
    return payload

def build_graph_query_entity_cache_from_extraction_results(
    dataset_name: str,
    split: str,
    *,
    results_path: str | Path | None = None,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    split = split.lower()
    questions_dataset = _load_questions_for_split(dataset_name, split)
    queries = [q["question"] for q in questions_dataset.questions]

    resolved_results_path = Path(
        results_path
        if results_path is not None
        else C.get_graph_question_entity_extraction_results_path(dataset_name, split)
    )
    if not resolved_results_path.exists():
        raise FileNotFoundError(
            f"Missing saved question entity extraction results: {resolved_results_path}"
        )

    resolved_output_path = Path(
        output_path
        if output_path is not None
        else C.get_graph_query_entities_path(dataset_name, split)
    )
    if resolved_output_path.exists() and not overwrite:
        return resolved_output_path

    with resolved_results_path.open("rb") as f:
        source_payload = pickle.load(f)
    answers = (
        source_payload.get("answers", [])
        if isinstance(source_payload, dict)
        else source_payload
    )

    entities_by_question_id: dict[int, list[str]] = {}
    for answer in answers:
        if answer is None or answer.get("kind") != "question":
            continue
        if answer.get("split") is not None and answer.get("split") != split:
            continue
        question_id = answer.get("question_id")
        if not isinstance(question_id, int):
            continue
        if 0 <= question_id < len(queries) and answer.get("question") not in (None, queries[question_id]):
            raise ValueError(
                f"Question text mismatch in {resolved_results_path} at question_id={question_id}: "
                f"results={answer.get('question')!r}, questions={queries[question_id]!r}"
            )
        entities_by_question_id[question_id] = [
            normalized
            for normalized in (normalize_entity(entity) for entity in (answer.get("entities") or []))
            if normalized
        ]

    missing = [idx for idx in range(len(queries)) if idx not in entities_by_question_id]
    if missing:
        raise ValueError(
            f"Question entity extraction results at {resolved_results_path} are missing "
            f"{len(missing)} question ids; first missing ids: {missing[:10]}"
        )

    payload = {
        "dataset": dataset_name,
        "split": split,
        "queries": queries,
        "entities": [entities_by_question_id[idx] for idx in range(len(queries))],
        "source_results_path": str(resolved_results_path),
    }

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return resolved_output_path

def _load_graph_dense_batch_retriever(
    dataset_name,
    split,
    device,
    retriever_params=None,
):
    retriever_params = retriever_params or C.GRAPH_DENSE_POOL_PARAMETERS
    embedder_keys = list(dict.fromkeys(
        C.normalize_embedder_key(param.get("embedder"))
        for param in retriever_params
    ))
    graph_index_path = C.get_graph_index_path(dataset_name)
    if not os.path.exists(graph_index_path):
        raise FileNotFoundError(f"Missing graph index for graph_dense retrieval: {graph_index_path}")

    graph_index = GraphIndex.load(graph_index_path)
    missing_chunk_embedding_keys = [
        key for key in embedder_keys
        if key not in getattr(graph_index, "chunk_embeddings", {})
    ]
    if missing_chunk_embedding_keys:
        graph_index.attach_chunk_embeddings_from_vector_dbs(
            dataset_name,
            embedders=missing_chunk_embedding_keys,
        )

    query_entity_cache = load_query_entity_cache(dataset_name, split)
    embedders = {
        embedder_key: Embedder(device, embedder=embedder_key)
        for embedder_key in embedder_keys
    }
    batch_retriever = BatchGraphDenseRetriever(
        graph_index=graph_index,
        embedder=embedders,
        retriever_params=retriever_params,
        query_entities=query_entity_cache["entities"],
        device=device,
    )
    return batch_retriever, query_entity_cache, graph_index_path

def _load_q_embeddings_if_available(dataset_name, split, embedder_key, expected_queries):
    embeddings_path = (
        C.get_embeddings_train(dataset_name, embedder=embedder_key)
        if split == "train"
        else C.get_embeddings_test(dataset_name, embedder=embedder_key)
    )
    if not os.path.exists(embeddings_path):
        tqdm.write(
            f"[graph-dense] question embeddings missing at {embeddings_path}; "
            "queries will be embedded on demand"
        )
        return None

    with open(embeddings_path, "rb") as f:
        q_embeddings = pickle.load(f)
    if q_embeddings.get("queries") != expected_queries:
        raise ValueError(
            f"Question embedding query order mismatch at {embeddings_path}; "
            "recompute embeddings for this split/embedder."
        )
    return q_embeddings["embeddings"]

def _load_q_embeddings_by_embedder_if_available(dataset_name, split, embedder_keys, expected_queries):
    embeddings_by_key = {}
    for embedder_key in embedder_keys:
        embeddings = _load_q_embeddings_if_available(
            dataset_name,
            split,
            embedder_key,
            expected_queries,
        )
        if embeddings is not None:
            embeddings_by_key[embedder_key] = embeddings
    return embeddings_by_key or None

def _compute_retrievals_graph_dense(
    dataset_name,
    split,
    num_docs_to_fetch=4,
    device="cpu",
    embedder=None,
    retriever_params=None,
    output_file=None,
):
    split = split.lower()
    artifact_embedder_key = C.GRAPH_DENSE_MIXED_EMBEDDER_KEY
    retriever_params = retriever_params or C.GRAPH_DENSE_POOL_PARAMETERS
    embedder_keys = list(dict.fromkeys(
        C.normalize_embedder_key(param.get("embedder"))
        for param in retriever_params
    ))
    questions_dataset = _load_questions_for_split(dataset_name, split)
    queries = [q["question"] for q in questions_dataset.questions]
    batch_retriever, query_entity_cache, graph_index_path = _load_graph_dense_batch_retriever(
        dataset_name=dataset_name,
        split=split,
        device=device,
        retriever_params=retriever_params,
    )
    q_embeddings = _load_q_embeddings_by_embedder_if_available(
        dataset_name,
        split,
        embedder_keys,
        queries,
    )
    candidates_per_query = [[] for _ in queries]

    if output_file is None:
        output_file = (
            C.get_retrievals_train(dataset_name, C.GRAPH_DENSE, num_docs_to_fetch, embedder=artifact_embedder_key)
            if split == "train"
            else C.get_retrievals_test(dataset_name, C.GRAPH_DENSE, num_docs_to_fetch, embedder=artifact_embedder_key)
        )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"[graph-dense] dataset={dataset_name} split={split} embedders={embedder_keys} "
        f"variants={batch_retriever.num_retrievers()} graph_index={graph_index_path} "
        f"query_entity_cache={query_entity_cache['path']}"
    )

    return precompute_retrievals_to_file(
        questions_dataset=questions_dataset,
        batch_retriever=batch_retriever,
        q_embeddings=q_embeddings,
        candidates_per_query=candidates_per_query,
        retriever_results=num_docs_to_fetch,
        output_file=output_file,
        prefilter_num=0,
    )

def compute_retrievals_train_graph_dense(
    dataset_name,
    num_docs_to_fetch=4,
    device="cpu",
    embedder=None,
):
    return _compute_retrievals_graph_dense(
        dataset_name=dataset_name,
        split="train",
        num_docs_to_fetch=num_docs_to_fetch,
        device=device,
        embedder=embedder,
    )

def compute_retrievals_test_graph_dense(
    dataset_name,
    num_docs_to_fetch=4,
    device="cpu",
    embedder=None,
):
    return _compute_retrievals_graph_dense(
        dataset_name=dataset_name,
        split="test",
        num_docs_to_fetch=num_docs_to_fetch,
        device=device,
        embedder=embedder,
    )

# -----------------------------------------------------------------------------
# Retrieval Computation and Recall Scoring
# -----------------------------------------------------------------------------

# Functions: precompute_retrievals_to_file, compute_recalls_to_file, compute_full_pool_recalls_to_file
#            compute_portfolio_prefix_argmax_recalls, compute_retrievals_train, compute_retrievals_test
#            compute_portfolio_retrievals_test, compute_single_retriever_retrievals

def precompute_retrievals_to_file(
    questions_dataset,
    batch_retriever,
    q_embeddings,
    candidates_per_query,
    retriever_results: int = 5,
    output_file: str = None,
    prefilter_num = 300,
    save_every = 1000,
    num_questions = None):
    """
        Receives prefiltered candidates & query embeddings, runs batch_retriever.query
        for all questions and saves the result in the output file.

        Supports intermediate checkpoins using the output_file.
    """
    if num_questions is None: num_questions = len(questions_dataset.questions)

    # Map dataset questions to indices
    given_questions = [q["question"] for q in questions_dataset.questions[:num_questions]]
    idx_map = {q: i for i, q in enumerate(given_questions[:num_questions])}
    task_indices = [idx_map[q] for q in given_questions]

    # Parallel calls with tqdm progress bar
    results_all = [None] * len(task_indices)
    completed = 0
    
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            old = pickle.load(f)
            results_all = old["results"]
            completed = sum(r is not None for r in results_all)
            print(f"Resuming from checkpoint: {completed}/{len(results_all)} done.")
    
    missing = [pos for pos, res in enumerate(results_all) if res is None]
    if not missing:
        print("All queries already completed.")
        return output_file

    query_signature = inspect.signature(batch_retriever.query)
    supports_query_idx = "query_idx" in query_signature.parameters
    
    # Task wrapper
    def _run_one(i):
        q_text = given_questions[i]
        cands = candidates_per_query[i][:prefilter_num] if candidates_per_query is not None else None
        if isinstance(q_embeddings, dict):
            q_vec = {
                embedder_key: embeddings[i]
                for embedder_key, embeddings in q_embeddings.items()
            }
        else:
            q_vec = q_embeddings[i] if q_embeddings is not None else None
        query_kwargs = {
            "num_results": retriever_results,
            "candidates": cands,
            "q_vec": q_vec,
        }
        if supports_query_idx:
            query_kwargs["query_idx"] = i
        return batch_retriever.query(q_text, **query_kwargs)
    
    completed = 0
    for idx in tqdm(missing):
        results_all[idx] = _run_one(idx)
        completed += 1
        if completed % save_every == 0:
            with open(output_file, "wb") as f:
                pickle.dump({"queries": given_questions, "results": results_all,}, f,)
            print(f"Checkpoint saved!",flush=True)
    
    # Save final results
    out = {
        "queries": given_questions,
        "results": results_all,
    }
    
    with open(output_file, "wb") as f:
        pickle.dump(out, f)

    return output_file

def compute_recalls_to_file(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    split='train',
    universal: bool = False,
    embedder=None,
):
    """
        Loads retrieval outputs for a retriever configuration and computes per-question recall.
        Persists an R x Q matrix (retrievers x questions) to the appropriate scores file.
        If split='test', it uses the recalls of the portfolio on test.
    """
    embedder = _artifact_embedder_for_retriever(retriever, embedder)
    split = split.lower()
    questions_path = (
        C.get_questions_train(dataset_name)
        if split == "train"
        else C.get_questions_test(dataset_name)
    )
    with open(questions_path, "rb") as f:
        questions_dataset = pickle.load(f)

    if split == "train":
        retrievals_file = C.get_retrievals_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        scores_file = C.get_retriever_scores_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
    else:
        if universal:
            retrievals_file = C.get_universal_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
            scores_file = C.get_universal_portfolio_scores_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        else:
            retrievals_file = C.get_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
            scores_file = C.get_portfolio_scores_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)

    with open(retrievals_file, "rb") as f:
        payload = pickle.load(f)

    queries_saved = payload["queries"]
    results = payload["results"]
    num_questions = len(results)

    num_retrievers = len(results[0])
    recall_matrix = [[0.0 for _ in range(num_questions)] for _ in range(num_retrievers)]

    for qid in range(num_questions):
        gold_docs = questions_dataset.questions[qid]["target"]
        for rid in range(num_retrievers):
            retrieved_units = results[qid][rid]
            retrieved_doc_ids = [tu.doc_id for tu in retrieved_units]
            _, _, recall = f1_support(retrieved_doc_ids, gold_docs)
            recall_matrix[rid][qid] = recall

    Path(scores_file).parent.mkdir(parents=True, exist_ok=True)
    with open(scores_file, "wb") as f:
        pickle.dump(recall_matrix, f)

    return scores_file

def compute_full_pool_recalls_to_file(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    split="train",
    embedder=None,
):
    """
    Compute full-pool recall scores from retrievals_{split}.pickle.

    Unlike compute_recalls_to_file(split="test"), this writes full-pool test
    recalls to C.get_retriever_scores_test rather than portfolio test scores.
    """
    embedder = _artifact_embedder_for_retriever(retriever, embedder)
    split = split.lower()
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    questions_path = _questions_file_for_split(dataset_name, split)
    questions_read_path = _artifact_read_path(questions_path)
    if not os.path.exists(questions_read_path):
        raise FileNotFoundError(
            f"Missing questions file for dataset={dataset_name} split={split}: "
            f"expected_path={questions_path}, checked_path={questions_read_path}"
        )
    with open(questions_read_path, "rb") as f:
        questions_dataset = pickle.load(f)
    target_queries = [q["question"] for q in questions_dataset.questions]

    retrievals_file = (
        C.get_retrievals_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        if split == "train"
        else C.get_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
    )
    scores_file = (
        C.get_retriever_scores_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        if split == "train"
        else C.get_retriever_scores_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
    )
    retrievals_read_path = _artifact_read_path(retrievals_file)
    if not os.path.exists(retrievals_read_path):
        raise FileNotFoundError(
            f"Missing full-pool retrievals for dataset={dataset_name} split={split} "
            f"retriever={retriever} embedder={embedder}: expected_path={retrievals_file}, "
            f"checked_path={retrievals_read_path}"
        )

    with open(retrievals_read_path, "rb") as f:
        payload = pickle.load(f)
    queries_saved = payload["queries"]
    results = payload["results"]
    if len(queries_saved) != len(target_queries) or len(results) != len(target_queries):
        raise ValueError(
            f"Question count mismatch for dataset={dataset_name} split={split} "
            f"retriever={retriever} embedder={embedder}: "
            f"queries={len(queries_saved)}, results={len(results)}, "
            f"questions={len(target_queries)}, path={retrievals_file}"
        )
    for q_idx, (saved, expected) in enumerate(zip(queries_saved, target_queries)):
        if saved != expected:
            raise ValueError(
                f"Query order mismatch for dataset={dataset_name} split={split} "
                f"retriever={retriever} embedder={embedder} at q_idx={q_idx}: "
                f"retrieval_query={saved!r}, questions_query={expected!r}, "
                f"path={retrievals_file}"
            )

    num_questions = len(results)
    num_retrievers = len(results[0]) if results else 0
    recall_matrix = [[0.0 for _ in range(num_questions)] for _ in range(num_retrievers)]

    for qid in tqdm(range(num_questions), desc=f"Full-pool recalls {dataset_name} {split} {retriever}"):
        gold_docs = questions_dataset.questions[qid]["target"]
        for rid in range(num_retrievers):
            retrieved_units = results[qid][rid]
            if retrieved_units is None:
                raise ValueError(
                    f"Missing retrieval results for dataset={dataset_name} split={split} "
                    f"retriever={retriever} embedder={embedder} qid={qid} rid={rid}"
                )
            retrieved_doc_ids = [tu.doc_id for tu in retrieved_units]
            _, _, recall = f1_support(retrieved_doc_ids, gold_docs)
            recall_matrix[rid][qid] = recall

    scores_write_path = _artifact_write_path(scores_file)
    Path(scores_write_path).parent.mkdir(parents=True, exist_ok=True)
    with open(scores_write_path, "wb") as f:
        pickle.dump(recall_matrix, f)
    if scores_write_path != scores_file:
        print(
            f"[full-pool-recalls] writing translated path: expected={scores_file} "
            f"actual={scores_write_path}",
            flush=True,
        )
    print(
        f"[full-pool-recalls] wrote dataset={dataset_name} split={split} "
        f"retriever={retriever} embedder={embedder} scores={scores_file}",
        flush=True,
    )
    return scores_file

def compute_portfolio_prefix_argmax_recalls(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    split="train",
    universal: bool = False,
    embedder=None,
):
    """
        For a given dataset / retriever pool / #docs / split, load the
        (already computed) portfolio and per-retriever recall scores and
        compute, for every prefix of the portfolio:
          - the average recall of the best retriever in that prefix
            (argmax over retrievers), and
          - the average recall of a random retriever from that prefix
            (mean over retrievers).

        Returns two lists of length K (K = portfolio size):
          - avg_recalls[k] is the average argmax recall using the first k+1 retrievers
          - random_recalls[k] is the average recall of a random retriever in the first k+1
            retrievers (i.e., mean over the prefix).
    """
    split = split.lower()
    embedder = _artifact_embedder_for_retriever(retriever, embedder)

    # Load portfolio (indices refer to positions in the full pool)
    if universal:
        portfolio_path = C.get_universal_portfolio(retriever, num_docs_to_fetch, embedder=embedder)
    else:
        portfolio_path = C.get_retriever_portfolio(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)

    with open(portfolio_path, "rb") as f:
        portfolio_payload = pickle.load(f)
    portfolio_indices = portfolio_payload["portfolio"]

    # Load recall scores and restrict to portfolio retrievers
    if split == "train":
        scores_path = C.get_retriever_scores_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        with open(scores_path, "rb") as f:
            recall_matrix = pickle.load(f)  # [num_retrievers][num_questions]
        # Select rows corresponding to the portfolio, preserving order
        scores_subset = [recall_matrix[idx] for idx in portfolio_indices]
    elif split == "test":
        if universal:
            scores_path = C.get_universal_portfolio_scores_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        else:
            scores_path = C.get_portfolio_scores_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        with open(scores_path, "rb") as f:
            recall_matrix = pickle.load(f)  # [portfolio_size][num_questions]
        # On test, scores are stored only for portfolio retrievers in order
        scores_subset = recall_matrix
    else:
        raise ValueError(f"Unsupported split: {split}")

    if not scores_subset:
        return []

    num_retrievers = len(scores_subset)
    num_questions = len(scores_subset[0])

    avg_recalls = []
    random_recalls = []
    for k in range(1, num_retrievers + 1):
        total_best_recall = 0.0
        total_random_recall = 0.0
        for q in range(num_questions):
            best = 0.0
            sum_scores = 0.0
            for r in range(k):
                score = scores_subset[r][q]
                sum_scores += score
                if score > best:
                    best = score
            total_best_recall += best
            total_random_recall += (sum_scores / k if k > 0 else 0.0)
        denom = num_questions if num_questions > 0 else 1
        avg_recalls.append(total_best_recall / denom)
        random_recalls.append(total_random_recall / denom)

    return avg_recalls, random_recalls

def compute_retrievals_train(
    dataset_name,
    retriever,
    num_docs_to_fetch=4,
    device='cpu',
    embedder=None,
    prefilter_num=1000,
):
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    if retriever == C.GRAPH_DENSE:
        return compute_retrievals_train_graph_dense(
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs_to_fetch,
            device=device,
            embedder=embedder_key,
        )

    vector_db = FaissVectorDB.load(C.get_vector_db_dir(dataset_name, embedder=embedder_key))
    embedder = Embedder(device, embedder=embedder_key)

    if retriever == C.VENDI:
        retrievers = []
        for s in C.VENDI_POOL_PARAMETERS:
            retrievers.append(VendiRetriever(embedder, vector_db, s, device))
        
        batch_retriever = BatchVendiRetriever(retrievers, device)
    
    elif retriever == C.DS:
        retrievers = []
        for z in C.DS_POOL_PARAMETERS:
            gamma, r = z
            retrievers.append(DiscountedSimilarity(embedder, vector_db, gamma, r, metric='dot', device=device))

        batch_retriever = BatchDiscountedSimilarity(retrievers, device)
    
    else:
        raise ValueError(f'Invalid retriever argument: {retriever}')
    
    with open(C.get_questions_train(dataset_name), 'rb') as f:
        questions_dataset_train = pickle.load(f)
    
    with open(C.get_embeddings_train(dataset_name, embedder=embedder_key), 'rb') as f:
        q_embeddings_train = pickle.load(f)
    
    with open(C.get_prefilters_train(dataset_name, embedder=embedder_key), 'rb') as f:
        prefilters = pickle.load(f)
    
    
    output_file = C.get_retrievals_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"[train-retrievals] dataset={dataset_name} retriever={retriever} "
        f"embedder={embedder_key} prefilter_num={prefilter_num} "
        f"questions={len(questions_dataset_train.questions)}"
    )

    precompute_retrievals_to_file(
        questions_dataset=questions_dataset_train,
        batch_retriever=batch_retriever,
        q_embeddings=q_embeddings_train["embeddings"],
        candidates_per_query=prefilters["candidates"],
        retriever_results=num_docs_to_fetch,
        output_file=output_file,
        prefilter_num=prefilter_num,
    )

def compute_retrievals_test(
    dataset_name,
    retriever,
    num_docs_to_fetch=4,
    device='cpu',
    prefilter_num=1000,
    embedder=None,
):
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    if retriever == C.GRAPH_DENSE:
        return compute_retrievals_test_graph_dense(
            dataset_name=dataset_name,
            num_docs_to_fetch=num_docs_to_fetch,
            device=device,
            embedder=embedder_key,
        )

    vector_db = FaissVectorDB.load(C.get_vector_db_dir(dataset_name, embedder=embedder_key))
    embedder = Embedder(device, embedder=embedder_key)

    if retriever == C.VENDI:
        retrievers = [
            VendiRetriever(embedder, vector_db, s, device)
            for s in C.VENDI_POOL_PARAMETERS
        ]
        batch_retriever = BatchVendiRetriever(retrievers, device)
    elif retriever == C.DS:
        retrievers = []
        for gamma, r in C.DS_POOL_PARAMETERS:
            retrievers.append(
                DiscountedSimilarity(embedder, vector_db, gamma, r, metric='dot', device=device)
            )
        batch_retriever = BatchDiscountedSimilarity(retrievers, device)
    else:
        raise ValueError(f'Invalid retriever argument: {retriever}')

    with open(C.get_questions_test(dataset_name), 'rb') as f:
        questions_dataset_test = pickle.load(f)

    with open(C.get_embeddings_test(dataset_name, embedder=embedder_key), 'rb') as f:
        q_embeddings_test = pickle.load(f)

    with open(C.get_prefilters_test(dataset_name, embedder=embedder_key), 'rb') as f:
        prefilters = pickle.load(f)

    output_file = C.get_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    precompute_retrievals_to_file(
        questions_dataset=questions_dataset_test,
        batch_retriever=batch_retriever,
        q_embeddings=q_embeddings_test["embeddings"],
        candidates_per_query=prefilters["candidates"],
        retriever_results=num_docs_to_fetch,
        output_file=output_file,
        prefilter_num=prefilter_num,
    )

    return output_file

def compute_portfolio_retrievals_test(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    device='cpu',
    prefilter_num=1000,
    universal: bool = False,
    embedder=None,
):
    """
        Runs retrieval on the test split using only the retrievers in the saved portfolio.

        If universal=False (default), uses the dataset-specific portfolio stored at
        C.get_retriever_portfolio(dataset_name, retriever, num_docs_to_fetch) and writes
        outputs to C.get_portfolio_retrievals_test(...).

        If universal=True, uses the universal portfolio stored at
        C.get_universal_portfolio(retriever, num_docs_to_fetch) and writes outputs to
        C.get_universal_portfolio_retrievals_test(...).
    """
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    if universal:
        portfolio_path = C.get_universal_portfolio(retriever, num_docs_to_fetch, embedder=embedder_key)
    else:
        portfolio_path = C.get_retriever_portfolio(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)

    with open(portfolio_path, "rb") as f:
        portfolio_payload = pickle.load(f)
    portfolio_indices = portfolio_payload["portfolio"]

    if retriever == C.GRAPH_DENSE:
        selected_params = [
            C.GRAPH_DENSE_POOL_PARAMETERS[idx]
            for idx in portfolio_indices
        ]
        output_file = (
            C.get_universal_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
            if universal
            else C.get_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
        )
        return _compute_retrievals_graph_dense(
            dataset_name=dataset_name,
            split="test",
            num_docs_to_fetch=num_docs_to_fetch,
            device=device,
            retriever_params=selected_params,
            output_file=output_file,
        )

    vector_db = FaissVectorDB.load(C.get_vector_db_dir(dataset_name, embedder=embedder_key))
    embedder = Embedder(device, embedder=embedder_key)

    if retriever == C.VENDI:
        retrievers = [
            VendiRetriever(embedder, vector_db, C.VENDI_POOL_PARAMETERS[idx], device)
            for idx in portfolio_indices
        ]
        batch_retriever = BatchVendiRetriever(retrievers, device)
    elif retriever == C.DS:
        retrievers = []
        for idx in portfolio_indices:
            gamma, r = C.DS_POOL_PARAMETERS[idx]
            retrievers.append(
                DiscountedSimilarity(embedder, vector_db, gamma, r, metric="dot", device=device)
            )
        batch_retriever = BatchDiscountedSimilarity(retrievers, device)
    else:
        raise ValueError(f"Unsupported retriever type: {retriever}")

    # Running on test
    questions_file = C.get_questions_test(dataset_name)
    embeddings_file = C.get_embeddings_test(dataset_name, embedder=embedder_key)
    prefilters_file = C.get_prefilters_test(dataset_name, embedder=embedder_key)
    if universal:
        output_file = C.get_universal_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    else:
        output_file = C.get_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(questions_file, "rb") as f:
        questions_dataset = pickle.load(f)
    with open(embeddings_file, "rb") as f:
        q_embeddings = pickle.load(f)
    with open(prefilters_file, "rb") as f:
        prefilters = pickle.load(f)

    precompute_retrievals_to_file(
        questions_dataset=questions_dataset,
        batch_retriever=batch_retriever,
        q_embeddings=q_embeddings["embeddings"],
        candidates_per_query=prefilters["candidates"],
        retriever_results=num_docs_to_fetch,
        output_file=output_file,
        prefilter_num=prefilter_num,
    )

    return output_file

def compute_single_retriever_retrievals(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    portfolio_size=10,
    split='test',
    device='cpu',
    prefilter_num=1000,
    universal: bool = False,
    embedder=None,
):
    """
        Fetches portfolio_size * num_docs_to_fetch documents using the first portfolio retriever.
        Saves the per-question results for the requested split (train/test).
    """
    split = split.lower()
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    num_results = portfolio_size * num_docs_to_fetch

    if split == "train":
        questions_file = C.get_questions_train(dataset_name)
        embeddings_file = C.get_embeddings_train(dataset_name, embedder=embedder_key)
        prefilters_file = C.get_prefilters_train(dataset_name, embedder=embedder_key)
        output_file = C.get_single_retriever_retrievals_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    else:
        questions_file = C.get_questions_test(dataset_name)
        embeddings_file = C.get_embeddings_test(dataset_name, embedder=embedder_key)
        prefilters_file = C.get_prefilters_test(dataset_name, embedder=embedder_key)
        output_file = C.get_single_retriever_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)

    if universal:
        portfolio_path = C.get_universal_portfolio(retriever, num_docs_to_fetch, embedder=embedder_key)
    else:
        portfolio_path = C.get_retriever_portfolio(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)

    with open(portfolio_path, "rb") as f:
        portfolio_payload = pickle.load(f)
    single_idx = portfolio_payload["portfolio"][0]

    if retriever == C.GRAPH_DENSE:
        with open(questions_file, "rb") as f:
            questions_dataset = pickle.load(f)
        queries = [q["question"] for q in questions_dataset.questions]
        selected_param = C.GRAPH_DENSE_POOL_PARAMETERS[single_idx]
        selected_embedder_key = C.normalize_embedder_key(selected_param["embedder"])
        q_vecs = _load_q_embeddings_if_available(dataset_name, split, selected_embedder_key, queries)
        batch_retriever, _query_entity_cache, _graph_index_path = _load_graph_dense_batch_retriever(
            dataset_name=dataset_name,
            split=split,
            device=device,
            retriever_params=[selected_param],
        )

        results = []
        for i in tqdm(range(len(queries)), desc=f"Single graph_dense retriever ({split})"):
            q_vec = q_vecs[i] if q_vecs is not None else None
            retrieved = batch_retriever.query(
                queries[i],
                num_results=num_results,
                candidates=None,
                q_vec=q_vec,
                query_idx=i,
            )[0]
            results.append(retrieved)

        payload = {
            "queries": queries,
            "results": results,
            "retriever_index": single_idx,
            "docs_per_query": num_results,
        }

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "wb") as f:
            pickle.dump(payload, f)

        return output_file

    vector_db = FaissVectorDB.load(C.get_vector_db_dir(dataset_name, embedder=embedder_key))
    embedder = Embedder(device, embedder=embedder_key)

    if retriever == C.VENDI:
        param = C.VENDI_POOL_PARAMETERS[single_idx]
        batch_retriever = BatchVendiRetriever(
            [VendiRetriever(embedder, vector_db, param, device=device)],
            device,
        )
    elif retriever == C.DS:
        gamma, r = C.DS_POOL_PARAMETERS[single_idx]
        batch_retriever = BatchDiscountedSimilarity(
            [DiscountedSimilarity(embedder, vector_db, gamma, r, metric="dot", device=device)],
            device,
        )
    else:
        raise ValueError(f"Unsupported retriever type: {retriever}")

    with open(questions_file, "rb") as f:
        questions_dataset = pickle.load(f)
    with open(embeddings_file, "rb") as f:
        q_embeddings = pickle.load(f)
    with open(prefilters_file, "rb") as f:
        prefilters = pickle.load(f)

    candidates = prefilters["candidates"]
    q_vecs = q_embeddings["embeddings"]
    questions = questions_dataset.questions
    queries = [q["question"] for q in questions]

    results = []
    for i in tqdm(range(len(queries)), desc=f"Single retriever ({split})"):
        cands = candidates[i][:prefilter_num]
        q_vec = q_vecs[i]
        retrieved = batch_retriever.query(
            queries[i],
            num_results=num_results,
            candidates=cands,
            q_vec=q_vec,
        )[0]
        results.append(retrieved)

    payload = {
        "queries": queries,
        "results": results,
        "retriever_index": single_idx,
        "docs_per_query": num_results,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

# -----------------------------------------------------------------------------
# Portfolio Selection and Artifact Audits
# -----------------------------------------------------------------------------

# Functions: compute_portfolio, compute_universal_portfolio, _pool_artifact_paths, audit_pool_artifacts
#            _load_train_scores_for_pool, compute_universal_portfolio_union

def compute_portfolio(dataset_name, retriever, num_docs_to_fetch, portfolio_size=10, device='cpu', embedder=None):
    """
        Loads recall scores and selects a greedy portfolio of retrievers.
        Saves the selected indices (and metadata) to disk.
    """
    embedder = _artifact_embedder_for_retriever(retriever, embedder)
    scores_file = C.get_retriever_scores_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
    output_file = C.get_retriever_portfolio(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)

    portfolio, portfolio_score, opt_score, topk_retrievers, topk_portfolio_score = select_portfolio(
        scores_file,
        portfolio_size=portfolio_size,
        device=device,
    )

    payload = {
        "portfolio": portfolio,
        "portfolio_size": portfolio_size,
        "portfolio_score": portfolio_score,
        "opt_retriever_per_question_score": opt_score,
        "topk_retrievers": topk_retrievers,
        "topk_portfolio_score": topk_portfolio_score,
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

def compute_universal_portfolio(
    retriever,
    num_docs_to_fetch,
    portfolio_size=10,
    device="cpu",
    embedder=None,
):
    """
        Loads recall scores for all datasets and selects a greedy
        "universal" portfolio of retrievers over the union of their questions.
        Saves the selected indices (and metadata) to disk.
    """
    embedder = _artifact_embedder_for_retriever(retriever, embedder)
    scores_tensors = []
    num_retrievers = None

    for dataset_name in C.DATASETS:
        scores_file = C.get_retriever_scores_train(dataset_name, retriever, num_docs_to_fetch, embedder=embedder)
        if not os.path.exists(scores_file):
            raise FileNotFoundError(
                f"Missing train score file for dataset={dataset_name}, "
                f"retriever={retriever}, embedder={embedder}: {scores_file}"
            )
        with open(scores_file, "rb") as f:
            scores = pickle.load(f)
        t = torch.as_tensor(scores, dtype=torch.float32, device=device)
        if t.ndim != 2:
            raise ValueError(
                f"Expected 2D train score matrix at {scores_file}, got shape={tuple(t.shape)}"
            )

        if num_retrievers is None:
            num_retrievers = t.shape[0]
        elif t.shape[0] != num_retrievers:
            raise ValueError(
                f"Mismatched retriever count for dataset {dataset_name}: "
                f"expected {num_retrievers}, got {t.shape[0]}"
            )

        scores_tensors.append(t)

    if not scores_tensors:
        raise ValueError("No scores loaded for universal portfolio computation.")

    scores_all = torch.cat(scores_tensors, dim=1)  # [R, sum_Q]

    R, Q = scores_all.shape
    k = min(portfolio_size, R)

    scores_clone = scores_all.clone()

    # --- Top-k by average (vectorized) ---
    avg_scores = scores_all.mean(dim=1)
    topk_vals, topk_idx = torch.topk(avg_scores, k=k, largest=True)
    topk_retrievers = [(int(i), float(v)) for i, v in zip(topk_idx.tolist(), topk_vals.tolist())]

    # Portfolio score if we just took top-k-by-avg
    topk_portfolio_score = torch.max(scores_clone[topk_idx], dim=0).values.sum() / Q

    # --- Greedy submodular maximization ---
    portfolio = []
    current_max = torch.zeros(Q, device=device, dtype=scores_all.dtype)

    for _ in range(k):
        marginal = torch.relu(scores_all - current_max.unsqueeze(0)).sum(dim=1)
        best = int(torch.argmax(marginal).item())
        portfolio.append(best)
        current_max = torch.maximum(current_max, scores_all[best])

    opt_retriever_per_question_score = torch.max(scores_clone, dim=0).values.sum() / Q
    portfolio_score = current_max.sum() / Q

    output_file = C.get_universal_portfolio(retriever, num_docs_to_fetch, embedder=embedder)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "portfolio": portfolio,
        "portfolio_size": k,
        "portfolio_score": float(portfolio_score),
        "opt_retriever_per_question_score": float(opt_retriever_per_question_score),
        "topk_retrievers": topk_retrievers,
        "topk_portfolio_score": float(topk_portfolio_score),
    }

    with open(output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

def _pool_artifact_paths(dataset_name, pool_spec, num_docs_to_fetch):
    spec = C.normalize_pool_spec(pool_spec)
    retriever = spec["retriever"]
    embedder = spec["artifact_embedder_key"]
    return {
        "scores_train": C.get_retriever_scores_train(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=embedder,
        ),
        "retrievals_train": C.get_retrievals_train(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=embedder,
        ),
        "retrievals_test": C.get_retrievals_test(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=embedder,
        ),
        "scores_test": C.get_retriever_scores_test(
            dataset_name,
            retriever,
            num_docs_to_fetch,
            embedder=embedder,
        ),
    }

def audit_pool_artifacts(
    datasets=None,
    num_docs_to_fetch=4,
    pool_specs=None,
    strict=False,
):
    """
        Read-only audit for full-pool train/test artifacts used by portfolio
        selection. Missing files are reported and only raise when strict=True.
    """
    datasets = list(C.DATASETS if datasets is None else datasets)
    pool_specs = (
        C.get_pool_specs_for_set(C.POOL_SET_ALL_IMPLEMENTED)
        if pool_specs is None
        else [C.normalize_pool_spec(spec) for spec in pool_specs]
    )
    rows = []

    for dataset_name in datasets:
        for spec in pool_specs:
            artifact_paths = _pool_artifact_paths(dataset_name, spec, num_docs_to_fetch)
            row = {
                "dataset": dataset_name,
                "pool_id": spec["pool_id"],
                "retriever": spec["retriever"],
                "artifact_embedder_key": spec["artifact_embedder_key"],
                "artifacts": {},
            }
            for artifact_name, expected_path in artifact_paths.items():
                exists, checked_path = _artifact_exists_for_audit(expected_path)
                row["artifacts"][artifact_name] = {
                    "path": expected_path,
                    "checked_path": checked_path,
                    "exists": exists,
                }
            rows.append(row)

    print(
        f"[artifact-audit] datasets={','.join(datasets)} "
        f"num_docs={num_docs_to_fetch} pools={','.join(spec['pool_id'] for spec in pool_specs)}",
        flush=True,
    )
    for row in rows:
        print(
            f"\n[{row['dataset']}] {row['pool_id']} "
            f"(retriever={row['retriever']}, artifact_embedder={row['artifact_embedder_key']})",
            flush=True,
        )
        for artifact_name in ["scores_train", "retrievals_train", "retrievals_test", "scores_test"]:
            artifact = row["artifacts"][artifact_name]
            status = "OK" if artifact["exists"] else "MISSING"
            print(f"  {artifact_name}: {status}", flush=True)
            print(f"    expected: {artifact['path']}", flush=True)
            if artifact["checked_path"] != artifact["path"]:
                print(f"    checked:  {artifact['checked_path']}", flush=True)

    missing = [
        {
            "dataset": row["dataset"],
            "pool_id": row["pool_id"],
            "artifact": artifact_name,
            "path": artifact["path"],
            "checked_path": artifact["checked_path"],
        }
        for row in rows
        for artifact_name, artifact in row["artifacts"].items()
        if not artifact["exists"]
    ]
    print(
        f"\n[artifact-audit] complete: checked={len(rows) * 4} missing={len(missing)}",
        flush=True,
    )
    if strict and missing:
        first = missing[0]
        raise FileNotFoundError(
            f"Missing {len(missing)} required artifacts; first missing "
            f"{first['artifact']} for dataset={first['dataset']} pool={first['pool_id']}: "
            f"{first['path']}"
        )
    return {
        "datasets": datasets,
        "num_docs": num_docs_to_fetch,
        "pool_specs": pool_specs,
        "rows": rows,
        "missing": missing,
    }

def _load_train_scores_for_pool(dataset_name, pool_spec, num_docs_to_fetch, device):
    spec = C.normalize_pool_spec(pool_spec)
    scores_file = C.get_retriever_scores_train(
        dataset_name,
        spec["retriever"],
        num_docs_to_fetch,
        embedder=spec["artifact_embedder_key"],
    )
    scores_read_path = _artifact_read_path(scores_file)
    if not os.path.exists(scores_read_path):
        raise FileNotFoundError(
            f"Missing train score file for pool={spec['pool_id']} "
            f"dataset={dataset_name}; expected: {scores_file}"
        )
    with open(scores_read_path, "rb") as f:
        scores = pickle.load(f)
    tensor = torch.as_tensor(scores, dtype=torch.float32, device=device)
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected 2D train score matrix for pool={spec['pool_id']} "
            f"dataset={dataset_name} at {scores_file}; got shape={tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) != spec["pool_size"]:
        raise ValueError(
            f"Pool-size mismatch for pool={spec['pool_id']} dataset={dataset_name} "
            f"at {scores_file}: catalog={spec['pool_size']}, scores={int(tensor.shape[0])}"
        )
    return tensor, scores_file, scores_read_path

def compute_universal_portfolio_union(
    pool_specs,
    num_docs_to_fetch,
    portfolio_size=10,
    device="cpu",
    datasets=None,
    union_name="union",
):
    """
        Compute a universal portfolio over the union of multiple retriever pools.

        pool_specs: list of dicts with keys:
            - retriever (str)
            - embedder (optional)
            - label (optional)

        Saves a single portfolio over the concatenated retriever dimension.
    """
    if datasets is None:
        datasets = C.DATASETS
    datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets provided for union portfolio.")
    if not pool_specs:
        raise ValueError("No pool specs provided for union portfolio.")
    pool_specs = [C.normalize_pool_spec(spec) for spec in pool_specs]

    pool_sizes = [spec["pool_size"] for spec in pool_specs]
    retriever_map = C.build_retriever_map_for_pools(pool_specs, pool_sizes)

    # Load scores for each dataset and concatenate across pools (retriever dimension)
    scores_tensors = []
    input_score_artifacts = []
    question_counts_by_dataset = {}
    for dataset_name in datasets:
        parts = []
        question_count = None
        for spec in pool_specs:
            tensor, scores_file, scores_read_path = _load_train_scores_for_pool(
                dataset_name,
                spec,
                num_docs_to_fetch,
                device,
            )
            if question_count is None:
                question_count = int(tensor.shape[1])
            elif int(tensor.shape[1]) != question_count:
                raise ValueError(
                    f"Question-count mismatch for dataset={dataset_name}: "
                    f"expected {question_count}, got {int(tensor.shape[1])} "
                    f"for pool={spec['pool_id']} at {scores_file}"
                )
            input_score_artifacts.append(
                {
                    "dataset": dataset_name,
                    "pool_id": spec["pool_id"],
                    "retriever": spec["retriever"],
                    "artifact_embedder_key": spec["artifact_embedder_key"],
                    "path": scores_file,
                    "loaded_path": scores_read_path,
                }
            )
            parts.append(tensor)
        scores_concat = torch.cat(parts, dim=0)
        if int(scores_concat.shape[0]) != sum(pool_sizes):
            raise ValueError(
                f"Concatenated retriever count mismatch for dataset={dataset_name}: "
                f"expected {sum(pool_sizes)}, got {int(scores_concat.shape[0])}"
            )
        question_counts_by_dataset[dataset_name] = int(scores_concat.shape[1])
        scores_tensors.append(scores_concat)

    scores_all = torch.cat(scores_tensors, dim=1)  # [R_total, sum_Q]
    R, Q = scores_all.shape
    k = min(portfolio_size, R)

    scores_clone = scores_all.clone()
    avg_scores = scores_all.mean(dim=1)
    topk_vals, topk_idx = torch.topk(avg_scores, k=k, largest=True)
    topk_retrievers = [(int(i), float(v)) for i, v in zip(topk_idx.tolist(), topk_vals.tolist())]
    topk_portfolio_score = torch.max(scores_clone[topk_idx], dim=0).values.sum() / Q

    portfolio = []
    current_max = torch.zeros(Q, device=device, dtype=scores_all.dtype)
    for _ in range(k):
        marginal = torch.relu(scores_all - current_max.unsqueeze(0)).sum(dim=1)
        best = int(torch.argmax(marginal).item())
        portfolio.append(best)
        current_max = torch.maximum(current_max, scores_all[best])

    opt_retriever_per_question_score = torch.max(scores_clone, dim=0).values.sum() / Q
    portfolio_score = current_max.sum() / Q

    output_file = C.get_universal_portfolio_union(union_name, num_docs_to_fetch)
    write_output_file = _artifact_write_path(output_file)
    Path(write_output_file).parent.mkdir(parents=True, exist_ok=True)
    if write_output_file != output_file:
        print(
            f"[portfolio-union] writing translated path: expected={output_file} "
            f"actual={write_output_file}",
            flush=True,
        )
    avg_scores_cpu = avg_scores.detach().cpu().tolist()
    selected_retrievers = []
    for rank, global_idx in enumerate(portfolio, start=1):
        member = dict(retriever_map[global_idx])
        member["rank"] = rank
        member["avg_train_score"] = float(avg_scores_cpu[global_idx])
        selected_retrievers.append(member)

    topk_retriever_metadata = []
    for rank, (global_idx, avg_score) in enumerate(topk_retrievers, start=1):
        member = dict(retriever_map[global_idx])
        member["rank"] = rank
        member["avg_train_score"] = float(avg_score)
        topk_retriever_metadata.append(member)

    payload = {
        "schema": "universal_portfolio_union_manifest",
        "schema_version": 1,
        "portfolio_id": union_name,
        "portfolio_name": union_name,
        "score_split": "train",
        "portfolio": portfolio,
        "selected_global_indices": portfolio,
        "requested_portfolio_size": int(portfolio_size),
        "portfolio_size": k,
        "actual_portfolio_size": k,
        "portfolio_score": float(portfolio_score),
        "opt_retriever_per_question_score": float(opt_retriever_per_question_score),
        "topk_retrievers": topk_retrievers,
        "topk_by_average_retriever_baseline": topk_retriever_metadata,
        "topk_portfolio_score": float(topk_portfolio_score),
        "pool_specs": pool_specs,
        "pool_sizes": pool_sizes,
        "retriever_map": retriever_map,
        "selected_retrievers": selected_retrievers,
        "datasets": datasets,
        "num_docs": int(num_docs_to_fetch),
        "total_retrievers": int(R),
        "total_questions": int(Q),
        "question_counts_by_dataset": question_counts_by_dataset,
        "input_score_artifact_paths": input_score_artifacts,
        "output_path": output_file,
        "written_path": write_output_file,
    }

    with open(write_output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

# -----------------------------------------------------------------------------
# All-Pool Portfolio Materialization
# -----------------------------------------------------------------------------

# Functions: _load_portfolio_union_manifest, _selected_retrievers_from_union_manifest, _member_pool_label
#            _member_retriever, _member_local_idx, _member_artifact_embedder_key, _member_error_context
#            _source_full_pool_test_retrieval_path, _load_source_full_pool_test_payload
#            _slice_source_member_results, _compute_materialized_recall_matrix
#            _materialize_portfolio_test_for_dataset, materialize_portfolio_test

def _load_portfolio_union_manifest(
    portfolio_path=None,
    portfolio_id=None,
    num_docs_to_fetch=4,
):
    if portfolio_path is None:
        resolved_portfolio_id = portfolio_id or C.POOL_SET_ALL_IMPLEMENTED
        portfolio_path = C.get_universal_portfolio_union_manifest(
            resolved_portfolio_id,
            num_docs_to_fetch,
        )
    else:
        resolved_portfolio_id = portfolio_id

    expected_path = str(portfolio_path)
    read_path = _artifact_read_path(expected_path)
    if not os.path.exists(read_path):
        raise FileNotFoundError(
            f"Missing portfolio union manifest: portfolio_id={resolved_portfolio_id}, "
            f"num_docs={num_docs_to_fetch}, expected_path={expected_path}, "
            f"checked_path={read_path}"
        )

    with open(read_path, "rb") as f:
        manifest = pickle.load(f)

    manifest_num_docs = manifest.get("num_docs")
    if manifest_num_docs is not None and int(manifest_num_docs) != int(num_docs_to_fetch):
        raise ValueError(
            f"Portfolio manifest num_docs mismatch at {expected_path}: "
            f"manifest={manifest_num_docs}, requested={num_docs_to_fetch}"
        )

    manifest_id = (
        manifest.get("portfolio_id")
        or manifest.get("portfolio_name")
        or resolved_portfolio_id
        or Path(expected_path).stem
    )
    return manifest, expected_path, read_path, str(manifest_id)

def _selected_retrievers_from_union_manifest(manifest):
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
                "Portfolio manifest must contain selected_retrievers or "
                "both retriever_map and portfolio."
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
    return members

def _member_pool_label(member):
    return (
        member.get("pool_id")
        or member.get("pool_label")
        or member.get("label")
        or "-"
    )

def _member_retriever(member):
    retriever = member.get("retriever") or member.get("family")
    if retriever not in {C.DS, C.VENDI, C.GRAPH_DENSE}:
        raise ValueError(
            f"Selected retriever has unsupported retriever family: {retriever!r}; "
            f"pool={_member_pool_label(member)}, local_idx={member.get('local_idx', '-')}"
        )
    return retriever

def _member_local_idx(member):
    if "local_idx" not in member:
        raise ValueError(
            f"Selected retriever is missing local_idx: pool={_member_pool_label(member)}"
        )
    local_idx = int(member["local_idx"])
    if local_idx < 0:
        raise ValueError(
            f"Selected retriever has negative local_idx={local_idx}: "
            f"pool={_member_pool_label(member)}"
        )
    return local_idx

def _member_artifact_embedder_key(member):
    retriever = _member_retriever(member)
    if retriever == C.GRAPH_DENSE:
        artifact_key = member.get("artifact_embedder_key")
        if artifact_key is None:
            return C.GRAPH_DENSE_MIXED_EMBEDDER_KEY
        artifact_key = C.normalize_embedder_key(artifact_key)
        if artifact_key != C.GRAPH_DENSE_MIXED_EMBEDDER_KEY:
            raise ValueError(
                f"graph_dense selected retriever must use artifact_embedder_key="
                f"{C.GRAPH_DENSE_MIXED_EMBEDDER_KEY!r}, got {artifact_key!r}; "
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

def _member_error_context(dataset_name, portfolio_id, member, expected_path):
    return (
        f"dataset={dataset_name}, portfolio_id={portfolio_id}, "
        f"pool={_member_pool_label(member)}, "
        f"retriever={member.get('retriever', member.get('family', '-'))}, "
        f"artifact_embedder_key={member.get('artifact_embedder_key', '-')}, "
        f"local_idx={member.get('local_idx', '-')}, "
        f"expected_path={expected_path}"
    )

def _source_full_pool_test_retrieval_path(dataset_name, member, num_docs_to_fetch):
    retriever = _member_retriever(member)
    local_idx = _member_local_idx(member)
    artifact_embedder_key = _member_artifact_embedder_key(member)
    expected_path = C.get_retrievals_test(
        dataset_name,
        retriever,
        num_docs_to_fetch,
        embedder=artifact_embedder_key,
    )
    return expected_path, retriever, artifact_embedder_key, local_idx

def _load_source_full_pool_test_payload(
    dataset_name,
    portfolio_id,
    member,
    num_docs_to_fetch,
    target_queries,
):
    expected_path, retriever, artifact_embedder_key, local_idx = (
        _source_full_pool_test_retrieval_path(
            dataset_name,
            member,
            num_docs_to_fetch,
        )
    )
    read_path = _artifact_read_path(expected_path)
    context = _member_error_context(dataset_name, portfolio_id, member, expected_path)
    if not os.path.exists(read_path):
        raise FileNotFoundError(
            f"Missing source full-pool test retrieval artifact: {context}; "
            f"checked_path={read_path}"
        )

    with open(read_path, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Source retrieval payload is not a dict: {context}")
    queries = payload.get("queries")
    results = payload.get("results")
    if not isinstance(queries, list) or not isinstance(results, list):
        raise ValueError(
            f"Source retrieval payload must contain list queries/results: {context}"
        )
    if len(queries) != len(target_queries):
        raise ValueError(
            f"Source query count mismatch: {context}; "
            f"source_queries={len(queries)}, target_test_questions={len(target_queries)}"
        )
    if len(results) != len(target_queries):
        raise ValueError(
            f"Source result count mismatch: {context}; "
            f"source_results={len(results)}, target_test_questions={len(target_queries)}"
        )
    first_mismatch = None
    for idx, (source_query, target_query) in enumerate(zip(queries, target_queries)):
        if source_query != target_query:
            first_mismatch = (idx, source_query, target_query)
            break
    if first_mismatch is not None:
        idx, source_query, target_query = first_mismatch
        raise ValueError(
            f"Source query order/text mismatch: {context}; "
            f"first_mismatch_idx={idx}, source_query={source_query!r}, "
            f"target_query={target_query!r}"
        )

    artifact = {
        "dataset": dataset_name,
        "pool_id": _member_pool_label(member),
        "retriever": retriever,
        "artifact_embedder_key": artifact_embedder_key,
        "local_idx": local_idx,
        "path": expected_path,
        "loaded_path": read_path,
    }
    return payload, artifact

def _slice_source_member_results(
    source_payload,
    source_artifact,
    member,
    dataset_name,
    portfolio_id,
    num_docs_to_fetch,
):
    expected_path = source_artifact["path"]
    local_idx = int(source_artifact["local_idx"])
    context = _member_error_context(dataset_name, portfolio_id, member, expected_path)
    sliced = []
    for q_idx, row in enumerate(source_payload["results"]):
        if row is None:
            raise ValueError(
                f"Source retrieval payload has missing/None results row: {context}; "
                f"question_idx={q_idx}"
            )
        if local_idx >= len(row):
            raise IndexError(
                f"Selected local retriever index out of range: {context}; "
                f"question_idx={q_idx}, available_retrievers={len(row)}"
            )
        retrieved_units = row[local_idx]
        if retrieved_units is None:
            raise ValueError(
                f"Source retrieval payload has None results for selected member: "
                f"{context}; question_idx={q_idx}"
            )
        sliced.append(retrieved_units)

    expected_docs = int(num_docs_to_fetch)
    short_rows = [
        idx for idx, units in enumerate(sliced)
        if hasattr(units, "__len__") and len(units) < expected_docs
    ]
    if short_rows:
        first_idx = short_rows[0]
        print(
            f"[portfolio-materialize] warning: selected member returned fewer than "
            f"{expected_docs} docs for dataset={dataset_name} portfolio_id={portfolio_id} "
            f"pool={_member_pool_label(member)} local_idx={local_idx}; "
            f"first_question_idx={first_idx} returned={len(sliced[first_idx])}",
            flush=True,
        )
    return sliced

def _compute_materialized_recall_matrix(questions_dataset, materialized_results):
    num_questions = len(materialized_results)
    portfolio_size = len(materialized_results[0]) if materialized_results else 0
    recall_matrix = [
        [0.0 for _ in range(num_questions)]
        for _ in range(portfolio_size)
    ]

    for q_idx in range(num_questions):
        gold_docs = questions_dataset.questions[q_idx]["target"]
        for rank in range(portfolio_size):
            retrieved_units = materialized_results[q_idx][rank]
            retrieved_doc_ids = [tu.doc_id for tu in retrieved_units]
            _, _, recall = f1_support(retrieved_doc_ids, gold_docs)
            recall_matrix[rank][q_idx] = recall
    return recall_matrix

def _materialize_portfolio_test_for_dataset(
    dataset_name,
    manifest,
    portfolio_path,
    portfolio_read_path,
    portfolio_id,
    selected_retrievers,
    num_docs_to_fetch,
):
    questions_dataset = _load_questions_for_split(dataset_name, "test")
    target_queries = [q["question"] for q in questions_dataset.questions]
    portfolio_size = len(selected_retrievers)
    print(
        f"[portfolio-materialize] dataset={dataset_name} portfolio_id={portfolio_id} "
        f"split=test questions={len(target_queries)} selected_members={portfolio_size}",
        flush=True,
    )

    materialized_results = [
        [None for _ in range(portfolio_size)]
        for _ in target_queries
    ]
    enriched_members = []
    source_artifacts = [None for _ in range(portfolio_size)]
    members_by_source_path = {}

    for portfolio_rank, raw_member in enumerate(selected_retrievers):
        member = dict(raw_member)
        member.setdefault("rank", portfolio_rank + 1)
        member["portfolio_rank"] = portfolio_rank
        expected_path, retriever, artifact_embedder_key, local_idx = (
            _source_full_pool_test_retrieval_path(
                dataset_name,
                member,
                num_docs_to_fetch,
            )
        )
        print(
            f"[portfolio-materialize] dataset={dataset_name} "
            f"member={portfolio_rank + 1}/{portfolio_size} "
            f"pool={_member_pool_label(member)} retriever={retriever} "
            f"artifact_embedder={artifact_embedder_key} local_idx={local_idx}",
            flush=True,
        )
        print(
            f"[portfolio-materialize] source={expected_path}",
            flush=True,
        )

        member.update(
            {
                "retriever": retriever,
                "artifact_embedder_key": artifact_embedder_key,
                "local_idx": local_idx,
                "source_retrievals_test_path": expected_path,
            }
        )
        enriched_members.append(member)
        members_by_source_path.setdefault(expected_path, []).append(
            {
                "portfolio_rank": portfolio_rank,
                "member": member,
                "retriever": retriever,
                "artifact_embedder_key": artifact_embedder_key,
                "local_idx": local_idx,
            }
        )

    for expected_path, entries in members_by_source_path.items():
        first_entry = entries[0]
        source_payload, source_artifact = _load_source_full_pool_test_payload(
            dataset_name=dataset_name,
            portfolio_id=portfolio_id,
            member=first_entry["member"],
            num_docs_to_fetch=num_docs_to_fetch,
            target_queries=target_queries,
        )
        print(
            f"[portfolio-materialize] loaded source dataset={dataset_name} "
            f"path={expected_path} selected_members={len(entries)}",
            flush=True,
        )
        for entry in entries:
            portfolio_rank = entry["portfolio_rank"]
            member = entry["member"]
            selected_results = _slice_source_member_results(
                source_payload=source_payload,
                source_artifact={
                    **source_artifact,
                    "local_idx": entry["local_idx"],
                    "retriever": entry["retriever"],
                    "artifact_embedder_key": entry["artifact_embedder_key"],
                },
                member=member,
                dataset_name=dataset_name,
                portfolio_id=portfolio_id,
                num_docs_to_fetch=num_docs_to_fetch,
            )
            for q_idx, retrieved_units in enumerate(selected_results):
                materialized_results[q_idx][portfolio_rank] = retrieved_units
            enriched_members[portfolio_rank]["source_retrievals_test_loaded_path"] = source_artifact["loaded_path"]
            source_artifacts[portfolio_rank] = {
                **source_artifact,
                "portfolio_rank": portfolio_rank,
                "rank": member["rank"],
                "global_idx": member.get("global_idx"),
                "local_idx": entry["local_idx"],
            }
        del source_payload

    bad_rows = [
        q_idx for q_idx, per_question_results in enumerate(materialized_results)
        if len(per_question_results) != portfolio_size
        or any(result is None for result in per_question_results)
    ]
    if bad_rows:
        first_idx = bad_rows[0]
        raise ValueError(
            f"Materialized result width mismatch for dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}: question_idx={first_idx}, "
            f"expected={portfolio_size}, got={len(materialized_results[first_idx])}, "
            "missing portfolio slots remain unfilled"
        )

    recall_matrix = _compute_materialized_recall_matrix(
        questions_dataset,
        materialized_results,
    )

    retrievals_path = C.get_portfolio_union_retrievals_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    scores_path = C.get_portfolio_union_scores_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    metadata_path = C.get_portfolio_union_materialization_metadata(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    retrievals_write_path = _artifact_write_path(retrievals_path)
    scores_write_path = _artifact_write_path(scores_path)
    metadata_write_path = _artifact_write_path(metadata_path)
    for output_path in [retrievals_write_path, scores_write_path, metadata_write_path]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    retrieval_payload = {
        "schema": "portfolio_union_materialized_retrievals",
        "schema_version": 1,
        "queries": target_queries,
        "results": materialized_results,
        "portfolio_id": portfolio_id,
        "portfolio_path": portfolio_path,
        "portfolio_loaded_path": portfolio_read_path,
        "portfolio_size": portfolio_size,
        "num_docs": int(num_docs_to_fetch),
        "dataset": dataset_name,
        "split": "test",
        "selected_retrievers": enriched_members,
        "source_artifacts": source_artifacts,
    }
    with open(retrievals_write_path, "wb") as f:
        pickle.dump(retrieval_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(scores_write_path, "wb") as f:
        pickle.dump(recall_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "schema": "portfolio_union_materialization_metadata",
        "schema_version": 1,
        "dataset": dataset_name,
        "split": "test",
        "portfolio_id": portfolio_id,
        "portfolio_path": portfolio_path,
        "portfolio_loaded_path": portfolio_read_path,
        "portfolio_manifest_schema": manifest.get("schema"),
        "num_docs": int(num_docs_to_fetch),
        "portfolio_size": portfolio_size,
        "num_questions": len(target_queries),
        "retrievals_path": retrievals_path,
        "retrievals_written_path": retrievals_write_path,
        "scores_path": scores_path,
        "scores_written_path": scores_write_path,
        "metadata_path": metadata_path,
        "metadata_written_path": metadata_write_path,
        "selected_retrievers": enriched_members,
        "source_artifacts": source_artifacts,
        "recall_matrix_shape": [portfolio_size, len(target_queries)],
    }
    with open(metadata_write_path, "wb") as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[portfolio-materialize] saved dataset={dataset_name} "
        f"retrievals={retrievals_path} scores={scores_path} metadata={metadata_path}",
        flush=True,
    )
    if retrievals_write_path != retrievals_path or scores_write_path != scores_path:
        print(
            f"[portfolio-materialize] written_paths dataset={dataset_name} "
            f"retrievals={retrievals_write_path} scores={scores_write_path} "
            f"metadata={metadata_write_path}",
            flush=True,
        )

    return {
        "dataset": dataset_name,
        "retrievals_path": retrievals_path,
        "retrievals_written_path": retrievals_write_path,
        "scores_path": scores_path,
        "scores_written_path": scores_write_path,
        "metadata_path": metadata_path,
        "metadata_written_path": metadata_write_path,
        "portfolio_size": portfolio_size,
        "num_questions": len(target_queries),
        "recall_matrix_shape": [portfolio_size, len(target_queries)],
        "source_artifacts": source_artifacts,
    }

def materialize_portfolio_test(
    portfolio_path=None,
    portfolio_id=None,
    datasets=None,
    num_docs_to_fetch=4,
    strict=True,
):
    """
    Materialize a saved all-pool portfolio manifest on the test split.

    This function does not select or reorder retrievers. It preserves the
    manifest portfolio order, slices each selected local retriever from its
    full-pool test retrieval payload, writes a materialized retrieval payload
    shaped [question_idx][portfolio_rank], and writes a plain recall matrix
    shaped [portfolio_rank][question_idx] plus a metadata sidecar.
    """
    manifest, manifest_path, manifest_read_path, resolved_portfolio_id = (
        _load_portfolio_union_manifest(
            portfolio_path=portfolio_path,
            portfolio_id=portfolio_id,
            num_docs_to_fetch=num_docs_to_fetch,
        )
    )
    selected_retrievers = _selected_retrievers_from_union_manifest(manifest)
    if not selected_retrievers:
        raise ValueError(
            f"Portfolio manifest has no selected retrievers: {manifest_path}"
        )

    datasets = list(C.DATASETS if datasets is None else datasets)
    if not datasets:
        raise ValueError("No datasets provided for portfolio materialization.")

    print(
        f"[portfolio-materialize] manifest={manifest_path} "
        f"loaded_manifest={manifest_read_path} portfolio_id={resolved_portfolio_id} "
        f"num_docs={num_docs_to_fetch} datasets={','.join(datasets)}",
        flush=True,
    )

    materialized = []
    failures = []
    for dataset_name in datasets:
        try:
            materialized.append(
                _materialize_portfolio_test_for_dataset(
                    dataset_name=dataset_name,
                    manifest=manifest,
                    portfolio_path=manifest_path,
                    portfolio_read_path=manifest_read_path,
                    portfolio_id=resolved_portfolio_id,
                    selected_retrievers=selected_retrievers,
                    num_docs_to_fetch=num_docs_to_fetch,
                )
            )
        except Exception as exc:
            if strict:
                raise
            failure = {
                "dataset": dataset_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[portfolio-materialize] failed dataset={dataset_name} "
                f"error_type={failure['error_type']} error={failure['error']}",
                flush=True,
            )

    return {
        "portfolio_id": resolved_portfolio_id,
        "portfolio_path": manifest_path,
        "portfolio_loaded_path": manifest_read_path,
        "num_docs": int(num_docs_to_fetch),
        "datasets_requested": datasets,
        "datasets_materialized": [item["dataset"] for item in materialized],
        "materialized": materialized,
        "failures": failures,
    }


_FAMILY_BEST_ALLOWED_FAMILIES = (C.DS, C.VENDI, C.GRAPH_DENSE)

# -----------------------------------------------------------------------------
# Family-Best Baselines
# -----------------------------------------------------------------------------

# Functions: _normalize_family_list, _family_best_pool_specs, _member_parameters
#            select_family_best_retrievers, _load_family_best_manifest, _selected_family_member_from_manifest
#            _load_pickle_artifact, _validate_query_payload, _compute_family_best_score_matrices
#            _write_family_best_outputs, _family_best_output_paths, _family_best_checkpoint_path
#            _completed_family_best_outputs_if_available, _family_best_checkpoint_metadata
#            _load_family_best_checkpoint, _write_family_best_checkpoint, _compute_family_best_dense_results
#            _compute_family_best_graph_dense_results, compute_family_best_test_retrievals

def _normalize_family_list(families):
    if families is None:
        return list(_FAMILY_BEST_ALLOWED_FAMILIES)
    normalized = []
    for family in families:
        if family not in _FAMILY_BEST_ALLOWED_FAMILIES:
            raise ValueError(
                f"Unsupported family for family-best baselines: {family!r}. "
                f"Allowed: {', '.join(_FAMILY_BEST_ALLOWED_FAMILIES)}"
            )
        if family not in normalized:
            normalized.append(family)
    if not normalized:
        raise ValueError("No families provided for family-best baselines.")
    return normalized

def _family_best_pool_specs(portfolio_id, families):
    pool_set = portfolio_id if portfolio_id in C.POOL_SETS else C.POOL_SET_ALL_IMPLEMENTED
    pool_specs = C.get_pool_specs_for_set(pool_set)
    by_family = {}
    for family in families:
        specs = [
            spec for spec in pool_specs
            if spec["family"] == family or spec["retriever"] == family
        ]
        if not specs:
            raise ValueError(
                f"No pool specs found for family={family!r} in pool_set={pool_set!r}."
            )
        by_family[family] = specs
    return by_family, pool_set

def _member_parameters(member):
    retriever = member.get("retriever") or member.get("family")
    if retriever == C.DS:
        return {"gamma": member["gamma"], "r": member["r"]}
    if retriever == C.VENDI:
        return {"s": member["s"]}
    if retriever == C.GRAPH_DENSE:
        return dict(member.get("parameters") or {
            "name": member.get("name"),
            "embedder": member.get("embedder"),
            "max_hops": member.get("max_hops"),
            "max_entity_df": member.get("max_entity_df"),
            "max_candidates": member.get("max_candidates"),
        })
    raise ValueError(f"Unsupported retriever family for member parameters: {retriever!r}")

def select_family_best_retrievers(
    portfolio_id=C.POOL_SET_ALL_IMPLEMENTED,
    datasets=C.DATASETS,
    num_docs_to_fetch=4,
    families=(C.DS, C.VENDI, C.GRAPH_DENSE),
    device="cpu",
):
    """
    Select the best single retriever per family using train recall only.

    The score for each candidate is its mean recall over the concatenation of
    all requested datasets and train questions. The saved manifest is the
    source of truth for compute_family_best_test_retrievals.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets provided for family-best selection.")
    families = _normalize_family_list(families)
    specs_by_family, source_pool_set = _family_best_pool_specs(portfolio_id, families)

    selected_by_family = {}
    candidates_by_family = {}
    pool_specs_used = []
    for specs in specs_by_family.values():
        for spec in specs:
            if not any(existing["pool_id"] == spec["pool_id"] for existing in pool_specs_used):
                pool_specs_used.append(spec)

    print(
        f"[family-best-select] portfolio_id={portfolio_id} source_pool_set={source_pool_set} "
        f"datasets={','.join(datasets)} families={','.join(families)} "
        f"num_docs={num_docs_to_fetch}",
        flush=True,
    )

    for family in families:
        candidates = []
        candidate_order = 0
        for spec in specs_by_family[family]:
            score_sum = None
            question_count = 0
            source_train_score_paths = []

            for dataset_name in datasets:
                tensor, scores_file, scores_read_path = _load_train_scores_for_pool(
                    dataset_name,
                    spec,
                    num_docs_to_fetch,
                    device,
                )
                per_retriever_sum = tensor.sum(dim=1).detach().cpu()
                score_sum = per_retriever_sum if score_sum is None else score_sum + per_retriever_sum
                question_count += int(tensor.shape[1])
                source_train_score_paths.append(
                    {
                        "dataset": dataset_name,
                        "path": scores_file,
                        "loaded_path": scores_read_path,
                        "num_questions": int(tensor.shape[1]),
                    }
                )

            if question_count <= 0:
                raise ValueError(
                    f"No train questions found while selecting family={family} "
                    f"pool={spec['pool_id']}."
                )
            average_scores = (score_sum / question_count).tolist()
            for local_idx, average_score in enumerate(average_scores):
                member = C.describe_pool_member(spec, local_idx)
                member_parameters = _member_parameters(member)
                candidate = {
                    **member,
                    "member_parameters": member_parameters,
                    "average_train_score": float(average_score),
                    "avg_train_score": float(average_score),
                    "train_score_num_questions": int(question_count),
                    "source_train_score_paths": source_train_score_paths,
                    "candidate_order": candidate_order,
                }
                candidates.append(candidate)
                candidate_order += 1

        if not candidates:
            raise ValueError(f"No family-best candidates found for family={family}.")
        best = sorted(
            candidates,
            key=lambda item: (-item["average_train_score"], item["candidate_order"]),
        )[0]
        selected_by_family[family] = best
        candidates_by_family[family] = candidates
        print(
            f"[family-best-select] selected family={family} pool={best['pool_id']} "
            f"local_idx={best['local_idx']} avg_train_score={best['average_train_score']:.6f}",
            flush=True,
        )

    output_file = C.get_family_best_baseline_manifest(portfolio_id, num_docs_to_fetch)
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)
    if output_write_path != output_file:
        print(
            f"[family-best-select] writing translated path: expected={output_file} "
            f"actual={output_write_path}",
            flush=True,
        )

    manifest = {
        "schema": "family_best_baselines_manifest",
        "schema_version": 1,
        "portfolio_id": portfolio_id,
        "source_pool_set": source_pool_set,
        "score_split": "train",
        "datasets": datasets,
        "num_docs": int(num_docs_to_fetch),
        "families": families,
        "pool_specs": pool_specs_used,
        "selected_by_family": selected_by_family,
        "selected_retrievers": [selected_by_family[family] for family in families],
        "candidates_by_family": candidates_by_family,
        "output_path": output_file,
        "written_path": output_write_path,
    }
    with open(output_write_path, "wb") as f:
        pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[family-best-select] saved manifest={output_file}", flush=True)
    return output_file

def _load_family_best_manifest(portfolio_id, num_docs_to_fetch):
    expected_path = C.get_family_best_baseline_manifest(portfolio_id, num_docs_to_fetch)
    read_path = _artifact_read_path(expected_path)
    if not os.path.exists(read_path):
        raise FileNotFoundError(
            f"Missing family-best baseline manifest: portfolio_id={portfolio_id}, "
            f"num_docs={num_docs_to_fetch}, expected_path={expected_path}, "
            f"checked_path={read_path}. Run select-family-best-baselines first."
        )
    with open(read_path, "rb") as f:
        manifest = pickle.load(f)
    if manifest.get("schema") != "family_best_baselines_manifest":
        raise ValueError(
            f"Unexpected family-best manifest schema at {expected_path}: "
            f"{manifest.get('schema')!r}"
        )
    manifest_num_docs = int(manifest.get("num_docs", num_docs_to_fetch))
    if manifest_num_docs != int(num_docs_to_fetch):
        raise ValueError(
            f"Family-best manifest num_docs mismatch at {expected_path}: "
            f"manifest={manifest_num_docs}, requested={num_docs_to_fetch}"
        )
    return manifest, expected_path, read_path

def _selected_family_member_from_manifest(manifest, family):
    selected = manifest.get("selected_by_family")
    if not isinstance(selected, dict) or family not in selected:
        raise ValueError(
            f"Family-best manifest does not contain selected member for family={family!r}. "
            f"Available families: {sorted(selected) if isinstance(selected, dict) else '<invalid>'}"
        )
    member = dict(selected[family])
    member["family"] = family
    member.setdefault("retriever", family)
    member.setdefault("pool_id", member.get("pool_label", family))
    if "artifact_embedder_key" not in member:
        raise ValueError(
            f"Family-best selected member is missing artifact_embedder_key: "
            f"family={family}, pool={_member_pool_label(member)}, "
            f"local_idx={member.get('local_idx', '-')}. "
            "Rebuild the manifest with select-family-best-baselines."
        )
    return member

def _load_pickle_artifact(expected_path, *, purpose):
    read_path = _artifact_read_path(expected_path)
    if not os.path.exists(read_path):
        raise FileNotFoundError(
            f"Missing {purpose}: expected_path={expected_path}, checked_path={read_path}"
        )
    with open(read_path, "rb") as f:
        return pickle.load(f), read_path

def _validate_query_payload(payload, expected_queries, *, purpose, path):
    queries = payload.get("queries") if isinstance(payload, dict) else None
    if queries != expected_queries:
        if not isinstance(queries, list):
            detail = f"queries_type={type(queries).__name__}"
        elif len(queries) != len(expected_queries):
            detail = f"queries={len(queries)}, expected={len(expected_queries)}"
        else:
            first_mismatch = next(
                (
                    idx for idx, (actual, expected) in enumerate(zip(queries, expected_queries))
                    if actual != expected
                ),
                None,
            )
            detail = f"first_mismatch_idx={first_mismatch}"
        raise ValueError(
            f"Query order mismatch for {purpose} at {path}: {detail}. "
            "Recompute the artifact for the current question split/order."
        )

def _compute_family_best_score_matrices(
    questions_dataset,
    retrieval_results,
    num_docs_to_fetch,
    max_k,
    *,
    desc,
):
    num_questions = len(questions_dataset.questions)
    recall_matrix = [[0.0 for _ in range(num_questions)] for _ in range(max_k)]
    f1_matrix = [[0.0 for _ in range(num_questions)] for _ in range(max_k)]

    if len(retrieval_results) != num_questions:
        raise ValueError(
            f"Family-best retrieval result count mismatch: "
            f"results={len(retrieval_results)}, questions={num_questions}"
        )

    for q_idx in tqdm(range(num_questions), desc=desc):
        gold_docs = questions_dataset.questions[q_idx]["target"]
        retrieved_units = retrieval_results[q_idx]
        if retrieved_units is None:
            raise ValueError(f"Missing family-best retrieval row at question_idx={q_idx}")
        for k_idx in range(max_k):
            limit = (k_idx + 1) * int(num_docs_to_fetch)
            retrieved_doc_ids = [tu.doc_id for tu in retrieved_units[:limit]]
            f1, _precision, recall = f1_support(retrieved_doc_ids, gold_docs)
            recall_matrix[k_idx][q_idx] = recall
            f1_matrix[k_idx][q_idx] = f1

    return recall_matrix, f1_matrix

def _write_family_best_outputs(
    portfolio_id,
    dataset_name,
    family,
    num_docs_to_fetch,
    max_k,
    retrieval_payload,
    recall_matrix,
    f1_matrix,
):
    retrievals_path = C.get_family_best_retrievals_test(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    recall_scores_path = C.get_family_best_scores_test(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    f1_scores_path = C.get_family_best_scores_test_f1(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    retrievals_write_path = _artifact_write_path(retrievals_path)
    recall_write_path = _artifact_write_path(recall_scores_path)
    f1_write_path = _artifact_write_path(f1_scores_path)
    for output_path in [retrievals_write_path, recall_write_path, f1_write_path]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(retrievals_write_path, "wb") as f:
        pickle.dump(retrieval_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(recall_write_path, "wb") as f:
        pickle.dump(recall_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f1_write_path, "wb") as f:
        pickle.dump(f1_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[family-best-test] saved dataset={dataset_name} family={family} "
        f"retrievals={retrievals_path} recall_scores={recall_scores_path} "
        f"f1_scores={f1_scores_path}",
        flush=True,
    )
    if (
        retrievals_write_path != retrievals_path
        or recall_write_path != recall_scores_path
        or f1_write_path != f1_scores_path
    ):
        print(
            f"[family-best-test] written_paths dataset={dataset_name} family={family} "
            f"retrievals={retrievals_write_path} recall_scores={recall_write_path} "
            f"f1_scores={f1_write_path}",
            flush=True,
        )

    return {
        "retrievals_path": retrievals_path,
        "retrievals_written_path": retrievals_write_path,
        "scores_path": recall_scores_path,
        "scores_written_path": recall_write_path,
        "scores_f1_path": f1_scores_path,
        "scores_f1_written_path": f1_write_path,
        "score_shape": [int(max_k), len(retrieval_payload["queries"])],
    }

def _family_best_output_paths(portfolio_id, dataset_name, family, num_docs_to_fetch, max_k):
    return {
        "retrievals": C.get_family_best_retrievals_test(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
        ),
        "scores": C.get_family_best_scores_test(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
        ),
        "scores_f1": C.get_family_best_scores_test_f1(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
        ),
    }

def _family_best_checkpoint_path(portfolio_id, dataset_name, family, num_docs_to_fetch, max_k):
    return (
        C.get_family_best_retrievals_test(
            portfolio_id,
            dataset_name,
            family,
            num_docs_to_fetch,
            max_k,
        )
        + ".checkpoint"
    )

def _completed_family_best_outputs_if_available(
    portfolio_id,
    dataset_name,
    family,
    num_docs_to_fetch,
    max_k,
    expected_queries,
):
    paths = _family_best_output_paths(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    read_paths = {name: _artifact_read_path(path) for name, path in paths.items()}
    if not all(os.path.exists(path) for path in read_paths.values()):
        return None

    with open(read_paths["retrievals"], "rb") as f:
        retrieval_payload = pickle.load(f)
    _validate_query_payload(
        retrieval_payload,
        expected_queries,
        purpose="completed family-best retrievals",
        path=read_paths["retrievals"],
    )
    results = retrieval_payload.get("results")
    if not isinstance(results, list) or len(results) != len(expected_queries):
        return None
    if any(row is None for row in results):
        return None

    with open(read_paths["scores"], "rb") as f:
        recall_matrix = pickle.load(f)
    with open(read_paths["scores_f1"], "rb") as f:
        f1_matrix = pickle.load(f)
    expected_shape = [int(max_k), len(expected_queries)]
    actual_recall_shape = [
        len(recall_matrix) if isinstance(recall_matrix, list) else -1,
        len(recall_matrix[0]) if isinstance(recall_matrix, list) and recall_matrix else 0,
    ]
    actual_f1_shape = [
        len(f1_matrix) if isinstance(f1_matrix, list) else -1,
        len(f1_matrix[0]) if isinstance(f1_matrix, list) and f1_matrix else 0,
    ]
    if actual_recall_shape != expected_shape or actual_f1_shape != expected_shape:
        return None

    print(
        f"[family-best-test] skipping completed dataset={dataset_name} family={family} "
        f"retrievals={paths['retrievals']} scores={paths['scores']} "
        f"scores_f1={paths['scores_f1']}",
        flush=True,
    )
    return {
        "dataset": dataset_name,
        "family": family,
        "retrievals_path": paths["retrievals"],
        "retrievals_written_path": read_paths["retrievals"],
        "scores_path": paths["scores"],
        "scores_written_path": read_paths["scores"],
        "scores_f1_path": paths["scores_f1"],
        "scores_f1_written_path": read_paths["scores_f1"],
        "score_shape": expected_shape,
        "skipped_existing": True,
    }

def _family_best_checkpoint_metadata(
    portfolio_id,
    dataset_name,
    family,
    member,
    num_docs_to_fetch,
    max_k,
):
    return {
        "schema": "family_best_baseline_retrievals_checkpoint",
        "schema_version": 1,
        "portfolio_id": portfolio_id,
        "dataset": dataset_name,
        "family": family,
        "split": "test",
        "num_docs": int(num_docs_to_fetch),
        "max_k": int(max_k),
        "docs_per_query": int(num_docs_to_fetch) * int(max_k),
        "pool_id": _member_pool_label(member),
        "retriever": _member_retriever(member),
        "artifact_embedder_key": _member_artifact_embedder_key(member),
        "local_idx": _member_local_idx(member),
    }

def _load_family_best_checkpoint(checkpoint_path, expected_queries, metadata):
    checkpoint_read_path = _artifact_read_path(checkpoint_path)
    if not os.path.exists(checkpoint_read_path):
        return [None for _ in expected_queries], 0, checkpoint_read_path

    with open(checkpoint_read_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Family-best checkpoint must be a dict: {checkpoint_path}")
    for key, expected_value in metadata.items():
        if payload.get(key) != expected_value:
            raise ValueError(
                f"Family-best checkpoint metadata mismatch at {checkpoint_path}: "
                f"{key}={payload.get(key)!r}, expected={expected_value!r}"
            )
    _validate_query_payload(
        payload,
        expected_queries,
        purpose="family-best retrieval checkpoint",
        path=checkpoint_read_path,
    )
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(expected_queries):
        raise ValueError(
            f"Family-best checkpoint result length mismatch at {checkpoint_path}: "
            f"results={len(results) if isinstance(results, list) else type(results).__name__}, "
            f"expected={len(expected_queries)}"
        )
    completed = sum(row is not None for row in results)
    print(
        f"[family-best-test] resuming checkpoint={checkpoint_path} "
        f"loaded_path={checkpoint_read_path} completed={completed}/{len(expected_queries)}",
        flush=True,
    )
    return results, completed, checkpoint_read_path

def _write_family_best_checkpoint(
    checkpoint_path,
    queries,
    results,
    metadata,
    source_artifacts,
    *,
    label,
):
    checkpoint_write_path = _artifact_write_path(checkpoint_path)
    Path(checkpoint_write_path).parent.mkdir(parents=True, exist_ok=True)
    completed = sum(row is not None for row in results)
    payload = {
        **metadata,
        "queries": queries,
        "results": results,
        "source_artifacts": source_artifacts,
        "completed": completed,
        "num_questions": len(queries),
    }
    tmp_path = f"{checkpoint_write_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, checkpoint_write_path)
    print(
        f"[family-best-test] {label} checkpoint saved: "
        f"{completed}/{len(queries)} path={checkpoint_path}",
        flush=True,
    )

def _compute_family_best_dense_results(
    dataset_name,
    member,
    questions_dataset,
    queries,
    num_docs_to_fetch,
    max_k,
    device,
    prefilter_num,
    resume_results=None,
    checkpoint_path=None,
    checkpoint_metadata=None,
    save_every=100,
):
    retriever = _member_retriever(member)
    artifact_embedder_key = _member_artifact_embedder_key(member)
    num_results = int(max_k) * int(num_docs_to_fetch)
    if int(prefilter_num) < num_results:
        raise ValueError(
            f"prefilter_num={prefilter_num} is smaller than required num_results={num_results} "
            f"for dataset={dataset_name} family={member.get('family')}."
        )

    vector_db_dir = C.get_vector_db_dir(dataset_name, embedder=artifact_embedder_key)
    vector_db_read_dir = _artifact_read_path(vector_db_dir)
    if not _faiss_index_exists(vector_db_read_dir):
        raise FileNotFoundError(
            f"Missing FAISS vector DB for family-best dense retrieval: "
            f"dataset={dataset_name}, retriever={retriever}, "
            f"artifact_embedder_key={artifact_embedder_key}, expected_path={vector_db_dir}, "
            f"checked_path={vector_db_read_dir}"
        )
    vector_db = FaissVectorDB.load(vector_db_read_dir)
    embedder = Embedder(device, embedder=artifact_embedder_key)

    embeddings_path = C.get_embeddings_test(dataset_name, embedder=artifact_embedder_key)
    q_embeddings, embeddings_read_path = _load_pickle_artifact(
        embeddings_path,
        purpose=(
            f"test question embeddings for dataset={dataset_name} "
            f"embedder={artifact_embedder_key}"
        ),
    )
    _validate_query_payload(
        q_embeddings,
        queries,
        purpose="family-best test question embeddings",
        path=embeddings_read_path,
    )

    prefilters_path = C.get_prefilters_test(dataset_name, embedder=artifact_embedder_key)
    prefilters, prefilters_read_path = _load_pickle_artifact(
        prefilters_path,
        purpose=(
            f"test prefilters for dataset={dataset_name} "
            f"embedder={artifact_embedder_key}"
        ),
    )
    _validate_query_payload(
        prefilters,
        queries,
        purpose="family-best test prefilters",
        path=prefilters_read_path,
    )
    if len(q_embeddings.get("embeddings", [])) != len(queries):
        raise ValueError(
            f"Question embedding count mismatch at {embeddings_path}: "
            f"embeddings={len(q_embeddings.get('embeddings', []))}, queries={len(queries)}"
        )
    if len(prefilters.get("candidates", [])) != len(queries):
        raise ValueError(
            f"Prefilter count mismatch at {prefilters_path}: "
            f"candidates={len(prefilters.get('candidates', []))}, queries={len(queries)}"
        )

    if retriever == C.DS:
        gamma = float(member["gamma"])
        r = float(member["r"])
        batch_retriever = BatchDiscountedSimilarity(
            [DiscountedSimilarity(embedder, vector_db, gamma, r, metric="dot", device=device)],
            device,
        )
    elif retriever == C.VENDI:
        s = float(member["s"])
        batch_retriever = BatchVendiRetriever(
            [VendiRetriever(embedder, vector_db, s, device=device)],
            device,
        )
    else:
        raise ValueError(f"Unsupported dense family-best retriever: {retriever}")

    source_artifacts = {
        "vector_db_path": vector_db_dir,
        "vector_db_loaded_path": vector_db_read_dir,
        "embeddings_path": embeddings_path,
        "embeddings_loaded_path": embeddings_read_path,
        "prefilters_path": prefilters_path,
        "prefilters_loaded_path": prefilters_read_path,
    }

    results = resume_results if resume_results is not None else [None for _ in queries]
    missing_indices = [idx for idx, row in enumerate(results) if row is None]
    print(
        f"[family-best-test] dense retrieval progress dataset={dataset_name} "
        f"family={member.get('family', retriever)} completed={len(queries) - len(missing_indices)}/"
        f"{len(queries)}",
        flush=True,
    )
    completed_since_checkpoint = 0
    for q_idx in tqdm(
        missing_indices,
        desc=f"Family-best {dataset_name} {member.get('family', retriever)}",
    ):
        candidates = prefilters["candidates"][q_idx][:prefilter_num]
        if len(candidates) < num_results:
            raise ValueError(
                f"Not enough prefilter candidates for family-best retrieval: "
                f"dataset={dataset_name}, family={member.get('family')}, "
                f"question_idx={q_idx}, required={num_results}, available={len(candidates)}"
            )
        retrieved = batch_retriever.query(
            queries[q_idx],
            num_results=num_results,
            candidates=candidates,
            q_vec=q_embeddings["embeddings"][q_idx],
        )[0]
        results[q_idx] = retrieved
        completed_since_checkpoint += 1
        if (
            checkpoint_path is not None
            and save_every
            and completed_since_checkpoint >= int(save_every)
        ):
            _write_family_best_checkpoint(
                checkpoint_path,
                queries,
                results,
                checkpoint_metadata,
                source_artifacts,
                label="Periodic",
            )
            completed_since_checkpoint = 0

    if checkpoint_path is not None:
        _write_family_best_checkpoint(
            checkpoint_path,
            queries,
            results,
            checkpoint_metadata,
            source_artifacts,
            label="Final retrieval",
        )

    return results, source_artifacts

def _compute_family_best_graph_dense_results(
    dataset_name,
    member,
    questions_dataset,
    queries,
    num_docs_to_fetch,
    max_k,
    device,
    resume_results=None,
    checkpoint_path=None,
    checkpoint_metadata=None,
    save_every=100,
):
    params = dict(member.get("parameters") or _member_parameters(member))
    selected_embedder_key = C.normalize_embedder_key(params.get("embedder"))
    if selected_embedder_key == C.GRAPH_DENSE_MIXED_EMBEDDER_KEY:
        raise ValueError(
            f"Family-best graph_dense member has invalid query embedder 'mixed': {params}"
        )
    num_results = int(max_k) * int(num_docs_to_fetch)
    batch_retriever, query_entity_cache, graph_index_path = _load_graph_dense_batch_retriever(
        dataset_name=dataset_name,
        split="test",
        device=device,
        retriever_params=[params],
    )
    q_embeddings_by_key = _load_q_embeddings_by_embedder_if_available(
        dataset_name,
        "test",
        [selected_embedder_key],
        queries,
    )

    source_artifacts = {
        "graph_index_path": graph_index_path,
        "query_entity_cache_path": query_entity_cache["path"],
        "query_embedder_key": selected_embedder_key,
        "query_embeddings_loaded": q_embeddings_by_key is not None,
    }

    results = resume_results if resume_results is not None else [None for _ in queries]
    missing_indices = [idx for idx, row in enumerate(results) if row is None]
    print(
        f"[family-best-test] graph_dense retrieval progress dataset={dataset_name} "
        f"completed={len(queries) - len(missing_indices)}/{len(queries)}",
        flush=True,
    )
    completed_since_checkpoint = 0
    for q_idx in tqdm(missing_indices, desc=f"Family-best {dataset_name} graph_dense"):
        if isinstance(q_embeddings_by_key, dict):
            q_vec = {selected_embedder_key: q_embeddings_by_key[selected_embedder_key][q_idx]}
        else:
            q_vec = None
        retrieved = batch_retriever.query(
            queries[q_idx],
            num_results=num_results,
            candidates=None,
            q_vec=q_vec,
            query_idx=q_idx,
        )[0]
        results[q_idx] = retrieved
        completed_since_checkpoint += 1
        if (
            checkpoint_path is not None
            and save_every
            and completed_since_checkpoint >= int(save_every)
        ):
            _write_family_best_checkpoint(
                checkpoint_path,
                queries,
                results,
                checkpoint_metadata,
                source_artifacts,
                label="Periodic",
            )
            completed_since_checkpoint = 0

    if checkpoint_path is not None:
        _write_family_best_checkpoint(
            checkpoint_path,
            queries,
            results,
            checkpoint_metadata,
            source_artifacts,
            label="Final retrieval",
        )

    return results, source_artifacts

def compute_family_best_test_retrievals(
    portfolio_id=C.POOL_SET_ALL_IMPLEMENTED,
    datasets=C.DATASETS,
    num_docs_to_fetch=4,
    max_k=5,
    families=(C.DS, C.VENDI, C.GRAPH_DENSE),
    device="cuda",
    prefilter_num=1000,
    save_every=100,
):
    """
    Compute test retrievals and recall/F1-support score matrices for the
    train-selected best single retriever in each family.
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("No datasets provided for family-best test retrievals.")
    families = _normalize_family_list(families)
    if int(max_k) <= 0:
        raise ValueError(f"max_k must be positive, got {max_k}")
    if int(num_docs_to_fetch) <= 0:
        raise ValueError(f"num_docs_to_fetch must be positive, got {num_docs_to_fetch}")
    if int(save_every) < 0:
        raise ValueError(f"save_every must be non-negative, got {save_every}")

    manifest, manifest_path, manifest_read_path = _load_family_best_manifest(
        portfolio_id,
        num_docs_to_fetch,
    )
    summary = {
        "portfolio_id": portfolio_id,
        "manifest_path": manifest_path,
        "manifest_loaded_path": manifest_read_path,
        "num_docs": int(num_docs_to_fetch),
        "max_k": int(max_k),
        "save_every": int(save_every),
        "datasets": datasets,
        "families": families,
        "outputs": [],
    }
    print(
        f"[family-best-test] manifest={manifest_path} loaded_manifest={manifest_read_path} "
        f"datasets={','.join(datasets)} families={','.join(families)} "
        f"num_docs={num_docs_to_fetch} max_k={max_k} device={device}",
        flush=True,
    )

    for dataset_name in datasets:
        questions_dataset = _load_questions_for_split(dataset_name, "test")
        queries = [q["question"] for q in questions_dataset.questions]
        for family in families:
            completed_output = _completed_family_best_outputs_if_available(
                portfolio_id=portfolio_id,
                dataset_name=dataset_name,
                family=family,
                num_docs_to_fetch=num_docs_to_fetch,
                max_k=max_k,
                expected_queries=queries,
            )
            if completed_output is not None:
                summary["outputs"].append(completed_output)
                continue

            member = _selected_family_member_from_manifest(manifest, family)
            retriever = _member_retriever(member)
            artifact_embedder_key = _member_artifact_embedder_key(member)
            local_idx = _member_local_idx(member)
            checkpoint_path = _family_best_checkpoint_path(
                portfolio_id,
                dataset_name,
                family,
                num_docs_to_fetch,
                max_k,
            )
            checkpoint_metadata = _family_best_checkpoint_metadata(
                portfolio_id,
                dataset_name,
                family,
                member,
                num_docs_to_fetch,
                max_k,
            )
            resume_results, _completed, _checkpoint_read_path = _load_family_best_checkpoint(
                checkpoint_path,
                queries,
                checkpoint_metadata,
            )
            print(
                f"[family-best-test] dataset={dataset_name} family={family} "
                f"pool={_member_pool_label(member)} retriever={retriever} "
                f"artifact_embedder={artifact_embedder_key} local_idx={local_idx} "
                f"checkpoint={checkpoint_path}",
                flush=True,
            )

            if retriever in {C.DS, C.VENDI}:
                results, source_artifacts = _compute_family_best_dense_results(
                    dataset_name=dataset_name,
                    member=member,
                    questions_dataset=questions_dataset,
                    queries=queries,
                    num_docs_to_fetch=num_docs_to_fetch,
                    max_k=max_k,
                    device=device,
                    prefilter_num=prefilter_num,
                    resume_results=resume_results,
                    checkpoint_path=checkpoint_path,
                    checkpoint_metadata=checkpoint_metadata,
                    save_every=save_every,
                )
            elif retriever == C.GRAPH_DENSE:
                results, source_artifacts = _compute_family_best_graph_dense_results(
                    dataset_name=dataset_name,
                    member=member,
                    questions_dataset=questions_dataset,
                    queries=queries,
                    num_docs_to_fetch=num_docs_to_fetch,
                    max_k=max_k,
                    device=device,
                    resume_results=resume_results,
                    checkpoint_path=checkpoint_path,
                    checkpoint_metadata=checkpoint_metadata,
                    save_every=save_every,
                )
            else:
                raise ValueError(f"Unsupported family-best retriever: {retriever}")

            recall_matrix, f1_matrix = _compute_family_best_score_matrices(
                questions_dataset,
                results,
                num_docs_to_fetch,
                int(max_k),
                desc=f"Family-best scores {dataset_name} {family}",
            )
            retrieval_payload = {
                "schema": "family_best_baseline_retrievals_test",
                "schema_version": 1,
                "portfolio_id": portfolio_id,
                "family": family,
                "dataset": dataset_name,
                "split": "test",
                "num_docs": int(num_docs_to_fetch),
                "max_k": int(max_k),
                "docs_per_query": int(max_k) * int(num_docs_to_fetch),
                "prefilter_num": int(prefilter_num),
                "queries": queries,
                "results": results,
                "selected_retriever": member,
                "manifest_path": manifest_path,
                "manifest_loaded_path": manifest_read_path,
                "source_artifacts": source_artifacts,
                "score_shape": [int(max_k), len(queries)],
            }
            output_info = _write_family_best_outputs(
                portfolio_id=portfolio_id,
                dataset_name=dataset_name,
                family=family,
                num_docs_to_fetch=num_docs_to_fetch,
                max_k=max_k,
                retrieval_payload=retrieval_payload,
                recall_matrix=recall_matrix,
                f1_matrix=f1_matrix,
            )
            output_info.update(
                {
                    "dataset": dataset_name,
                    "family": family,
                    "selected_retriever": member,
                }
            )
            summary["outputs"].append(output_info)

    return summary

# -----------------------------------------------------------------------------
# Prompt Builders
# -----------------------------------------------------------------------------

# Functions: build_answer_prompts, _retrieved_unit_text, build_portfolio_union_answer_prompts
#            build_selector_prompts, _load_pickle_artifact, build_family_best_answer_prompts
#            _portfolio_router_judge_context, _validate_router_judge_common, _validate_optional_payload_field
#            _validate_optional_int_payload_field, _coerce_router_judge_prediction_payload
#            _answer_rank_from_record, _portfolio_union_answers_to_matrix, _normalize_portfolio_answer_record
#            _response_from_portfolio_answer_record, build_portfolio_router_judge_prompts
#            build_baseline_answer_prompts

def build_answer_prompts(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    portfolio_size=10,
    answer_prompt_fn=answer_prompt,
    universal: bool = False,
    embedder=None,
):
    """
        Prepares one answer prompt per selected portfolio retriever.
        Saves prompts to disk for later answering.
    """
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    questions_file = C.get_questions_test(dataset_name)
    with open(questions_file, "rb") as f:
        questions_dataset = pickle.load(f)
    questions = [q["question"] for q in questions_dataset.questions]

    if universal:
        portfolio_path = C.get_universal_portfolio(retriever, num_docs_to_fetch, embedder=embedder_key)
        retrievals_path = C.get_universal_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    else:
        portfolio_path = C.get_retriever_portfolio(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
        retrievals_path = C.get_portfolio_retrievals_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)

    with open(portfolio_path, "rb") as f:
        portfolio_info = pickle.load(f)
    portfolio_indices = portfolio_info["portfolio"]
    if portfolio_size is not None:
        k = min(portfolio_size, len(portfolio_indices))
    else:
        k = len(portfolio_indices)
    selected_indices = portfolio_indices[:k]

    with open(retrievals_path, "rb") as f:
        portfolio_retrievals = pickle.load(f)

    answer_fn = answer_prompt_fn

    portfolio_prompts = []

    print(f"Building prompts for {len(questions)} questions and {k} portfolio retrievers...", flush=True)
    for qidx, question in enumerate(tqdm(questions, desc="Prompts", unit="question")):
        per_retriever_units = portfolio_retrievals["results"][qidx][:k]

        for ridx, units in enumerate(per_retriever_units):
            passages = [tu.text for tu in units]
            sp, up = answer_fn(question, passages)
            portfolio_prompts.append({
                "question_idx": qidx,
                "retriever_idx": ridx,
                "system_prompt": sp,
                "user_prompt": up,
            })

    output_file = C.get_answer_prompts_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "dataset": dataset_name,
            "retriever": retriever,
            "embedder": embedder_key,
            "num_docs_to_fetch": num_docs_to_fetch,
            "portfolio_indices": selected_indices,
            "portfolio_size": k,
        },
        "portfolio_prompts": portfolio_prompts,
    }

    with open(output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

def _retrieved_unit_text(unit, *, dataset_name, portfolio_id, question_idx, portfolio_rank, doc_idx):
    if hasattr(unit, "text"):
        return unit.text
    if isinstance(unit, dict) and "text" in unit:
        return unit["text"]
    if isinstance(unit, str):
        return unit
    raise ValueError(
        f"Retrieved unit has no text field: dataset={dataset_name}, "
        f"portfolio_id={portfolio_id}, question_idx={question_idx}, "
        f"portfolio_rank={portfolio_rank}, doc_idx={doc_idx}, "
        f"type={type(unit).__name__}"
    )

def build_portfolio_union_answer_prompts(
    portfolio_id,
    dataset_name,
    num_docs_to_fetch=4,
    portfolio_size=None,
    answer_prompt_fn=answer_prompt,
    split="test",
    max_questions=None,
):
    """
    Build per-member answer prompts for materialized all-pool portfolio test retrievals.

    This reads C.get_portfolio_union_retrievals_test(...), preserves the materialized
    portfolio order, and writes only portfolio member prompts.
    """
    split = split.lower()
    if split != "test":
        raise NotImplementedError(
            f"All-pool portfolio answer prompts are implemented for split='test' only; got {split!r}."
        )
    if max_questions is not None and int(max_questions) < 0:
        raise ValueError(f"max_questions must be non-negative; got {max_questions}.")

    questions_path = C.get_questions_test(dataset_name)
    questions_read_path = _artifact_read_path(questions_path)
    if not os.path.exists(questions_read_path):
        raise FileNotFoundError(
            f"Missing test questions for all-pool answer prompts: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, "
            f"num_docs={num_docs_to_fetch}, expected_path={questions_path}, "
            f"checked_path={questions_read_path}"
        )
    with open(questions_read_path, "rb") as f:
        questions_dataset = pickle.load(f)
    if not hasattr(questions_dataset, "questions") or not isinstance(questions_dataset.questions, list):
        raise ValueError(
            f"Questions payload has no list .questions: dataset={dataset_name}, "
            f"path={questions_path}, checked_path={questions_read_path}"
        )
    questions = [q["question"] for q in questions_dataset.questions]

    retrievals_path = C.get_portfolio_union_retrievals_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    retrievals_read_path = _artifact_read_path(retrievals_path)
    if not os.path.exists(retrievals_read_path):
        raise FileNotFoundError(
            f"Missing materialized all-pool test retrievals for answer prompts: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, "
            f"num_docs={num_docs_to_fetch}, expected_path={retrievals_path}, "
            f"checked_path={retrievals_read_path}"
        )
    with open(retrievals_read_path, "rb") as f:
        retrieval_payload = pickle.load(f)
    if not isinstance(retrieval_payload, dict):
        raise ValueError(
            f"Materialized all-pool retrieval payload must be a dict: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, path={retrievals_path}"
        )

    payload_split = retrieval_payload.get("split")
    if payload_split is not None and payload_split != "test":
        raise ValueError(
            f"Materialized all-pool retrieval split mismatch: expected=test, "
            f"actual={payload_split}, path={retrievals_path}"
        )
    payload_dataset = retrieval_payload.get("dataset")
    if payload_dataset is not None and payload_dataset != dataset_name:
        raise ValueError(
            f"Materialized all-pool retrieval dataset mismatch: expected={dataset_name}, "
            f"actual={payload_dataset}, path={retrievals_path}"
        )
    payload_portfolio_id = retrieval_payload.get("portfolio_id")
    if payload_portfolio_id is not None and payload_portfolio_id != portfolio_id:
        raise ValueError(
            f"Materialized all-pool retrieval portfolio_id mismatch: expected={portfolio_id}, "
            f"actual={payload_portfolio_id}, path={retrievals_path}"
        )
    payload_num_docs = retrieval_payload.get("num_docs", retrieval_payload.get("num_docs_to_fetch"))
    if payload_num_docs is not None and int(payload_num_docs) != int(num_docs_to_fetch):
        raise ValueError(
            f"Materialized all-pool retrieval num_docs mismatch: expected={num_docs_to_fetch}, "
            f"actual={payload_num_docs}, path={retrievals_path}"
        )

    queries = retrieval_payload.get("queries")
    results = retrieval_payload.get("results")
    if not isinstance(queries, list) or not isinstance(results, list):
        raise ValueError(
            f"Materialized all-pool retrieval payload must contain list queries/results: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, path={retrievals_path}"
        )
    if len(queries) != len(questions):
        raise ValueError(
            f"Question count mismatch for all-pool prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, questions={len(questions)}, "
            f"retrieval_queries={len(queries)}, path={retrievals_path}"
        )
    if len(results) != len(questions):
        raise ValueError(
            f"Retrieval result count mismatch for all-pool prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, questions={len(questions)}, "
            f"retrieval_results={len(results)}, path={retrievals_path}"
        )
    for idx, (question, query) in enumerate(zip(questions, queries)):
        if question != query:
            raise ValueError(
                f"Question text/order mismatch for all-pool prompts: dataset={dataset_name}, "
                f"portfolio_id={portfolio_id}, first_mismatch_idx={idx}, "
                f"question={question!r}, retrieval_query={query!r}, path={retrievals_path}"
            )

    selected_retrievers = retrieval_payload.get("selected_retrievers")
    if not isinstance(selected_retrievers, list):
        raise ValueError(
            f"Materialized all-pool retrieval payload must contain selected_retrievers list: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, path={retrievals_path}"
        )

    materialized_portfolio_size = None
    for qidx, row in enumerate(results):
        if not isinstance(row, (list, tuple)):
            raise ValueError(
                f"Materialized retrieval row must be a list/tuple: dataset={dataset_name}, "
                f"portfolio_id={portfolio_id}, question_idx={qidx}, path={retrievals_path}"
            )
        if materialized_portfolio_size is None:
            materialized_portfolio_size = len(row)
        elif len(row) != materialized_portfolio_size:
            raise ValueError(
                f"Materialized retrieval row width mismatch: dataset={dataset_name}, "
                f"portfolio_id={portfolio_id}, question_idx={qidx}, "
                f"expected_width={materialized_portfolio_size}, actual_width={len(row)}, "
                f"path={retrievals_path}"
            )
    if materialized_portfolio_size is None:
        materialized_portfolio_size = len(selected_retrievers)
    if materialized_portfolio_size <= 0:
        raise ValueError(
            f"Materialized all-pool retrieval payload has no portfolio members: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, path={retrievals_path}"
        )
    if len(selected_retrievers) < materialized_portfolio_size:
        raise ValueError(
            f"selected_retrievers shorter than materialized portfolio width: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, "
            f"selected_retrievers={len(selected_retrievers)}, "
            f"materialized_width={materialized_portfolio_size}, path={retrievals_path}"
        )

    if portfolio_size is None:
        k = materialized_portfolio_size
    else:
        k = int(portfolio_size)
        if k <= 0:
            raise ValueError(f"portfolio_size must be positive; got {portfolio_size}.")
        if k > materialized_portfolio_size:
            raise ValueError(
                f"Requested portfolio_size exceeds materialized all-pool portfolio size: "
                f"dataset={dataset_name}, portfolio_id={portfolio_id}, requested={k}, "
                f"materialized={materialized_portfolio_size}, path={retrievals_path}"
            )

    selected_retrievers = [dict(member) for member in selected_retrievers[:k]]
    for rank, member in enumerate(selected_retrievers):
        member.setdefault("portfolio_rank", rank)
        member.setdefault("retriever_idx", rank)

    question_limit = len(questions)
    if max_questions is not None:
        question_limit = min(int(max_questions), question_limit)

    portfolio_prompts = []
    print(
        f"[portfolio-union-prompts] Building prompts: dataset={dataset_name} "
        f"portfolio_id={portfolio_id} split=test questions={question_limit}/{len(questions)} "
        f"portfolio_size={k} num_docs={num_docs_to_fetch}",
        flush=True,
    )
    for qidx in tqdm(range(question_limit), desc="Portfolio union prompts", unit="question"):
        question = questions[qidx]
        per_member_units = results[qidx][:k]
        for rank, units in enumerate(per_member_units):
            if units is None:
                raise ValueError(
                    f"Materialized all-pool retrieval has None result: dataset={dataset_name}, "
                    f"portfolio_id={portfolio_id}, question_idx={qidx}, "
                    f"portfolio_rank={rank}, path={retrievals_path}"
                )
            passages = [
                _retrieved_unit_text(
                    unit,
                    dataset_name=dataset_name,
                    portfolio_id=portfolio_id,
                    question_idx=qidx,
                    portfolio_rank=rank,
                    doc_idx=doc_idx,
                )
                for doc_idx, unit in enumerate(units)
            ]
            system_prompt, user_prompt = answer_prompt_fn(question, passages)
            portfolio_prompts.append(
                {
                    "question_idx": qidx,
                    "retriever_idx": rank,
                    "portfolio_rank": rank,
                    "selected_retriever": selected_retrievers[rank],
                    "question": question,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )

    output_file = C.get_portfolio_union_answer_prompts_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "schema": "portfolio_union_answer_prompts",
            "schema_version": 1,
            "dataset": dataset_name,
            "portfolio_id": portfolio_id,
            "split": "test",
            "num_docs_to_fetch": int(num_docs_to_fetch),
            "portfolio_size": k,
            "num_questions": question_limit,
            "total_questions": len(questions),
            "portfolio_path": retrieval_payload.get(
                "portfolio_path",
                C.get_universal_portfolio_union_manifest(portfolio_id, num_docs_to_fetch),
            ),
            "portfolio_loaded_path": retrieval_payload.get("portfolio_loaded_path"),
            "retrievals_path": retrievals_path,
            "retrievals_loaded_path": retrievals_read_path,
            "questions_path": questions_path,
            "questions_loaded_path": questions_read_path,
            "selected_retrievers": selected_retrievers,
        },
        "portfolio_prompts": portfolio_prompts,
    }

    with open(output_write_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"[portfolio-union-prompts] saved dataset={dataset_name} "
        f"prompts={output_file} records={len(portfolio_prompts)}",
        flush=True,
    )
    if output_write_path != output_file:
        print(
            f"[portfolio-union-prompts] written_path={output_write_path}",
            flush=True,
        )
    return output_file

def build_selector_prompts(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    portfolio_size=None,
    selector_prompt_fn=selector_prompt,
    llm_name=None,
    embedder=None,
):
    """
        Build LLM selector prompts for the test split using the *universal*
        portfolio of retrievers for the given (retriever, num_docs_to_fetch)
        and a specific answer LLM (llm_name).

        For each test question and for every non-empty subset of the first
        `portfolio_size` retrievers from the universal portfolio, we create
        one selector prompt that compares the corresponding candidate answers.

        Prompts are saved to C.get_selector_prompts(dataset, retriever, llm_name, num_docs).
    """
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    # Load questions
    questions_file = C.get_questions_test(dataset_name)
    with open(questions_file, "rb") as f:
        questions_dataset = pickle.load(f)
    questions = [q["question"] for q in questions_dataset.questions]

    # Load universal portfolio and restrict to the requested prefix length.
    portfolio_path = C.get_universal_portfolio(retriever, num_docs_to_fetch, embedder=embedder_key)
    with open(portfolio_path, "rb") as f:
        portfolio_info = pickle.load(f)
    portfolio_indices = portfolio_info["portfolio"]
    if not portfolio_indices:
        raise ValueError(f"Universal portfolio is empty for retriever={retriever}, num_docs={num_docs_to_fetch}.")

    if portfolio_size is not None:
        k = min(portfolio_size, len(portfolio_indices))
    else:
        k = len(portfolio_indices)
    selected_indices = portfolio_indices[:k]

    # Determine which answer file to read candidate answers from.
    if llm_name is None:
        raise ValueError(
            "llm_name must be provided when building selector prompts. "
            "Run answer_prompts_with_llm with a specific model and pass the same llm_name here."
        )

    answers_file = C.get_answers_all(dataset_name, retriever, llm_name, num_docs_to_fetch, embedder=embedder_key)
    if not os.path.exists(answers_file):
        raise FileNotFoundError(
            f"Portfolio answers file not found: {answers_file}. "
            "Make sure answer_prompts_with_llm has been run."
        )

    with open(answers_file, "rb") as f:
        answers_payload = pickle.load(f)

    portfolio_answers = answers_payload.get("answers", [])
    if not portfolio_answers:
        raise ValueError(f"No portfolio answers found in {answers_file}.")

    # Build a lookup: [question_idx][local_retriever_idx] -> raw answer text
    num_questions = len(questions)
    answers_matrix = [[None for _ in range(k)] for _ in range(num_questions)]

    for entry in portfolio_answers:
        qidx = entry.get("question_idx")
        ridx = entry.get("retriever_idx")
        if qidx is None or ridx is None:
            continue
        if qidx < 0 or qidx >= num_questions:
            continue
        if ridx < 0 or ridx >= k:
            # Ignore answers for retrievers outside the selected prefix.
            continue
        answers_matrix[qidx][ridx] = entry.get("response")

    selector_prompts = []

    print(
        f"Building selector prompts for {num_questions} questions and "
        f"{k} universal-portfolio retrievers (all non-empty subsets)...",
        flush=True,
    )

    # For each question, build one prompt per non-empty subset of {0..k-1}.
    for qidx, question in enumerate(tqdm(questions, desc="Selector prompts", unit="question")):
        per_q_answers = answers_matrix[qidx]

        # Skip questions where no answers are available.
        if all(a is None for a in per_q_answers):
            continue

        # Enumerate all non-empty subsets of the first k retrievers.
        # We encode subset membership as a bitmask in [1, 2^k - 1].
        max_mask = (1 << k) - 1
        for mask in range(1, max_mask + 1):
            subset_indices = [
                ridx for ridx in range(k) if (mask & (1 << ridx))
            ]

            # Collect answers for this subset; if any are missing, skip.
            candidate_answers = []
            skip_subset = False
            for ridx in subset_indices:
                ans = per_q_answers[ridx]
                if ans is None:
                    skip_subset = True
                    break
                candidate_answers.append(ans)
            if skip_subset or not candidate_answers:
                continue

            sp, up = selector_prompt_fn(question, candidate_answers, passages_list=None)
            selector_prompts.append({
                "question_idx": qidx,
                "subset_mask": mask,
                "subset_retrievers": subset_indices,
                "system_prompt": sp,
                "user_prompt": up,
            })

    output_file = C.get_selector_prompts(dataset_name, retriever, llm_name, num_docs_to_fetch, embedder=embedder_key)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "dataset": dataset_name,
            "retriever": retriever,
            "embedder": embedder_key,
            "num_docs_to_fetch": num_docs_to_fetch,
            "portfolio_indices": selected_indices,
            "portfolio_size": k,
            "universal": True,
            "answer_llm_name": llm_name,
        },
        "selector_prompts": selector_prompts,
    }

    with open(output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

def _load_pickle_artifact(expected_path, *, context=None, purpose=None):
    context = context if context is not None else purpose
    if context is None:
        context = "pickle artifact"
    read_path = _artifact_read_path(expected_path)
    if not os.path.exists(read_path):
        raise FileNotFoundError(
            f"Missing {context}: expected_path={expected_path}, checked_path={read_path}"
        )
    with open(read_path, "rb") as f:
        return pickle.load(f), read_path

def build_family_best_answer_prompts(
    portfolio_id=C.POOL_SET_ALL_IMPLEMENTED,
    dataset_name=None,
    family=C.DS,
    num_docs_to_fetch=4,
    max_k=5,
    answer_prompt_fn=answer_prompt,
    split="test",
    max_questions=None,
):
    """
    Build answer prompts for the train-selected best single retriever in a family.

    The family-best retrieval artifact stores max_k * num_docs documents per
    question so it can support recall curves.  These answer prompts intentionally
    use only the first num_docs documents, matching the paper-table baseline.
    """
    if dataset_name is None:
        raise ValueError("dataset_name is required.")
    split = str(split).lower()
    if split != "test":
        raise NotImplementedError(f"Family-best answer prompts support split='test' only; got {split!r}.")
    family = _normalize_family_list([family])[0]
    if int(num_docs_to_fetch) <= 0:
        raise ValueError(f"num_docs_to_fetch must be positive; got {num_docs_to_fetch}.")
    if int(max_k) <= 0:
        raise ValueError(f"max_k must be positive; got {max_k}.")
    if max_questions is not None and int(max_questions) < 0:
        raise ValueError(f"max_questions must be non-negative; got {max_questions}.")

    context = (
        f"family-best answer prompts dataset={dataset_name} portfolio_id={portfolio_id} "
        f"family={family} num_docs={num_docs_to_fetch} max_k={max_k}"
    )
    questions_path = C.get_questions_test(dataset_name)
    questions_dataset, questions_read_path = _load_pickle_artifact(
        questions_path,
        context=f"test questions ({context})",
    )
    questions = getattr(questions_dataset, "questions", None)
    if not isinstance(questions, list):
        raise ValueError(f"Questions payload has no list .questions: path={questions_path}")

    retrievals_path = C.get_family_best_retrievals_test(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    try:
        retrieval_payload, retrievals_read_path = _load_pickle_artifact(
            retrievals_path,
            context=f"family-best retrievals ({context})",
        )
    except FileNotFoundError as exc:
        manifest, manifest_path, manifest_read_path = _load_family_best_manifest(
            portfolio_id,
            num_docs_to_fetch,
        )
        member = _selected_family_member_from_manifest(manifest, family)
        target_queries = [q["question"] for q in questions]
        source_payload, source_artifact = _load_source_full_pool_test_payload(
            dataset_name=dataset_name,
            portfolio_id=portfolio_id,
            member=member,
            num_docs_to_fetch=num_docs_to_fetch,
            target_queries=target_queries,
        )
        selected_results = _slice_source_member_results(
            source_payload=source_payload,
            source_artifact=source_artifact,
            member=member,
            dataset_name=dataset_name,
            portfolio_id=portfolio_id,
            num_docs_to_fetch=num_docs_to_fetch,
        )
        retrieval_payload = {
            "schema": "family_best_answer_prompt_source_from_full_pool_test",
            "schema_version": 1,
            "portfolio_id": portfolio_id,
            "family": family,
            "dataset": dataset_name,
            "split": "test",
            "num_docs": int(num_docs_to_fetch),
            "max_k": int(max_k),
            "queries": target_queries,
            "results": selected_results,
            "selected_retriever": member,
            "manifest_path": manifest_path,
            "manifest_loaded_path": manifest_read_path,
            "source_artifacts": [source_artifact],
            "missing_family_best_retrievals_error": str(exc),
        }
        retrievals_read_path = source_artifact["loaded_path"]
        print(
            f"[family-best-prompts] family-best retrieval artifact missing; "
            f"using selected member from source full-pool test retrievals for prompts only: "
            f"dataset={dataset_name} family={family} "
            f"missing_expected_path={retrievals_path} "
            f"source_path={source_artifact['path']} "
            f"source_loaded_path={source_artifact['loaded_path']} "
            f"local_idx={source_artifact['local_idx']}",
            flush=True,
        )
    if not isinstance(retrieval_payload, dict):
        raise ValueError(f"Family-best retrieval payload must be a dict: path={retrievals_path}")

    for key, expected in (
        ("portfolio_id", portfolio_id),
        ("family", family),
        ("dataset", dataset_name),
        ("split", "test"),
    ):
        actual = retrieval_payload.get(key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"Family-best retrieval payload {key} mismatch: "
                f"expected={expected}, actual={actual}, path={retrievals_path}"
            )
    for key, expected in (("num_docs", num_docs_to_fetch), ("max_k", max_k)):
        actual = retrieval_payload.get(key)
        if actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"Family-best retrieval payload {key} mismatch: "
                f"expected={expected}, actual={actual}, path={retrievals_path}"
            )

    queries = retrieval_payload.get("queries")
    results = retrieval_payload.get("results")
    if not isinstance(queries, list) or not isinstance(results, list):
        raise ValueError(f"Family-best retrieval payload must contain list queries/results: path={retrievals_path}")
    if len(results) != len(questions):
        raise ValueError(
            f"Family-best retrieval result count mismatch: dataset={dataset_name}, "
            f"family={family}, results={len(results)}, questions={len(questions)}, path={retrievals_path}"
        )
    if len(queries) != len(questions):
        raise ValueError(
            f"Family-best retrieval query count mismatch: dataset={dataset_name}, "
            f"family={family}, queries={len(queries)}, questions={len(questions)}, path={retrievals_path}"
        )
    for qidx, (query, question) in enumerate(zip(queries, questions)):
        expected_question = question.get("question")
        if query != expected_question:
            raise ValueError(
                f"Family-best retrieval question text/order mismatch: dataset={dataset_name}, "
                f"family={family}, first_mismatch_idx={qidx}, path={retrievals_path}"
            )

    question_limit = len(questions)
    if max_questions is not None:
        question_limit = min(int(max_questions), question_limit)

    selected_retriever = retrieval_payload.get("selected_retriever")
    if isinstance(selected_retriever, dict):
        selected_retriever = dict(selected_retriever)

    prompts = []
    print(
        f"[family-best-prompts] Building prompts: dataset={dataset_name} "
        f"portfolio_id={portfolio_id} family={family} questions={question_limit}/{len(questions)} "
        f"docs_per_prompt={num_docs_to_fetch}",
        flush=True,
    )
    for qidx in tqdm(range(question_limit), desc="Family-best prompts", unit="question"):
        question = questions[qidx].get("question")
        units = results[qidx]
        if not isinstance(units, (list, tuple)):
            raise ValueError(
                f"Family-best retrieval row must be a list/tuple: dataset={dataset_name}, "
                f"family={family}, question_idx={qidx}, path={retrievals_path}"
            )
        selected_units = list(units[: int(num_docs_to_fetch)])
        passages = [
            _retrieved_unit_text(
                unit,
                dataset_name=dataset_name,
                portfolio_id=portfolio_id,
                question_idx=qidx,
                portfolio_rank=0,
                doc_idx=doc_idx,
            )
            for doc_idx, unit in enumerate(selected_units)
        ]
        system_prompt, user_prompt = answer_prompt_fn(question, passages)
        prompts.append(
            {
                "question_idx": qidx,
                "family": family,
                "selected_retriever": selected_retriever,
                "question": question,
                "num_docs_used": len(passages),
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    output_file = C.get_family_best_answer_prompts_test(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "schema": "family_best_answer_prompts",
            "schema_version": 1,
            "dataset": dataset_name,
            "portfolio_id": portfolio_id,
            "family": family,
            "split": "test",
            "num_docs_to_fetch": int(num_docs_to_fetch),
            "max_k": int(max_k),
            "num_questions": question_limit,
            "total_questions": len(questions),
            "retrievals_path": retrievals_path,
            "retrievals_loaded_path": retrievals_read_path,
            "questions_path": questions_path,
            "questions_loaded_path": questions_read_path,
            "selected_retriever": selected_retriever,
        },
        "answer_prompts": prompts,
    }
    with open(output_write_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[family-best-prompts] saved dataset={dataset_name} family={family} "
        f"prompts={len(prompts)} path={output_file}",
        flush=True,
    )
    if output_write_path != output_file:
        print(f"[family-best-prompts] written_path={output_write_path}", flush=True)
    return output_file

def _portfolio_router_judge_context(
    *,
    dataset_name,
    portfolio_id,
    num_docs_to_fetch,
    portfolio_size,
    ell,
    split,
    run_id,
    answer_llm,
    judge_llm_name=None,
):
    parts = [
        f"dataset={dataset_name}",
        f"portfolio_id={portfolio_id}",
        f"num_docs={num_docs_to_fetch}",
        f"k={portfolio_size}",
        f"ell={ell}",
        f"split={split}",
        f"run_id={run_id}",
        f"answer_llm={answer_llm}",
    ]
    if judge_llm_name is not None:
        parts.append(f"judge_llm={judge_llm_name}")
    return ", ".join(parts)

def _validate_router_judge_common(
    *,
    portfolio_id,
    dataset_name,
    num_docs_to_fetch,
    portfolio_size,
    ell,
    run_id,
    answer_llm,
    split,
    judge_llm_name=None,
):
    del portfolio_id, dataset_name, num_docs_to_fetch, judge_llm_name
    split = split.lower()
    if split != "test":
        raise NotImplementedError(
            f"Portfolio-router judge artifacts currently support split='test' only; got {split!r}."
        )
    if run_id is None or str(run_id).strip() == "":
        raise ValueError("run_id is required for portfolio-router judge artifacts.")
    if answer_llm is None or str(answer_llm).strip() == "":
        raise ValueError("answer_llm is required for portfolio-router judge artifacts.")
    k = int(portfolio_size)
    ell = int(ell)
    if k <= 0:
        raise ValueError(f"portfolio_size must be positive; got {portfolio_size}.")
    if ell < 1 or ell > k:
        raise ValueError(f"ell must satisfy 1 <= ell <= portfolio_size; got ell={ell}, k={k}.")
    return split, k, ell

def _validate_optional_payload_field(payload, key, expected, *, path, context):
    actual = payload.get(key)
    if actual is not None and actual != expected:
        raise ValueError(
            f"{context} {key} mismatch: expected={expected}, actual={actual}, path={path}"
        )

def _validate_optional_int_payload_field(payload, keys, expected, *, path, context):
    for key in keys:
        if key in payload and payload[key] is not None:
            actual = int(payload[key])
            if actual != int(expected):
                raise ValueError(
                    f"{context} {key} mismatch: expected={expected}, actual={actual}, path={path}"
                )
            return

def _coerce_router_judge_matrix(value, *, name, portfolio_size, row_count=None, path=None):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] < int(portfolio_size):
        raise ValueError(
            f"Router {name} must be [Q, K]: path={path}, "
            f"{name}_shape={matrix.shape}, k={portfolio_size}"
        )
    if row_count is not None and matrix.shape[0] != int(row_count):
        raise ValueError(
            f"Router {name} row count mismatch: path={path}, "
            f"{name}_shape={matrix.shape}, rows={row_count}"
        )
    return matrix[:, : int(portfolio_size)]

def _router_judge_top_indices_from_matrix(matrix):
    return np.argsort(-np.asarray(matrix, dtype=np.float32), axis=1, kind="mergesort").astype(
        np.int64,
        copy=False,
    )

def _router_judge_top_indices_from_argmax(argmax, *, portfolio_size, path):
    argmax = np.asarray(argmax, dtype=np.int64)
    if argmax.ndim != 1:
        raise ValueError(f"argmax must be 1D when deriving top_indices: path={path}, shape={argmax.shape}")
    if argmax.size and (argmax.min() < 0 or argmax.max() >= int(portfolio_size)):
        raise ValueError(
            f"argmax contains out-of-range portfolio ranks: path={path}, "
            f"min={int(argmax.min())}, max={int(argmax.max())}, k={portfolio_size}"
        )
    base = np.arange(int(portfolio_size), dtype=np.int64)
    top_indices = np.empty((argmax.shape[0], int(portfolio_size)), dtype=np.int64)
    for row_idx, winner in enumerate(argmax):
        top_indices[row_idx, 0] = int(winner)
        top_indices[row_idx, 1:] = base[base != int(winner)]
    return top_indices

def _coerce_router_judge_prediction_payload(
    payload,
    *,
    dataset_name,
    portfolio_id,
    num_docs_to_fetch,
    portfolio_size,
    split,
    run_id,
    path,
):
    if not isinstance(payload, dict):
        raise ValueError(f"Router prediction payload must be a dict: path={path}")
    context = "Router prediction payload"
    _validate_optional_payload_field(payload, "dataset", dataset_name, path=path, context=context)
    _validate_optional_payload_field(payload, "portfolio_id", portfolio_id, path=path, context=context)
    _validate_optional_payload_field(payload, "split", split, path=path, context=context)
    _validate_optional_payload_field(payload, "run_id", run_id, path=path, context=context)
    _validate_optional_int_payload_field(
        payload,
        ("portfolio_size", "k"),
        portfolio_size,
        path=path,
        context=context,
    )
    _validate_optional_int_payload_field(
        payload,
        ("num_docs", "num_docs_to_fetch"),
        num_docs_to_fetch,
        path=path,
        context=context,
    )

    scores = payload.get("scores")
    probabilities = payload.get("probabilities")
    raw_top_indices = payload.get("top_indices")
    if raw_top_indices is not None:
        top_indices = np.asarray(raw_top_indices, dtype=np.int64)
        if top_indices.ndim != 2:
            raise ValueError(f"top_indices must be 2D [Q, K]: path={path}, shape={top_indices.shape}")
        if top_indices.shape[1] < int(portfolio_size):
            raise ValueError(
                f"top_indices has fewer than K columns: path={path}, "
                f"shape={top_indices.shape}, k={portfolio_size}"
            )
        top_indices = top_indices[:, : int(portfolio_size)]
        row_count = int(top_indices.shape[0])
    else:
        scores_preview = _coerce_router_judge_matrix(
            scores,
            name="scores",
            portfolio_size=portfolio_size,
            path=path,
        )
        probabilities_preview = _coerce_router_judge_matrix(
            probabilities,
            name="probabilities",
            portfolio_size=portfolio_size,
            path=path,
        )
        ranking_matrix = scores_preview if scores_preview is not None else probabilities_preview
        if ranking_matrix is not None:
            top_indices = _router_judge_top_indices_from_matrix(ranking_matrix)
        else:
            argmax = payload.get("argmax", payload.get("predictions", payload.get("labels")))
            if argmax is None:
                raise ValueError(
                    f"Router prediction payload is missing top_indices and cannot derive it "
                    f"from scores/probabilities/argmax: path={path}"
                )
            top_indices = _router_judge_top_indices_from_argmax(
                argmax,
                portfolio_size=portfolio_size,
                path=path,
            )
        row_count = int(top_indices.shape[0])

    if top_indices.size and (top_indices.min() < 0 or top_indices.max() >= int(portfolio_size)):
        raise ValueError(
            f"top_indices contains out-of-range portfolio ranks: path={path}, "
            f"min={int(top_indices.min())}, max={int(top_indices.max())}, k={portfolio_size}"
        )
    if top_indices.shape[1] != int(portfolio_size) or any(
        np.unique(row).shape[0] != int(portfolio_size) for row in top_indices
    ):
        raise ValueError(
            f"top_indices rows must be full permutations of portfolio ranks: "
            f"path={path}, shape={top_indices.shape}, k={portfolio_size}"
        )

    scores = _coerce_router_judge_matrix(
        scores,
        name="scores",
        portfolio_size=portfolio_size,
        row_count=row_count,
        path=path,
    )
    probabilities = _coerce_router_judge_matrix(
        probabilities,
        name="probabilities",
        portfolio_size=portfolio_size,
        row_count=row_count,
        path=path,
    )

    if "question_indices" not in payload:
        raise ValueError(f"Router prediction payload is missing question_indices: path={path}")
    question_indices = np.asarray(payload["question_indices"], dtype=np.int64)
    if question_indices.ndim != 1 or question_indices.shape[0] != top_indices.shape[0]:
        raise ValueError(
            f"question_indices must be [Q] and align with top_indices: "
            f"path={path}, shape={question_indices.shape}, top_indices_shape={top_indices.shape}"
        )
    if question_indices.size and question_indices.min() < 0:
        raise ValueError(f"question_indices contains negative values: path={path}")

    return top_indices, question_indices, scores, probabilities

def _answer_rank_from_record(record, *, k):
    for key in ("portfolio_rank", "retriever_idx", "portfolio_member_idx", "member_idx", "ridx", "rank"):
        if key not in record or record[key] is None:
            continue
        raw = int(record[key])
        if key == "rank":
            raw -= 1
        elif raw == int(k):
            raw -= 1
        if 0 <= raw < int(k):
            return raw
    return None

def _portfolio_union_answers_to_matrix(payload, *, num_questions, k):
    raw_answers = payload.get("answers", payload) if isinstance(payload, dict) else payload
    matrix = [[None for _ in range(int(k))] for _ in range(int(num_questions))]
    if raw_answers is None:
        return matrix

    def _assign(qidx, rank, value):
        if qidx is None or rank is None:
            return
        qidx = int(qidx)
        rank = int(rank)
        if 0 <= qidx < int(num_questions) and 0 <= rank < int(k):
            matrix[qidx][rank] = value

    if isinstance(raw_answers, dict):
        for raw_qidx, per_rank in raw_answers.items():
            try:
                qidx = int(raw_qidx)
            except (TypeError, ValueError):
                continue
            if isinstance(per_rank, dict):
                for rank in range(int(k)):
                    _assign(qidx, rank, per_rank.get(rank, per_rank.get(str(rank))))
            elif isinstance(per_rank, (list, tuple)):
                for rank in range(min(int(k), len(per_rank))):
                    _assign(qidx, rank, per_rank[rank])
        return matrix

    if not isinstance(raw_answers, list):
        raise ValueError(f"Unsupported all-pool answer payload type: {type(raw_answers).__name__}")

    first = next((item for item in raw_answers if item is not None), None)
    if isinstance(first, (list, tuple)):
        for qidx, row in enumerate(raw_answers[: int(num_questions)]):
            if isinstance(row, (list, tuple)):
                for rank in range(min(int(k), len(row))):
                    _assign(qidx, rank, row[rank])
            elif isinstance(row, dict):
                for rank in range(int(k)):
                    _assign(qidx, rank, row.get(rank, row.get(str(rank))))
        return matrix

    for record in raw_answers:
        if not isinstance(record, dict):
            continue
        qidx = record.get("question_idx", record.get("question_id", record.get("question_index", record.get("q_idx"))))
        rank = _answer_rank_from_record(record, k=k)
        _assign(qidx, rank, record)
    return matrix

def _normalize_portfolio_answer_record(cell, *, question_idx, portfolio_rank):
    if cell is None:
        return None
    if isinstance(cell, dict):
        record = dict(cell)
    else:
        record = {"response": str(cell), "error": None}
    record.setdefault("question_idx", int(question_idx))
    record.setdefault("portfolio_rank", int(portfolio_rank))
    record.setdefault("retriever_idx", int(portfolio_rank))
    if "portfolio_rank" not in record and "retriever_idx" in record:
        record["portfolio_rank"] = record["retriever_idx"]
    if "retriever_idx" not in record and "portfolio_rank" in record:
        record["retriever_idx"] = record["portfolio_rank"]
    return record

def _response_from_portfolio_answer_record(record):
    if record is None:
        return None, "missing"
    error = record.get("error") if isinstance(record, dict) else None
    if error not in (None, ""):
        return None, f"error={error}"
    if isinstance(record, dict):
        for key in ("response", "answer", "prediction", "text", "output"):
            value = record.get(key)
            if value is not None and str(value).strip():
                return str(value), None
        return None, "missing_response"
    return str(record), None

def build_portfolio_router_judge_prompts(
    portfolio_id,
    dataset_name,
    num_docs_to_fetch=4,
    portfolio_size=5,
    ell=2,
    run_id=None,
    answer_llm=None,
    split="test",
    selector_prompt_fn=selector_prompt,
    max_questions=None,
    strict_answers=True,
):
    """
    Build final all-pool portfolio-router judge prompts from portfolio-router top-ell answers.

    For ell=1, no judge prompts are created; the saved payload records
    selection_policy='direct_top1' and scoring should evaluate the portfolio-router top-1
    member answer directly.
    """
    split, k, ell = _validate_router_judge_common(
        portfolio_id=portfolio_id,
        dataset_name=dataset_name,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        ell=ell,
        run_id=run_id,
        answer_llm=answer_llm,
        split=split,
    )
    if max_questions is not None and int(max_questions) < 0:
        raise ValueError(f"max_questions must be non-negative; got {max_questions}.")

    context = _portfolio_router_judge_context(
        dataset_name=dataset_name,
        portfolio_id=portfolio_id,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=k,
        ell=ell,
        split=split,
        run_id=run_id,
        answer_llm=answer_llm,
    )
    predictions_path = C.get_portfolio_router_predictions(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
        k,
        split,
        run_id,
    )
    prediction_payload, predictions_read_path = _load_pickle_artifact(
        predictions_path,
        context=f"portfolio-router predictions ({context})",
    )
    top_indices, question_indices, router_scores, router_probabilities = (
        _coerce_router_judge_prediction_payload(
            prediction_payload,
            dataset_name=dataset_name,
            portfolio_id=portfolio_id,
            num_docs_to_fetch=num_docs_to_fetch,
            portfolio_size=k,
            split=split,
            run_id=run_id,
            path=predictions_path,
        )
    )

    answers_path = C.get_portfolio_union_answers_all(
        portfolio_id,
        dataset_name,
        answer_llm,
        num_docs_to_fetch,
    )
    answers_payload, answers_read_path = _load_pickle_artifact(
        answers_path,
        context=f"all-pool per-member answers ({context})",
    )
    if not isinstance(answers_payload, dict):
        raise ValueError(f"All-pool answer payload must be a dict: path={answers_path}")
    answers_meta = answers_payload.get("meta", {})
    if isinstance(answers_meta, dict):
        _validate_optional_payload_field(answers_meta, "dataset", dataset_name, path=answers_path, context="Answer payload meta")
        _validate_optional_payload_field(answers_meta, "portfolio_id", portfolio_id, path=answers_path, context="Answer payload meta")
        _validate_optional_payload_field(answers_meta, "split", split, path=answers_path, context="Answer payload meta")
        _validate_optional_payload_field(answers_meta, "llm_name", answer_llm, path=answers_path, context="Answer payload meta")
        _validate_optional_int_payload_field(
            answers_meta,
            ("num_docs_to_fetch", "num_docs"),
            num_docs_to_fetch,
            path=answers_path,
            context="Answer payload meta",
        )
        meta_k = answers_meta.get("portfolio_size")
        if meta_k is not None and int(meta_k) < k:
            raise ValueError(
                f"Answer payload portfolio_size is smaller than requested k: "
                f"{context}, answer_payload_k={meta_k}, path={answers_path}"
            )

    questions_path = C.get_questions_test(dataset_name)
    questions_dataset, questions_read_path = _load_pickle_artifact(
        questions_path,
        context=f"test questions ({context})",
    )
    questions = getattr(questions_dataset, "questions", None)
    if not isinstance(questions, list):
        raise ValueError(
            f"Questions payload has no list .questions: {context}, path={questions_path}"
        )
    if question_indices.size and int(question_indices.max()) >= len(questions):
        raise IndexError(
            f"Router prediction question index out of range: {context}, "
            f"max_question_idx={int(question_indices.max())}, questions={len(questions)}, "
            f"path={predictions_path}"
        )

    question_texts = prediction_payload.get("question_texts")
    if question_texts is not None:
        if len(question_texts) != top_indices.shape[0]:
            raise ValueError(
                f"Prediction question_texts length mismatch: {context}, "
                f"question_texts={len(question_texts)}, prediction_rows={top_indices.shape[0]}, "
                f"path={predictions_path}"
            )
        for row_idx, original_qidx in enumerate(question_indices):
            expected_question = questions[int(original_qidx)].get("question")
            if question_texts[row_idx] != expected_question:
                raise ValueError(
                    f"Prediction question text mismatch: {context}, "
                    f"prediction_row_idx={row_idx}, question_idx={int(original_qidx)}, "
                    f"path={predictions_path}"
                )

    selected_retrievers = prediction_payload.get("selected_retrievers")
    if not isinstance(selected_retrievers, list):
        selected_retrievers = answers_meta.get("selected_retrievers") if isinstance(answers_meta, dict) else None
    if not isinstance(selected_retrievers, list):
        selected_retrievers = []
    if selected_retrievers and len(selected_retrievers) < k:
        raise ValueError(
            f"selected_retrievers is shorter than k: {context}, "
            f"selected_retrievers={len(selected_retrievers)}, path={predictions_path}"
        )
    selected_retrievers = [dict(member) for member in selected_retrievers[:k]]
    for rank, member in enumerate(selected_retrievers):
        member.setdefault("portfolio_rank", rank)
        member.setdefault("retriever_idx", rank)

    answer_matrix = _portfolio_union_answers_to_matrix(
        answers_payload,
        num_questions=len(questions),
        k=k,
    )

    total_prediction_rows = int(top_indices.shape[0])
    row_limit = total_prediction_rows
    if max_questions is not None:
        row_limit = min(int(max_questions), row_limit)

    judge_prompts = []
    skipped_questions = 0
    skipped_missing_answers = 0
    skipped_error_answers = 0
    selection_policy = "direct_top1" if ell == 1 else "router_top_ell_judge"
    print(
        f"[portfolio-router-judge-prompts] Building: {context} "
        f"prediction_rows={row_limit}/{total_prediction_rows} policy={selection_policy}",
        flush=True,
    )
    for row_idx in tqdm(range(row_limit), desc="Portfolio router judge prompts", unit="question"):
        original_qidx = int(question_indices[row_idx])
        question = questions[original_qidx].get("question")
        candidate_ranks = [int(rank) for rank in top_indices[row_idx, :ell]]
        candidate_answers = []
        candidate_records = []
        missing_reasons = []
        for router_order, rank in enumerate(candidate_ranks):
            record = _normalize_portfolio_answer_record(
                answer_matrix[original_qidx][rank],
                question_idx=original_qidx,
                portfolio_rank=rank,
            )
            response, missing_reason = _response_from_portfolio_answer_record(record)
            if missing_reason is not None:
                missing_reasons.append(
                    {
                        "router_order": router_order,
                        "portfolio_rank": rank,
                        "reason": missing_reason,
                    }
                )
                if missing_reason.startswith("error="):
                    skipped_error_answers += 1
                else:
                    skipped_missing_answers += 1
                continue
            candidate_answers.append(response)
            candidate_records.append(record)

        if missing_reasons:
            message = (
                f"Missing selected candidate answer(s) for portfolio-router judge prompt: {context}, "
                f"prediction_row_idx={row_idx}, question_idx={original_qidx}, "
                f"missing={missing_reasons}, router_predictions_path={predictions_path}, "
                f"portfolio_answers_path={answers_path}"
            )
            if strict_answers:
                raise ValueError(message)
            skipped_questions += 1
            continue

        if ell == 1:
            continue

        candidate_scores = [
            None if router_scores is None else float(router_scores[row_idx, rank])
            for rank in candidate_ranks
        ]
        candidate_probabilities = [
            None if router_probabilities is None else float(router_probabilities[row_idx, rank])
            for rank in candidate_ranks
        ]
        candidate_selected_retrievers = [
            selected_retrievers[rank] if rank < len(selected_retrievers) else {"portfolio_rank": rank, "retriever_idx": rank}
            for rank in candidate_ranks
        ]
        system_prompt, user_prompt = selector_prompt_fn(question, candidate_answers, passages_list=None)
        judge_prompts.append(
            {
                "question_idx": original_qidx,
                "prediction_row_idx": row_idx,
                "ell": ell,
                "portfolio_size": k,
                "candidate_portfolio_ranks": candidate_ranks,
                "candidate_router_order": list(range(ell)),
                "candidate_router_scores": candidate_scores,
                "candidate_router_probabilities": candidate_probabilities,
                "candidate_answers": candidate_answers,
                "candidate_answer_records": candidate_records,
                "selected_retrievers": candidate_selected_retrievers,
                "question": question,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )

    output_file = C.get_portfolio_router_judge_prompts(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
        k,
        ell,
        split,
        run_id,
        answer_llm,
    )
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "schema": "portfolio_router_judge_prompts",
            "schema_version": 1,
            "dataset": dataset_name,
            "portfolio_id": portfolio_id,
            "split": split,
            "num_docs_to_fetch": int(num_docs_to_fetch),
            "portfolio_size": k,
            "ell": ell,
            "run_id": run_id,
            "answer_llm": answer_llm,
            "selection_policy": selection_policy,
            "router_predictions_path": predictions_path,
            "router_predictions_loaded_path": predictions_read_path,
            "portfolio_answers_path": answers_path,
            "portfolio_answers_loaded_path": answers_read_path,
            "questions_path": questions_path,
            "questions_loaded_path": questions_read_path,
            "num_questions": row_limit,
            "total_prediction_questions": total_prediction_rows,
            "total_test_questions": len(questions),
            "max_questions": max_questions,
            "num_prompts": len(judge_prompts),
            "skipped_questions": skipped_questions,
            "skipped_missing_answers": skipped_missing_answers,
            "skipped_error_answers": skipped_error_answers,
            "strict_answers": bool(strict_answers),
            "selected_retrievers": selected_retrievers,
        },
        "judge_prompts": judge_prompts,
    }
    with open(output_write_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[portfolio-router-judge-prompts] saved dataset={dataset_name} "
        f"prompts={len(judge_prompts)} path={output_file}",
        flush=True,
    )
    if output_write_path != output_file:
        print(
            f"[portfolio-router-judge-prompts] written_path={output_write_path}",
            flush=True,
        )
    return output_file

def build_baseline_answer_prompts(
    dataset_name,
    num_docs_to_fetch,
    *,
    device="cpu",
    pre_filter=1000,
    answer_prompt_fn=answer_prompt,
):
    with open(C.get_questions_test(dataset_name), "rb") as f:
        questions_dataset = pickle.load(f)
    questions = [q["question"] for q in questions_dataset.questions]

    retrievals_path = Path(C.get_retrievals_test(dataset_name, C.DS, num_docs_to_fetch))
    if not retrievals_path.exists():
        raise FileNotFoundError(f"DS retrievals_test not found: {retrievals_path}")
    with open(retrievals_path, "rb") as f:
        retrievals_payload = pickle.load(f)
    retrievals = retrievals_payload.get("results", [])

    try:
        ds_naive_idx = C.DS_POOL_PARAMETERS.index((0, 1))
    except ValueError as exc:
        raise ValueError("DS pool parameters missing (0, 1) entry.") from exc

    no_retrieval_prompts = []
    naive_retrieval_prompts = []

    for qidx, question in enumerate(tqdm(questions, desc=f"Baseline prompts ({dataset_name})", unit="question")):
        sp_none, up_none = answer_prompt_fn(question, [])
        no_retrieval_prompts.append(
            {
                "question_idx": qidx,
                "system_prompt": sp_none,
                "user_prompt": up_none,
            }
        )

        units = []
        if qidx < len(retrievals) and retrievals[qidx] is not None:
            q_retrievals = retrievals[qidx]
            if ds_naive_idx < len(q_retrievals) and q_retrievals[ds_naive_idx] is not None:
                units = q_retrievals[ds_naive_idx]

        passages = [tu.text for tu in units]
        sp_naive, up_naive = answer_prompt_fn(question, passages)
        naive_retrieval_prompts.append(
            {
                "question_idx": qidx,
                "system_prompt": sp_naive,
                "user_prompt": up_naive,
            }
        )

    output_file = C.get_baseline_answer_prompts(dataset_name)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "dataset": dataset_name,
            "num_docs_to_fetch": num_docs_to_fetch,
            "retriever": "ds_naive",
            "split": "test",
        },
        "no_retrieval_prompts": no_retrieval_prompts,
        "naive_retrieval_prompts": naive_retrieval_prompts,
    }

    with open(output_file, "wb") as f:
        pickle.dump(payload, f)

    return output_file

# -----------------------------------------------------------------------------
# Prompt Answering
# -----------------------------------------------------------------------------

# Functions: answer_prompts_with_llm, answer_portfolio_union_prompts_with_llm
#            answer_family_best_prompts_with_llm, answer_portfolio_router_judge_prompts_with_llm
#            answer_baseline_prompts_with_llm, answer_selector_prompts_with_llm

def answer_prompts_with_llm(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    llm,
    llm_name,
    max_workers=16,
    checkpoint_every=1000,
    embedder=None,
):
    """
        Consumes the saved prompts and generates answers using the provided LLM.
        Saves one answer per selected portfolio retriever.
    """
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    prompts_file = C.get_answer_prompts_test(dataset_name, retriever, num_docs_to_fetch, embedder=embedder_key)
    with open(prompts_file, "rb") as f:
        payload = pickle.load(f)

    meta = payload["meta"]
    portfolio_prompts = payload.get("portfolio_prompts", [])

    portfolio_file = C.get_answers_all(dataset_name, retriever, llm_name, num_docs_to_fetch, embedder=embedder_key)
    Path(portfolio_file).parent.mkdir(parents=True, exist_ok=True)

    meta_out = {
        **meta,
        "llm_name": llm_name,
        "num_workers": max_workers,
    }

    def _run_with_checkpoint(prompts, label, output_path):
        """
            Run LLM over the given prompts with checkpointing support.
            If output_path exists, resume from the existing answers and
            only process missing entries. Saves a checkpoint every
            `checkpoint_every` completed answers (per file).
        """
        if not prompts:
            return []

        # Initialize answers list, seeding from any existing file.
        answers = [None] * len(prompts)
        if os.path.exists(output_path):
            try:
                with open(output_path, "rb") as f:
                    existing_payload = pickle.load(f)
                existing_answers = existing_payload.get("answers", [])
            except Exception as exc:
                print(f"[{label}] Could not load existing answers from {output_path}: {exc}", flush=True)
                existing_answers = []

            # If lengths match, trust positional alignment; otherwise,
            # copy as much as we can by index.
            if existing_answers:
                if len(existing_answers) == len(prompts):
                    answers = existing_answers
                else:
                    n = min(len(existing_answers), len(prompts))
                    for i in range(n):
                        answers[i] = existing_answers[i]

        # Determine which indices still need to be processed.
        remaining_indices = [i for i, ans in enumerate(answers) if ans is None]
        if not remaining_indices:
            print(f"[{label}] All prompts already answered ({len(prompts)} total).", flush=True)
            return answers

        def _call(idx):
            prompt = prompts[idx]
            return llm.answer(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        total_done_before = len([a for a in answers if a is not None])
        print(
            f"[{label}] Starting/resuming: {len(prompts)} prompts "
            f"({total_done_before} done, {len(remaining_indices)} remaining).",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(remaining_indices),
            desc=label,
            unit="prompt",
        ) as pbar:
            future_to_idx = {
                executor.submit(_call, idx): idx for idx in remaining_indices
            }
            since_last_checkpoint = 0

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                prompt = prompts[idx]
                try:
                    response = future.result()
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)

                entry = {
                    "question_idx": prompt["question_idx"],
                    "system_prompt": prompt["system_prompt"],
                    "user_prompt": prompt["user_prompt"],
                    "response": response,
                    "error": error,
                }
                if "retriever_idx" in prompt:
                    entry["retriever_idx"] = prompt["retriever_idx"]
                if "prefix_idx" in prompt:
                    entry["prefix_idx"] = prompt["prefix_idx"]

                answers[idx] = entry
                pbar.update(1)
                since_last_checkpoint += 1

                # Periodic checkpoint
                if checkpoint_every and since_last_checkpoint >= checkpoint_every:
                    with open(output_path, "wb") as f:
                        pickle.dump({"meta": meta_out, "answers": answers}, f)
                    total_done = len([a for a in answers if a is not None])
                    print(
                        f"[{label}] Checkpoint saved to {output_path} "
                        f"({total_done}/{len(answers)} answered).",
                        flush=True,
                    )
                    since_last_checkpoint = 0

        # Final save after finishing all remaining prompts.
        with open(output_path, "wb") as f:
            pickle.dump({"meta": meta_out, "answers": answers}, f)
        print(
            f"[{label}] Completed all prompts "
            f"({len([a for a in answers if a is not None])}/{len(answers)} answered).",
            flush=True,
        )
        return answers

    print(
        f"Answering {len(portfolio_prompts)} portfolio prompts...",
        flush=True,
    )

    _run_with_checkpoint(
        portfolio_prompts,
        "Portfolio answers",
        portfolio_file,
    )

    return portfolio_file

def answer_portfolio_union_prompts_with_llm(
    portfolio_id,
    dataset_name,
    num_docs_to_fetch,
    llm,
    llm_name,
    max_workers=16,
    checkpoint_every=1000,
    portfolio_size=None,
):
    """
    Answer materialized all-pool portfolio member prompts for the test split.

    Only portfolio_prompts are consumed. No single-prefix or union-prefix answer
    artifacts are produced.
    """
    prompts_file = C.get_portfolio_union_answer_prompts_test(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
    )
    prompts_read_path = _artifact_read_path(prompts_file)
    if not os.path.exists(prompts_read_path):
        raise FileNotFoundError(
            f"Missing all-pool portfolio answer prompts: dataset={dataset_name}, "
            f"portfolio_id={portfolio_id}, num_docs={num_docs_to_fetch}, "
            f"expected_path={prompts_file}, checked_path={prompts_read_path}"
        )
    with open(prompts_read_path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"All-pool portfolio prompt payload must be a dict: "
            f"dataset={dataset_name}, portfolio_id={portfolio_id}, path={prompts_file}"
        )

    meta = payload.get("meta", {})
    if meta.get("dataset") not in {None, dataset_name}:
        raise ValueError(
            f"All-pool prompt dataset mismatch: expected={dataset_name}, "
            f"actual={meta.get('dataset')}, path={prompts_file}"
        )
    if meta.get("portfolio_id") not in {None, portfolio_id}:
        raise ValueError(
            f"All-pool prompt portfolio_id mismatch: expected={portfolio_id}, "
            f"actual={meta.get('portfolio_id')}, path={prompts_file}"
        )
    if meta.get("split") not in {None, "test"}:
        raise ValueError(f"All-pool prompt split must be test: path={prompts_file}")
    if meta.get("num_docs_to_fetch") is not None and int(meta["num_docs_to_fetch"]) != int(num_docs_to_fetch):
        raise ValueError(
            f"All-pool prompt num_docs mismatch: expected={num_docs_to_fetch}, "
            f"actual={meta.get('num_docs_to_fetch')}, path={prompts_file}"
        )
    if portfolio_size is not None and meta.get("portfolio_size") is not None:
        if int(meta["portfolio_size"]) != int(portfolio_size):
            raise ValueError(
                f"All-pool prompt portfolio_size mismatch: expected={portfolio_size}, "
                f"actual={meta.get('portfolio_size')}, path={prompts_file}. "
                "Rebuild prompts for the requested portfolio size."
            )

    portfolio_prompts = payload.get("portfolio_prompts", [])
    if not isinstance(portfolio_prompts, list):
        raise ValueError(f"portfolio_prompts must be a list: path={prompts_file}")
    for idx, prompt in enumerate(portfolio_prompts):
        if not isinstance(prompt, dict):
            raise ValueError(
                f"portfolio_prompts[{idx}] must be a dict: path={prompts_file}"
            )
        for key in ("question_idx", "system_prompt", "user_prompt"):
            if key not in prompt:
                raise ValueError(
                    f"portfolio_prompts[{idx}] is missing {key!r}: path={prompts_file}"
                )
        if "retriever_idx" not in prompt and "portfolio_rank" not in prompt:
            raise ValueError(
                f"portfolio_prompts[{idx}] is missing retriever_idx/portfolio_rank: "
                f"path={prompts_file}"
            )

    output_file = C.get_portfolio_union_answers_all(
        portfolio_id,
        dataset_name,
        llm_name,
        num_docs_to_fetch,
    )
    output_read_path = _artifact_read_path(output_file)
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)

    meta_out = {
        **meta,
        "llm_name": llm_name,
        "num_workers": max_workers,
        "prompts_path": prompts_file,
        "prompts_loaded_path": prompts_read_path,
        "answers_path": output_file,
    }

    answers = [None] * len(portfolio_prompts)
    if os.path.exists(output_read_path):
        try:
            with open(output_read_path, "rb") as f:
                existing_payload = pickle.load(f)
            existing_answers = (
                existing_payload.get("answers", [])
                if isinstance(existing_payload, dict)
                else existing_payload
            )
        except Exception as exc:
            print(
                f"[Portfolio union answers] Could not load existing answers from "
                f"{output_read_path}: {exc}",
                flush=True,
            )
            existing_answers = []

        if isinstance(existing_answers, list) and existing_answers:
            if len(existing_answers) == len(portfolio_prompts):
                answers = existing_answers
            else:
                n = min(len(existing_answers), len(portfolio_prompts))
                for idx in range(n):
                    answers[idx] = existing_answers[idx]

    def _normalize_answer_entry(idx, answer):
        prompt = portfolio_prompts[idx]
        if answer is None:
            return None
        if isinstance(answer, dict):
            entry = dict(answer)
        else:
            entry = {"response": str(answer), "error": None}
        for key in (
            "question_idx",
            "retriever_idx",
            "portfolio_rank",
            "selected_retriever",
            "question",
            "system_prompt",
            "user_prompt",
        ):
            if key in prompt:
                entry[key] = prompt[key]
        if "portfolio_rank" not in entry and "retriever_idx" in entry:
            entry["portfolio_rank"] = entry["retriever_idx"]
        if "retriever_idx" not in entry and "portfolio_rank" in entry:
            entry["retriever_idx"] = entry["portfolio_rank"]
        entry.setdefault("response", None)
        entry.setdefault("error", None)
        return entry

    for idx, answer in enumerate(answers):
        answers[idx] = _normalize_answer_entry(idx, answer)

    remaining_indices = [idx for idx, answer in enumerate(answers) if answer is None]
    print(
        f"[Portfolio union answers] Starting/resuming: dataset={dataset_name} "
        f"portfolio_id={portfolio_id} prompts={len(portfolio_prompts)} "
        f"done={len(portfolio_prompts) - len(remaining_indices)} "
        f"remaining={len(remaining_indices)} output={output_file}",
        flush=True,
    )

    def _save_checkpoint():
        with open(output_write_path, "wb") as f:
            pickle.dump(
                {
                    "meta": meta_out,
                    "answers": answers,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    if remaining_indices:
        def _call(idx):
            prompt = portfolio_prompts[idx]
            return llm.answer(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(remaining_indices),
            desc="Portfolio union answers",
            unit="prompt",
        ) as pbar:
            future_to_idx = {
                executor.submit(_call, idx): idx for idx in remaining_indices
            }
            since_last_checkpoint = 0

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                prompt = portfolio_prompts[idx]
                try:
                    response = future.result()
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)

                entry = dict(prompt)
                if "portfolio_rank" not in entry and "retriever_idx" in entry:
                    entry["portfolio_rank"] = entry["retriever_idx"]
                if "retriever_idx" not in entry and "portfolio_rank" in entry:
                    entry["retriever_idx"] = entry["portfolio_rank"]
                entry["response"] = response
                entry["error"] = error
                answers[idx] = entry

                pbar.update(1)
                since_last_checkpoint += 1
                if checkpoint_every and since_last_checkpoint >= checkpoint_every:
                    _save_checkpoint()
                    total_done = len([answer for answer in answers if answer is not None])
                    print(
                        f"[Portfolio union answers] Checkpoint saved to {output_file} "
                        f"({total_done}/{len(answers)} answered).",
                        flush=True,
                    )
                    since_last_checkpoint = 0

    _save_checkpoint()
    print(
        f"[Portfolio union answers] Completed "
        f"({len([answer for answer in answers if answer is not None])}/{len(answers)} answered).",
        flush=True,
    )
    if output_write_path != output_file:
        print(
            f"[Portfolio union answers] written_path={output_write_path}",
            flush=True,
        )
    return output_file

def answer_family_best_prompts_with_llm(
    portfolio_id=C.POOL_SET_ALL_IMPLEMENTED,
    dataset_name=None,
    family=C.DS,
    num_docs_to_fetch=4,
    max_k=5,
    llm=None,
    llm_name=None,
    max_workers=16,
    checkpoint_every=1000,
):
    """Answer saved family-best prompts and persist one answer artifact."""
    if dataset_name is None:
        raise ValueError("dataset_name is required.")
    if llm is None:
        raise ValueError("llm object is required.")
    if llm_name is None or str(llm_name).strip() == "":
        raise ValueError("llm_name is required.")
    family = _normalize_family_list([family])[0]

    context = (
        f"family-best answers dataset={dataset_name} portfolio_id={portfolio_id} "
        f"family={family} num_docs={num_docs_to_fetch} max_k={max_k} llm={llm_name}"
    )
    prompts_file = C.get_family_best_answer_prompts_test(
        portfolio_id,
        dataset_name,
        family,
        num_docs_to_fetch,
        max_k,
    )
    prompts_payload, prompts_read_path = _load_pickle_artifact(
        prompts_file,
        context=f"family-best answer prompts ({context})",
    )
    if not isinstance(prompts_payload, dict):
        raise ValueError(f"Family-best prompt payload must be a dict: path={prompts_file}")
    meta = prompts_payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    for key, expected in (
        ("dataset", dataset_name),
        ("portfolio_id", portfolio_id),
        ("family", family),
        ("split", "test"),
    ):
        actual = meta.get(key)
        if actual is not None and actual != expected:
            raise ValueError(
                f"Family-best prompt meta {key} mismatch: "
                f"expected={expected}, actual={actual}, path={prompts_file}"
            )
    for key, expected in (("num_docs_to_fetch", num_docs_to_fetch), ("max_k", max_k)):
        actual = meta.get(key)
        if actual is not None and int(actual) != int(expected):
            raise ValueError(
                f"Family-best prompt meta {key} mismatch: "
                f"expected={expected}, actual={actual}, path={prompts_file}"
            )

    answer_prompts = prompts_payload.get("answer_prompts", [])
    if not isinstance(answer_prompts, list):
        raise ValueError(f"answer_prompts must be a list: path={prompts_file}")
    for idx, prompt in enumerate(answer_prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"answer_prompts[{idx}] must be a dict: path={prompts_file}")
        for key in ("question_idx", "system_prompt", "user_prompt"):
            if key not in prompt:
                raise ValueError(f"answer_prompts[{idx}] is missing {key!r}: path={prompts_file}")

    output_file = C.get_family_best_answers_test(
        portfolio_id,
        dataset_name,
        family,
        llm_name,
        num_docs_to_fetch,
        max_k,
    )
    output_read_path = _artifact_read_path(output_file)
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)
    meta_out = {
        **meta,
        "schema": "family_best_answers",
        "schema_version": 1,
        "llm_name": llm_name,
        "num_workers": int(max_workers),
        "checkpoint_every": int(checkpoint_every),
        "prompts_path": prompts_file,
        "prompts_loaded_path": prompts_read_path,
        "answers_path": output_file,
    }

    answers = [None] * len(answer_prompts)
    if os.path.exists(output_read_path):
        try:
            with open(output_read_path, "rb") as f:
                existing_payload = pickle.load(f)
            existing_answers = (
                existing_payload.get("answers", [])
                if isinstance(existing_payload, dict)
                else existing_payload
            )
        except Exception as exc:
            print(
                f"[Family-best answers] Could not load existing answers from "
                f"{output_read_path}: {exc}",
                flush=True,
            )
            existing_answers = []
        if isinstance(existing_answers, list) and existing_answers:
            if len(existing_answers) == len(answer_prompts):
                answers = existing_answers
            else:
                n = min(len(existing_answers), len(answer_prompts))
                for idx in range(n):
                    answers[idx] = existing_answers[idx]

    def _normalize_answer_entry(idx, answer):
        prompt = answer_prompts[idx]
        if answer is None:
            return None
        if isinstance(answer, dict):
            entry = dict(answer)
        else:
            entry = {"response": str(answer), "error": None}
        for key in (
            "question_idx",
            "family",
            "selected_retriever",
            "question",
            "num_docs_used",
            "system_prompt",
            "user_prompt",
        ):
            if key in prompt:
                entry[key] = prompt[key]
        entry.setdefault("response", None)
        entry.setdefault("error", None)
        return entry

    for idx, answer in enumerate(answers):
        answers[idx] = _normalize_answer_entry(idx, answer)

    remaining_indices = [idx for idx, answer in enumerate(answers) if answer is None]
    print(
        f"[Family-best answers] Starting/resuming: dataset={dataset_name} "
        f"family={family} prompts={len(answer_prompts)} "
        f"done={len(answer_prompts) - len(remaining_indices)} "
        f"remaining={len(remaining_indices)} output={output_file}",
        flush=True,
    )

    def _save_checkpoint():
        with open(output_write_path, "wb") as f:
            pickle.dump(
                {
                    "meta": meta_out,
                    "answers": answers,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    if remaining_indices:
        def _call(idx):
            prompt = answer_prompts[idx]
            return llm.answer(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(remaining_indices),
            desc="Family-best answers",
            unit="prompt",
        ) as pbar:
            future_to_idx = {
                executor.submit(_call, idx): idx for idx in remaining_indices
            }
            since_last_checkpoint = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                prompt = answer_prompts[idx]
                try:
                    response = future.result()
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)
                entry = dict(prompt)
                entry["response"] = response
                entry["error"] = error
                answers[idx] = entry
                pbar.update(1)
                since_last_checkpoint += 1
                if checkpoint_every and since_last_checkpoint >= checkpoint_every:
                    _save_checkpoint()
                    total_done = len([answer for answer in answers if answer is not None])
                    print(
                        f"[Family-best answers] Checkpoint saved to {output_file} "
                        f"({total_done}/{len(answers)} answered).",
                        flush=True,
                    )
                    since_last_checkpoint = 0

    _save_checkpoint()
    print(
        f"[Family-best answers] Completed "
        f"({len([answer for answer in answers if answer is not None])}/{len(answers)} answered).",
        flush=True,
    )
    if output_write_path != output_file:
        print(f"[Family-best answers] written_path={output_write_path}", flush=True)
    return output_file

def answer_portfolio_router_judge_prompts_with_llm(
    portfolio_id,
    dataset_name,
    num_docs_to_fetch=4,
    portfolio_size=5,
    ell=2,
    run_id=None,
    answer_llm=None,
    judge_llm_name=None,
    judge_llm=None,
    split="test",
    max_workers=16,
    checkpoint_every=1000,
):
    split, k, ell = _validate_router_judge_common(
        portfolio_id=portfolio_id,
        dataset_name=dataset_name,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=portfolio_size,
        ell=ell,
        run_id=run_id,
        answer_llm=answer_llm,
        split=split,
        judge_llm_name=judge_llm_name,
    )
    if ell == 1:
        raise ValueError(
            "ell=1 uses direct router top-1 answer selection and does not call a judge LLM. "
            "Score direct top-1 answers from the saved portfolio-router predictions."
        )
    if judge_llm_name is None or str(judge_llm_name).strip() == "":
        raise ValueError("judge_llm_name is required for portfolio-router judge answering.")
    if judge_llm is None:
        raise ValueError("judge_llm object is required for portfolio-router judge answering.")

    context = _portfolio_router_judge_context(
        dataset_name=dataset_name,
        portfolio_id=portfolio_id,
        num_docs_to_fetch=num_docs_to_fetch,
        portfolio_size=k,
        ell=ell,
        split=split,
        run_id=run_id,
        answer_llm=answer_llm,
        judge_llm_name=judge_llm_name,
    )
    prompts_file = C.get_portfolio_router_judge_prompts(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
        k,
        ell,
        split,
        run_id,
        answer_llm,
    )
    prompts_payload, prompts_read_path = _load_pickle_artifact(
        prompts_file,
        context=f"portfolio-router judge prompts ({context})",
    )
    if not isinstance(prompts_payload, dict):
        raise ValueError(f"Portfolio-router judge prompt payload must be a dict: path={prompts_file}")
    meta = prompts_payload.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    _validate_optional_payload_field(meta, "dataset", dataset_name, path=prompts_file, context="Judge prompt meta")
    _validate_optional_payload_field(meta, "portfolio_id", portfolio_id, path=prompts_file, context="Judge prompt meta")
    _validate_optional_payload_field(meta, "split", split, path=prompts_file, context="Judge prompt meta")
    _validate_optional_payload_field(meta, "run_id", run_id, path=prompts_file, context="Judge prompt meta")
    _validate_optional_payload_field(meta, "answer_llm", answer_llm, path=prompts_file, context="Judge prompt meta")
    _validate_optional_int_payload_field(meta, ("num_docs_to_fetch", "num_docs"), num_docs_to_fetch, path=prompts_file, context="Judge prompt meta")
    _validate_optional_int_payload_field(meta, ("portfolio_size", "k"), k, path=prompts_file, context="Judge prompt meta")
    _validate_optional_int_payload_field(meta, ("ell",), ell, path=prompts_file, context="Judge prompt meta")

    judge_prompts = prompts_payload.get("judge_prompts", [])
    if not isinstance(judge_prompts, list):
        raise ValueError(f"judge_prompts must be a list: path={prompts_file}")
    for idx, prompt in enumerate(judge_prompts):
        if not isinstance(prompt, dict):
            raise ValueError(f"judge_prompts[{idx}] must be a dict: path={prompts_file}")
        for key in ("question_idx", "prediction_row_idx", "system_prompt", "user_prompt"):
            if key not in prompt:
                raise ValueError(f"judge_prompts[{idx}] is missing {key!r}: path={prompts_file}")

    output_file = C.get_portfolio_router_judge_answers(
        portfolio_id,
        dataset_name,
        num_docs_to_fetch,
        k,
        ell,
        split,
        run_id,
        answer_llm,
        judge_llm_name,
    )
    output_read_path = _artifact_read_path(output_file)
    output_write_path = _artifact_write_path(output_file)
    Path(output_write_path).parent.mkdir(parents=True, exist_ok=True)
    meta_out = {
        **meta,
        "schema": "portfolio_router_judge_answers",
        "schema_version": 1,
        "judge_llm_name": judge_llm_name,
        "num_workers": int(max_workers),
        "checkpoint_every": int(checkpoint_every),
        "prompts_path": prompts_file,
        "prompts_loaded_path": prompts_read_path,
        "answers_path": output_file,
    }

    answers = [None] * len(judge_prompts)
    if os.path.exists(output_read_path):
        try:
            with open(output_read_path, "rb") as f:
                existing_payload = pickle.load(f)
            existing_answers = (
                existing_payload.get("answers", [])
                if isinstance(existing_payload, dict)
                else existing_payload
            )
        except Exception as exc:
            print(
                f"[Portfolio router judge answers] Could not load existing answers from "
                f"{output_read_path}: {exc}",
                flush=True,
            )
            existing_answers = []
        if isinstance(existing_answers, list) and existing_answers:
            if len(existing_answers) == len(judge_prompts):
                answers = existing_answers
            else:
                n = min(len(existing_answers), len(judge_prompts))
                for idx in range(n):
                    answers[idx] = existing_answers[idx]

    def _normalize_judge_answer_entry(idx, answer):
        if answer is None:
            return None
        prompt = judge_prompts[idx]
        if isinstance(answer, dict):
            entry = dict(answer)
        else:
            entry = {"response": str(answer), "error": None}
        for key, value in prompt.items():
            entry[key] = value
        entry.setdefault("response", None)
        entry.setdefault("error", None)
        return entry

    for idx, answer in enumerate(answers):
        answers[idx] = _normalize_judge_answer_entry(idx, answer)

    remaining_indices = [idx for idx, answer in enumerate(answers) if answer is None]
    print(
        f"[Portfolio router judge answers] Starting/resuming: {context} "
        f"prompts={len(judge_prompts)} done={len(judge_prompts) - len(remaining_indices)} "
        f"remaining={len(remaining_indices)} output={output_file}",
        flush=True,
    )

    def _save_checkpoint():
        with open(output_write_path, "wb") as f:
            pickle.dump(
                {
                    "meta": meta_out,
                    "answers": answers,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    if remaining_indices:
        def _call(idx):
            prompt = judge_prompts[idx]
            return judge_llm.answer(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(remaining_indices),
            desc="Portfolio router judge answers",
            unit="prompt",
        ) as pbar:
            future_to_idx = {
                executor.submit(_call, idx): idx for idx in remaining_indices
            }
            since_last_checkpoint = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                prompt = judge_prompts[idx]
                try:
                    response = future.result()
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)

                entry = dict(prompt)
                entry["response"] = response
                entry["error"] = error
                answers[idx] = entry
                pbar.update(1)
                since_last_checkpoint += 1
                if checkpoint_every and since_last_checkpoint >= checkpoint_every:
                    _save_checkpoint()
                    total_done = len([answer for answer in answers if answer is not None])
                    print(
                        f"[Portfolio router judge answers] Checkpoint saved to {output_file} "
                        f"({total_done}/{len(answers)} answered).",
                        flush=True,
                    )
                    since_last_checkpoint = 0

    _save_checkpoint()
    print(
        f"[Portfolio router judge answers] Completed "
        f"({len([answer for answer in answers if answer is not None])}/{len(answers)} answered).",
        flush=True,
    )
    if output_write_path != output_file:
        print(
            f"[Portfolio router judge answers] written_path={output_write_path}",
            flush=True,
        )
    return output_file

def answer_baseline_prompts_with_llm(
    dataset_name,
    num_docs_to_fetch,
    llm,
    llm_name,
    max_workers=64,
    checkpoint_every=1000,
):
    """
        Consumes baseline prompts (no retrieval + DS(0,1)) and generates answers
        using the provided LLM. Saves both answer sets to a single baseline file.
    """
    prompts_file = C.get_baseline_answer_prompts(dataset_name)
    with open(prompts_file, "rb") as f:
        payload = pickle.load(f)

    meta = payload.get("meta", {})
    no_prompts = payload.get("no_retrieval_prompts", [])
    naive_prompts = payload.get("naive_retrieval_prompts", [])

    output_file = C.get_answers_baseline(dataset_name, num_docs_to_fetch, llm_name)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    meta_out = {
        **meta,
        "llm_name": llm_name,
        "num_workers": max_workers,
        "num_docs_to_fetch": num_docs_to_fetch,
    }

    existing_payload = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "rb") as f:
                existing_payload = pickle.load(f) or {}
        except Exception as exc:
            print(
                f"[baseline] Could not load existing answers from {output_file}: {exc}",
                flush=True,
            )
            existing_payload = {}

    def _seed_answers(key, prompts):
        answers = [None] * len(prompts)
        existing_answers = existing_payload.get(key, [])
        if existing_answers:
            if len(existing_answers) == len(prompts):
                answers = existing_answers
            else:
                n = min(len(existing_answers), len(prompts))
                for i in range(n):
                    answers[i] = existing_answers[i]
        return answers

    no_answers = _seed_answers("no_retrieval_answers", no_prompts)
    naive_answers = _seed_answers("naive_retrieval_answers", naive_prompts)

    def _save_checkpoint():
        with open(output_file, "wb") as f:
            pickle.dump(
                {
                    "meta": meta_out,
                    "no_retrieval_answers": no_answers,
                    "naive_retrieval_answers": naive_answers,
                },
                f,
            )

    def _run_with_checkpoint(prompts, answers, label, baseline_tag):
        if not prompts:
            return

        remaining_indices = [i for i, ans in enumerate(answers) if ans is None]
        if not remaining_indices:
            print(f"[{label}] All prompts already answered ({len(prompts)} total).", flush=True)
            return

        def _call(idx):
            prompt = prompts[idx]
            return llm.answer(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        total_done_before = len([a for a in answers if a is not None])
        print(
            f"[{label}] Starting/resuming: {len(prompts)} prompts "
            f"({total_done_before} done, {len(remaining_indices)} remaining).",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(remaining_indices),
            desc=label,
            unit="prompt",
        ) as pbar:
            future_to_idx = {
                executor.submit(_call, idx): idx for idx in remaining_indices
            }
            since_last_checkpoint = 0

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                prompt = prompts[idx]
                try:
                    response = future.result()
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)

                entry = {
                    "question_idx": prompt["question_idx"],
                    "system_prompt": prompt["system_prompt"],
                    "user_prompt": prompt["user_prompt"],
                    "response": response,
                    "error": error,
                    "baseline": baseline_tag,
                }
                answers[idx] = entry
                pbar.update(1)
                since_last_checkpoint += 1

                if checkpoint_every and since_last_checkpoint >= checkpoint_every:
                    _save_checkpoint()
                    total_done = len([a for a in answers if a is not None])
                    print(
                        f"[{label}] Checkpoint saved to {output_file} "
                        f"({total_done}/{len(answers)} answered).",
                        flush=True,
                    )
                    since_last_checkpoint = 0

        _save_checkpoint()
        print(
            f"[{label}] Completed all prompts "
            f"({len([a for a in answers if a is not None])}/{len(answers)} answered).",
            flush=True,
        )

    print(
        f"Answering {len(no_prompts)} no-retrieval prompts and "
        f"{len(naive_prompts)} naive-retrieval prompts...",
        flush=True,
    )

    _run_with_checkpoint(
        no_prompts,
        no_answers,
        "Baseline no-retrieval answers",
        "no_retrieval",
    )
    _run_with_checkpoint(
        naive_prompts,
        naive_answers,
        "Baseline naive-retrieval answers",
        "naive_retrieval",
    )

    return output_file

def answer_selector_prompts_with_llm(
    dataset_name,
    retriever,
    num_docs_to_fetch,
    llm,
    llm_name,
    max_workers=16,
    checkpoint_every=1000,
    embedder=None,
):
    """
        Consumes the saved selector prompts for a specific answer LLM and
        generates selector outputs using the provided selector LLM.

        Saves results to C.get_answers_llm_selector(dataset, retriever, llm_name, num_docs).
    """
    embedder_key = _artifact_embedder_for_retriever(retriever, embedder)
    prompts_file = C.get_selector_prompts(dataset_name, retriever, llm_name, num_docs_to_fetch, embedder=embedder_key)
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(
            f"Selector prompts file not found: {prompts_file}. "
            "Run build_selector_prompts first."
        )

    with open(prompts_file, "rb") as f:
        payload = pickle.load(f)

    meta = payload.get("meta", {})
    selector_prompts = payload.get("selector_prompts", [])

    output_file = C.get_answers_llm_selector(dataset_name, retriever, llm_name, num_docs_to_fetch, embedder=embedder_key)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    meta_out = {
        **meta,
        "selector_llm_name": llm_name,
        "num_workers": max_workers,
    }

    def _run_with_checkpoint(prompts, label, output_path, allowed_indices=None):
        """
            Run LLM over the given prompts with checkpointing support.
            If output_path exists, resume from the existing answers and
            only process missing entries. Saves a checkpoint every
            `checkpoint_every` completed answers.
        """
        if not prompts:
            return []

        # Initialize answers list, seeding from any existing file.
        answers = [None] * len(prompts)
        if os.path.exists(output_path):
            try:
                with open(output_path, "rb") as f:
                    existing_payload = pickle.load(f)
                existing_answers = existing_payload.get("answers", [])
            except Exception as exc:
                print(f"[{label}] Could not load existing answers from {output_path}: {exc}", flush=True)
                existing_answers = []

            if existing_answers:
                if len(existing_answers) == len(prompts):
                    answers = existing_answers
                else:
                    n = min(len(existing_answers), len(prompts))
                    for i in range(n):
                        answers[i] = existing_answers[i]

        if allowed_indices is None:
            candidate_indices = range(len(answers))
        else:
            candidate_indices = allowed_indices
        remaining_indices = [i for i in candidate_indices if answers[i] is None]
        if not remaining_indices:
            print(f"[{label}] All prompts already answered ({len(prompts)} total).", flush=True)
            return answers

        def _call(idx):
            prompt = prompts[idx]
            return llm.answer(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        total_done_before = len([a for a in answers if a is not None])
        total_candidates = len(candidate_indices) if allowed_indices is not None else len(prompts)
        print(
            f"[{label}] Starting/resuming: {total_candidates} selected prompts "
            f"({total_done_before} done, {len(remaining_indices)} remaining).",
            flush=True,
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(
            total=len(remaining_indices),
            desc=label,
            unit="prompt",
        ) as pbar:
            future_to_idx = {
                executor.submit(_call, idx): idx for idx in remaining_indices
            }
            since_last_checkpoint = 0

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                prompt = prompts[idx]
                try:
                    response = future.result()
                    error = None
                except Exception as exc:
                    response = None
                    error = str(exc)

                entry = {
                    "question_idx": prompt["question_idx"],
                    "subset_mask": prompt["subset_mask"],
                    "subset_retrievers": prompt["subset_retrievers"],
                    "system_prompt": prompt["system_prompt"],
                    "user_prompt": prompt["user_prompt"],
                    "response": response,
                    "error": error,
                }

                answers[idx] = entry
                pbar.update(1)
                since_last_checkpoint += 1

                if checkpoint_every and since_last_checkpoint >= checkpoint_every:
                    with open(output_path, "wb") as f:
                        pickle.dump({"meta": meta_out, "answers": answers}, f)
                    total_done = len([a for a in answers if a is not None])
                    print(
                        f"[{label}] Checkpoint saved to {output_path} "
                        f"({total_done}/{len(answers)} answered).",
                        flush=True,
                    )
                    since_last_checkpoint = 0

        with open(output_path, "wb") as f:
            pickle.dump({"meta": meta_out, "answers": answers}, f)
        print(
            f"[{label}] Completed all selector prompts "
            f"({len([a for a in answers if a is not None])}/{len(answers)} answered).",
            flush=True,
        )
        return answers

    print(
        f"Answering {len(selector_prompts)} selector prompts "
        f"for dataset={dataset_name}, retriever={retriever}, num_docs={num_docs_to_fetch}...",
        flush=True,
    )

    _run_with_checkpoint(
        selector_prompts,
        "Selector answers",
        output_file,
    )

    return output_file

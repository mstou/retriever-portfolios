from __future__ import annotations

import os
import pickle
import re
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import TypeAlias

import numpy as np

import constants as C
from data_classes import Dataset, TextUnit
from models import OpenAI_LLM
from prompts import (
    graph_entity_extraction_system_prompt,
    graph_entity_extraction_user_prompt,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is expected in this repo
    tqdm = None


ChunkKey: TypeAlias = tuple[str | int, int]
DatasetLike: TypeAlias = Dataset | str
EntityExtractionPrompt: TypeAlias = dict[str, object]
EntityExtractionResult: TypeAlias = dict[str, object]

_WHITESPACE_RE = re.compile(r"\s+")
_SURROUNDING_PUNCTUATION = "\"'`“”‘’.,;:!?()[]{}<>"
_ENTITY_BLOCK_RE = re.compile(r"<entities>\s*(.*?)\s*</entities>", re.DOTALL | re.IGNORECASE)

def normalize_entity(text: str) -> str:
    """Return a deterministic, lightly normalized entity string."""
    if not isinstance(text, str):
        return ""

    normalized = text.lower().strip()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = normalized.strip(_SURROUNDING_PUNCTUATION)
    normalized = normalized.strip()

    if not normalized:
        return ""
    if not any(char.isalnum() for char in normalized):
        return ""
    return normalized


def _merge_entities(*entity_groups: list[str]) -> list[str]:
    """Merge entity lists while preserving first-seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for entity_group in entity_groups:
        for entity in entity_group:
            normalized = normalize_entity(entity)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def _get_document_title(document) -> str:
    """Return the stripped title stored in document metadata, if any."""
    metadata = getattr(document, "metadata", None) or {}
    title = metadata.get("title", "")
    return title.strip() if isinstance(title, str) else ""


def _default_graph_index_checkpoint_path(dataset_name: str) -> Path:
    return Path(C.get_graph_entity_extraction_results_path(dataset_name))


def _default_graph_entity_extraction_prompts_path(dataset_name: str) -> Path:
    return Path(C.get_graph_entity_extraction_prompts_path(dataset_name))


def _default_graph_entity_extraction_results_path(dataset_name: str) -> Path:
    return Path(C.get_graph_entity_extraction_results_path(dataset_name))


def _normalize_question_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in {"train", "test"}:
        raise ValueError(f"question split must be 'train' or 'test', got {split!r}")
    return normalized


def _default_graph_question_entity_extraction_prompts_path(dataset_name: str, split: str) -> Path:
    return Path(C.get_graph_question_entity_extraction_prompts_path(dataset_name, _normalize_question_split(split)))


def _default_graph_question_entity_extraction_results_path(dataset_name: str, split: str) -> Path:
    return Path(C.get_graph_question_entity_extraction_results_path(dataset_name, _normalize_question_split(split)))


def _slice_bounds(total: int, slice_index: int, num_slices: int) -> tuple[int, int]:
    if num_slices < 1:
        raise ValueError(f"num_slices must be >= 1, got {num_slices}")
    if slice_index < 0 or slice_index >= num_slices:
        raise ValueError(f"slice_index must be in [0, {num_slices}), got {slice_index}")

    start = total * slice_index // num_slices
    end = total * (slice_index + 1) // num_slices
    return start, end


def _default_graph_entity_extraction_results_slice_path(
    dataset_name: str,
    slice_index: int,
    num_slices: int,
) -> Path:
    base_path = _default_graph_entity_extraction_results_path(dataset_name)
    return base_path.with_name(
        f"{base_path.stem}_slice_{slice_index}_of_{num_slices}{base_path.suffix}"
    )


def _count_prompt_kinds(prompts: list[EntityExtractionPrompt]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for prompt in prompts:
        counts[str(prompt["kind"])] += 1
    return dict(counts)


def _print_progress(label: str, current: int, total: int, every: int = 10000) -> None:
    if current == total or current % every == 0:
        print(f"{label}: {current}/{total}", flush=True)


class EntityExtractor:
    """Minimal LLM-backed named-entity extractor with lazy client initialization."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        allowed_entity_labels: set[str] | None = None,
        max_tokens: int = 256,
    ):
        self.model_name = model_name or C.LLM_DIR[C.LLAMA70B]
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.base_url = base_url or C.LLM_BASE_URL[C.LLAMA70B]
        self.allowed_entity_labels = set(allowed_entity_labels) if allowed_entity_labels else None
        self.max_tokens = max_tokens
        self._llm = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_llm"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._llm = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm

        try:
            self._llm = OpenAI_LLM(
                model_name=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except Exception as exc:  # pragma: no cover - depends on local endpoint configuration
            raise RuntimeError(
                "Failed to initialize LLM entity extractor. "
                f"model_name={self.model_name!r}, base_url={self.base_url!r}. "
                "Ensure the configured OpenAI-compatible endpoint is running and reachable."
            ) from exc
        return self._llm

    def _parse_entities(self, response_text: str) -> list[str]:
        if not response_text or not isinstance(response_text, str):
            return []

        match = _ENTITY_BLOCK_RE.search(response_text)
        if match is None:
            return []

        entities: list[str] = []
        seen: set[str] = set()
        for line in match.group(1).splitlines():
            normalized = normalize_entity(line.strip())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            entities.append(normalized)
        return entities

    def build_prompt(self, text: str) -> tuple[str, str]:
        """Return the entity-extraction prompt pair for a text payload."""
        return (
            graph_entity_extraction_system_prompt(),
            graph_entity_extraction_user_prompt(text),
        )

    def parse_response(self, response_text: str) -> list[str]:
        """Parse one raw model response into normalized entity strings."""
        return self._parse_entities(response_text)

    def answer_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Run one entity-extraction prompt through the configured LLM."""
        return self._get_llm().answer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=0.0,
            top_p=1.0,
        )

    def extract_entities(self, text: str) -> list[str]:
        """Extract normalized named entities from text in deterministic order."""
        if not text or not isinstance(text, str):
            return []

        system_prompt, user_prompt = self.build_prompt(text)
        response_text = self.answer_prompt(system_prompt, user_prompt)
        return self._parse_entities(response_text)


class GraphIndex:
    """Dataset-level entity/chunk sidecar index over existing TextUnits."""

    def __init__(
        self,
        entity_extractor: EntityExtractor | None = None,
        dataset_name: str | None = None,
    ):
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.entity_to_chunk_keys: dict[str, list[ChunkKey]] = {}
        self.chunk_to_entities: dict[ChunkKey, list[str]] = {}
        self.chunk_lookup: dict[ChunkKey, TextUnit] = {}
        self.all_chunk_keys: list[ChunkKey] = []
        self.chunk_key_to_embedding_row: dict[ChunkKey, int] = {}
        self.chunk_embeddings: dict[str, np.ndarray] = {}
        self.chunk_embedding_meta: dict[str, dict[str, object]] = {}
        self.num_chunks = 0
        self.dataset_name = dataset_name

    def _resolve_dataset(self, dataset_or_name: DatasetLike) -> Dataset:
        """Resolve a Dataset object from a Dataset instance or dataset name."""
        if isinstance(dataset_or_name, Dataset):
            print(
                f"[graph index] using provided Dataset object: dataset={dataset_or_name.dataset_name}, "
                f"documents={len(dataset_or_name.documents)}",
                flush=True,
            )
            return dataset_or_name
        if isinstance(dataset_or_name, str):
            from experiment_utils import ensure_chunk_cache, load_chunk_cache
            from text_processing import ChunkedCorpusCache

            chunk_cache_dir = C.get_chunk_cache_dir(dataset_or_name)
            if ChunkedCorpusCache.exists(chunk_cache_dir):
                print(
                    f"[graph index] loading existing chunk cache for dataset={dataset_or_name} "
                    f"from {chunk_cache_dir}",
                    flush=True,
                )
                dataset, _metadata = load_chunk_cache(dataset_or_name)
                print(
                    f"[graph index] loaded chunk cache for dataset={dataset_or_name}: "
                    f"documents={len(dataset.documents)}",
                    flush=True,
                )
                return dataset
            print(
                f"[graph index] chunk cache missing for dataset={dataset_or_name}; creating it now",
                flush=True,
            )
            return ensure_chunk_cache(dataset_or_name)
        raise TypeError(
            "GraphIndex.build expected a Dataset object or dataset name string, "
            f"got {type(dataset_or_name).__name__}"
        )

    def prepare_entity_extraction_prompts(
        self,
        dataset_or_name: DatasetLike,
    ) -> tuple[Dataset, dict[ChunkKey, TextUnit], list[ChunkKey], list[EntityExtractionPrompt]]:
        """Prepare title/chunk entity-extraction prompts for one dataset."""
        dataset = self._resolve_dataset(dataset_or_name)
        print(
            f"[graph index] gathering text units for dataset={dataset.dataset_name}",
            flush=True,
        )
        dataset.gather_all_text_units()
        print(
            f"[graph index] dataset={dataset.dataset_name} has "
            f"{len(dataset.documents)} documents and {len(dataset.all_text_units)} chunks",
            flush=True,
        )

        chunk_lookup: dict[ChunkKey, TextUnit] = {}
        all_chunk_keys: list[ChunkKey] = []
        prompts: list[EntityExtractionPrompt] = []
        titles_with_text = 0

        document_iterator = dataset.documents
        if tqdm is not None:
            document_iterator = tqdm(
                document_iterator,
                total=len(dataset.documents),
                desc=f"Preparing title prompts [{dataset.dataset_name}]",
                unit="doc",
            )

        for doc_idx, document in enumerate(document_iterator, start=1):
            title = _get_document_title(document)
            if title:
                titles_with_text += 1
                system_prompt, user_prompt = self.entity_extractor.build_prompt(title)
                prompts.append(
                    {
                        "kind": "title",
                        "doc_id": document.doc_id,
                        "chunk_id": None,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    }
                )
            _print_progress(
                f"[graph index] title prompt scan [{dataset.dataset_name}]",
                doc_idx,
                len(dataset.documents),
            )

        chunk_iterator = dataset.all_text_units
        if tqdm is not None:
            chunk_iterator = tqdm(
                chunk_iterator,
                total=len(dataset.all_text_units),
                desc=f"Preparing chunk prompts [{dataset.dataset_name}]",
                unit="chunk",
            )

        for chunk_idx, text_unit in enumerate(chunk_iterator, start=1):
            chunk_key = (text_unit.doc_id, text_unit.chunk_id)
            chunk_lookup[chunk_key] = text_unit
            all_chunk_keys.append(chunk_key)

            system_prompt, user_prompt = self.entity_extractor.build_prompt(text_unit.text)
            prompts.append(
                {
                    "kind": "chunk",
                    "doc_id": text_unit.doc_id,
                    "chunk_id": text_unit.chunk_id,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            _print_progress(
                f"[graph index] chunk prompt prep [{dataset.dataset_name}]",
                chunk_idx,
                len(dataset.all_text_units),
            )

        print(
            f"[graph index] prepared prompts for dataset={dataset.dataset_name}: "
            f"title_prompts={titles_with_text}, chunk_prompts={len(all_chunk_keys)}, total_prompts={len(prompts)}",
            flush=True,
        )
        return dataset, chunk_lookup, all_chunk_keys, prompts

    def _build_entity_extraction_prompt_payload(
        self,
        dataset: Dataset,
        all_chunk_keys: list[ChunkKey],
        prompts: list[EntityExtractionPrompt],
    ) -> dict[str, object]:
        """Build the serialized prompt payload consumed by entity-extraction jobs."""
        prompt_kind_counts = _count_prompt_kinds(prompts)
        return {
            "meta": {
                "dataset_name": dataset.dataset_name,
                "chunk_cache_dir": C.get_chunk_cache_dir(dataset.dataset_name),
                "num_documents": len(dataset.documents),
                "num_chunks": len(all_chunk_keys),
                "num_prompts": len(prompts),
                "prompt_kind_counts": prompt_kind_counts,
            },
            "all_chunk_keys": all_chunk_keys,
            "prompts": prompts,
        }

    def save_entity_extraction_prompt_payload(
        self,
        payload: dict[str, object],
        output_path: str | Path,
    ) -> Path:
        """Persist prepared entity-extraction prompts to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @staticmethod
    def load_entity_extraction_prompt_payload(prompt_path: str | Path) -> dict[str, object]:
        """Load a prepared entity-extraction prompt payload."""
        with Path(prompt_path).open("rb") as f:
            return pickle.load(f)

    @staticmethod
    def _questions_path(dataset_name: str, split: str) -> Path:
        split = _normalize_question_split(split)
        if split == "train":
            return Path(C.get_questions_train(dataset_name))
        return Path(C.get_questions_test(dataset_name))

    def _load_questions_for_split(self, dataset_name: str, split: str):
        split = _normalize_question_split(split)
        questions_path = self._questions_path(dataset_name, split)
        if not questions_path.exists():
            print(
                f"[graph index] question cache missing for dataset={dataset_name} split={split}; "
                "creating train/test question split now",
                flush=True,
            )
            from experiment_utils import questions_train_test_split

            questions_train_test_split(dataset_name)
        with questions_path.open("rb") as f:
            return pickle.load(f), questions_path

    def prepare_question_entity_extraction_prompts(
        self,
        dataset_name: str,
        split: str,
    ) -> dict[str, object]:
        """Prepare entity-extraction prompts for cached train/test questions."""
        split = _normalize_question_split(split)
        questions, questions_path = self._load_questions_for_split(dataset_name, split)
        prompts: list[EntityExtractionPrompt] = []
        for question_idx, question_payload in enumerate(questions.questions):
            question_text = question_payload.get("question", "")
            system_prompt, user_prompt = self.entity_extractor.build_prompt(question_text)
            prompts.append(
                {
                    "kind": "question",
                    "dataset_name": dataset_name,
                    "split": split,
                    "question_id": question_idx,
                    "question": question_text,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            _print_progress(
                f"[graph index] prepare question entity prompts [{dataset_name} {split}]",
                question_idx + 1,
                len(questions.questions),
            )

        return {
            "meta": {
                "dataset_name": dataset_name,
                "split": split,
                "questions_path": str(questions_path),
                "num_questions": len(questions.questions),
                "num_prompts": len(prompts),
            },
            "prompts": prompts,
        }

    def prepare_and_save_question_entity_extraction_prompts(
        self,
        dataset_name: str,
        split: str,
        *,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Prepare and save entity-extraction prompts for one question split."""
        split = _normalize_question_split(split)
        resolved_output_path = (
            Path(output_path)
            if output_path is not None
            else _default_graph_question_entity_extraction_prompts_path(dataset_name, split)
        )
        if resolved_output_path.exists() and not overwrite:
            print(
                f"[graph index] question entity prompt file already exists for "
                f"dataset={dataset_name} split={split}: {resolved_output_path}",
                flush=True,
            )
            return resolved_output_path

        prompt_payload = self.prepare_question_entity_extraction_prompts(dataset_name, split)
        saved_path = self.save_entity_extraction_prompt_payload(prompt_payload, resolved_output_path)
        print(
            f"[graph index] saved question entity prompt file for dataset={dataset_name} split={split}: "
            f"path={saved_path}, prompts={len(prompt_payload['prompts'])}",
            flush=True,
        )
        return saved_path

    def answer_saved_question_entity_extraction_prompts(
        self,
        dataset_name: str,
        split: str,
        *,
        prompt_path: str | Path | None = None,
        output_path: str | Path | None = None,
        max_workers: int = 16,
        checkpoint_every: int = 500,
    ) -> Path:
        """Answer saved train/test question entity-extraction prompts."""
        split = _normalize_question_split(split)
        resolved_prompt_path = (
            Path(prompt_path)
            if prompt_path is not None
            else _default_graph_question_entity_extraction_prompts_path(dataset_name, split)
        )
        if not resolved_prompt_path.exists():
            self.prepare_and_save_question_entity_extraction_prompts(dataset_name, split)
        prompt_payload = self.load_entity_extraction_prompt_payload(resolved_prompt_path)
        prompts = prompt_payload["prompts"]
        resolved_output_path = (
            Path(output_path)
            if output_path is not None
            else _default_graph_question_entity_extraction_results_path(dataset_name, split)
        )
        label = f"Question entity extraction [{dataset_name} {split}]"
        print(
            f"[graph index] answering saved question entity prompts for dataset={dataset_name} split={split}: "
            f"prompt_path={resolved_prompt_path}, output_path={resolved_output_path}, prompts={len(prompts)}",
            flush=True,
        )
        self.run_entity_extraction_prompts(
            prompts,
            checkpoint_path=resolved_output_path,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
            label=label,
            result_meta={
                "dataset_name": dataset_name,
                "split": split,
                "prompt_path": str(resolved_prompt_path),
                "question_entity_extraction": True,
            },
        )
        return resolved_output_path

    def prepare_and_save_entity_extraction_prompts(
        self,
        dataset_or_name: DatasetLike,
        *,
        output_path: str | Path | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Prepare all title/chunk entity-extraction prompts for one dataset and save them."""
        dataset_name = dataset_or_name if isinstance(dataset_or_name, str) else dataset_or_name.dataset_name
        resolved_output_path = (
            Path(output_path)
            if output_path is not None
            else _default_graph_entity_extraction_prompts_path(dataset_name)
        )
        if resolved_output_path.exists() and not overwrite:
            print(
                f"[graph index] entity prompt file already exists for dataset={dataset_name}: "
                f"{resolved_output_path}",
                flush=True,
            )
            return resolved_output_path

        dataset, _chunk_lookup, all_chunk_keys, prompts = self.prepare_entity_extraction_prompts(dataset_or_name)
        payload = self._build_entity_extraction_prompt_payload(dataset, all_chunk_keys, prompts)
        saved_path = self.save_entity_extraction_prompt_payload(payload, resolved_output_path)
        print(
            f"[graph index] saved entity prompt file for dataset={dataset.dataset_name}: "
            f"path={saved_path}, prompts={len(prompts)}, chunks={len(all_chunk_keys)}",
            flush=True,
        )
        return saved_path

    def _load_chunk_lookup_for_prompt_payload(
        self,
        dataset_or_name: DatasetLike,
        prompt_payload: dict[str, object],
    ) -> tuple[Dataset, dict[ChunkKey, TextUnit], list[ChunkKey]]:
        """Load chunk lookup from the chunk cache."""
        dataset = self._resolve_dataset(dataset_or_name)
        print(
            f"[graph index] loading chunk lookup for dataset={dataset.dataset_name}",
            flush=True,
        )
        dataset.gather_all_text_units()

        chunk_lookup: dict[ChunkKey, TextUnit] = {}
        all_chunk_keys: list[ChunkKey] = []
        chunk_iterator = dataset.all_text_units
        if tqdm is not None:
            chunk_iterator = tqdm(
                dataset.all_text_units,
                total=len(dataset.all_text_units),
                desc=f"Loading chunk lookup [{dataset.dataset_name}]",
                unit="chunk",
            )

        for chunk_idx, text_unit in enumerate(chunk_iterator, start=1):
            chunk_key = (text_unit.doc_id, text_unit.chunk_id)
            chunk_lookup[chunk_key] = text_unit
            all_chunk_keys.append(chunk_key)
            _print_progress(
                f"[graph index] chunk lookup [{dataset.dataset_name}]",
                chunk_idx,
                len(dataset.all_text_units),
            )

        return dataset, chunk_lookup, prompt_payload.get("all_chunk_keys", all_chunk_keys)

    def answer_saved_entity_extraction_prompts(
        self,
        dataset_name: str,
        *,
        prompt_path: str | Path | None = None,
        output_path: str | Path | None = None,
        max_workers: int = 16,
        checkpoint_every: int = 500,
        slice_index: int | None = None,
        num_slices: int | None = None,
    ) -> Path:
        """Read prepared entity prompts from disk and answer them into a result file."""
        resolved_prompt_path = (
            Path(prompt_path)
            if prompt_path is not None
            else _default_graph_entity_extraction_prompts_path(dataset_name)
        )
        prompt_payload = self.load_entity_extraction_prompt_payload(resolved_prompt_path)
        prompts = prompt_payload["prompts"]

        prompt_start = 0
        prompt_end = len(prompts)
        label = f"Graph entity extraction [{dataset_name}]"
        if slice_index is not None or num_slices is not None:
            if slice_index is None or num_slices is None:
                raise ValueError("slice_index and num_slices must be provided together")
            prompt_start, prompt_end = _slice_bounds(len(prompts), slice_index, num_slices)
            prompts = prompts[prompt_start:prompt_end]
            label = f"Graph entity extraction [{dataset_name} slice {slice_index}/{num_slices}]"

        resolved_output_path = (
            Path(output_path)
            if output_path is not None
            else _default_graph_entity_extraction_results_slice_path(dataset_name, slice_index, num_slices)
            if slice_index is not None and num_slices is not None
            else _default_graph_entity_extraction_results_path(dataset_name)
        )
        print(
            f"[graph index] answering saved entity prompts for dataset={dataset_name}: "
            f"prompt_path={resolved_prompt_path}, output_path={resolved_output_path}, "
            f"prompts={len(prompts)}, prompt_range=[{prompt_start}, {prompt_end})",
            flush=True,
        )
        self.run_entity_extraction_prompts(
            prompts,
            checkpoint_path=resolved_output_path,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
            label=label,
            result_index_offset=prompt_start,
            result_meta={
                "dataset_name": dataset_name,
                "prompt_path": str(resolved_prompt_path),
                "total_prompts": len(prompt_payload["prompts"]),
                "slice_index": slice_index,
                "num_slices": num_slices,
                "slice_start": prompt_start,
                "slice_end": prompt_end,
            },
        )
        return resolved_output_path

    @staticmethod
    def _load_entity_extraction_results_payload(results_path: str | Path) -> tuple[dict[str, object], list[EntityExtractionResult]]:
        with Path(results_path).open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            return payload.get("meta", {}), payload.get("answers", [])
        return {}, payload

    def build_from_entity_extraction_results_file(
        self,
        dataset_or_name: DatasetLike,
        *,
        prompt_path: str | Path | None = None,
        results_path: str | Path | None = None,
    ) -> None:
        """Build this graph index from saved prompts plus saved entity-extraction responses."""
        dataset_name = dataset_or_name if isinstance(dataset_or_name, str) else dataset_or_name.dataset_name
        resolved_prompt_path = (
            Path(prompt_path)
            if prompt_path is not None
            else _default_graph_entity_extraction_prompts_path(dataset_name)
        )
        resolved_results_path = (
            Path(results_path)
            if results_path is not None
            else _default_graph_entity_extraction_results_path(dataset_name)
        )

        prompt_payload = self.load_entity_extraction_prompt_payload(resolved_prompt_path)
        _results_meta, results = self._load_entity_extraction_results_payload(resolved_results_path)
        dataset, chunk_lookup, all_chunk_keys = self._load_chunk_lookup_for_prompt_payload(
            dataset_or_name,
            prompt_payload,
        )
        print(
            f"[graph index] building graph index from saved entity results for dataset={dataset.dataset_name}: "
            f"prompt_path={resolved_prompt_path}, results_path={resolved_results_path}",
            flush=True,
        )
        self._populate_from_entity_extraction_results(chunk_lookup, all_chunk_keys, results)
        self.dataset_name = dataset.dataset_name
        self.attach_chunk_embeddings_from_vector_dbs(dataset.dataset_name)
        print(
            f"[graph index] completed build from saved entity results for dataset={dataset.dataset_name}: "
            f"chunks={self.num_chunks}, unique_entities={len(self.entity_to_chunk_keys)}, "
            f"embedding_tables={list(self.chunk_embeddings.keys())}",
            flush=True,
        )

    def run_entity_extraction_prompts(
        self,
        prompts: list[EntityExtractionPrompt],
        *,
        checkpoint_path: str | Path | None = None,
        max_workers: int = 16,
        checkpoint_every: int = 500,
        label: str = "Graph entity extraction",
        result_index_offset: int = 0,
        result_meta: dict[str, object] | None = None,
    ) -> list[EntityExtractionResult]:
        """Execute entity-extraction prompts with threaded LLM calls and checkpointing."""
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")

        if checkpoint_every < 1:
            raise ValueError(f"checkpoint_every must be >= 1, got {checkpoint_every}")

        if not prompts:
            print(f"[{label}] No prompts to execute.", flush=True)
            return []

        checkpoint_payload: list[EntityExtractionResult | None] = [None] * len(prompts)
        checkpoint_path_obj = Path(checkpoint_path) if checkpoint_path is not None else None
        prompt_kind_counts = _count_prompt_kinds(prompts)
        meta_out = {
            "label": label,
            "num_prompts": len(prompts),
            "result_index_offset": result_index_offset,
        }
        if result_meta:
            meta_out.update(result_meta)
        print(
            f"[{label}] Prompt mix: "
            f"title={prompt_kind_counts.get('title', 0)}, "
            f"chunk={prompt_kind_counts.get('chunk', 0)}, "
            f"total={len(prompts)}",
            flush=True,
        )
        if checkpoint_path_obj is not None:
            print(f"[{label}] Checkpoint path: {checkpoint_path_obj}", flush=True)

        if checkpoint_path_obj is not None and checkpoint_path_obj.exists():
            try:
                with checkpoint_path_obj.open("rb") as f:
                    existing_payload = pickle.load(f)
                loaded_answers = existing_payload.get("answers", checkpoint_payload)
                if len(loaded_answers) != len(prompts):
                    raise ValueError(
                        f"checkpoint answer count mismatch: expected {len(prompts)}, "
                        f"got {len(loaded_answers)}"
                    )
                checkpoint_payload = loaded_answers
                print(
                    f"[{label}] Loaded existing results from {checkpoint_path_obj}.",
                    flush=True,
                )
            except Exception as exc:
                print(f"[{label}] Failed to load checkpoint from {checkpoint_path_obj}: {exc}", flush=True)

        remaining_count = 0
        done_titles = 0
        done_chunks = 0
        for idx, entry in enumerate(checkpoint_payload):
            if entry is None or entry.get("response") is None:
                remaining_count += 1
            elif prompts[idx]["kind"] == "title":
                done_titles += 1
            else:
                done_chunks += 1

        if done_titles or done_chunks:
            print(
                f"[{label}] Resume state: completed title prompts={done_titles}, "
                f"completed chunk prompts={done_chunks}, remaining={remaining_count}",
                flush=True,
            )
        if remaining_count == 0:
            print(f"[{label}] All prompts already answered ({len(prompts)} total).", flush=True)
            return [entry for entry in checkpoint_payload if entry is not None]

        def _save_checkpoint() -> None:
            if checkpoint_path_obj is None:
                return
            checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)
            with checkpoint_path_obj.open("wb") as f:
                pickle.dump({"meta": meta_out, "answers": checkpoint_payload}, f)

        def _call(idx: int) -> str:
            prompt = prompts[idx]
            return self.entity_extractor.answer_prompt(
                system_prompt=prompt["system_prompt"],
                user_prompt=prompt["user_prompt"],
            )

        total_done_before = len(checkpoint_payload) - remaining_count
        print_every = max(25, min(250, checkpoint_every // 5 if checkpoint_every > 5 else checkpoint_every))
        print(
            f"[{label}] Starting/resuming: {len(prompts)} prompts "
            f"({total_done_before} done, {remaining_count} remaining, max_workers={max_workers}, "
            f"progress_print_every={print_every}).",
            flush=True,
        )

        progress = (
            tqdm(total=remaining_count, desc=label, unit="prompt", dynamic_ncols=True)
            if tqdm is not None
            else None
        )
        remaining_iter = (
            idx
            for idx, entry in enumerate(checkpoint_payload)
            if entry is None or entry.get("response") is None
        )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {}
                since_last_checkpoint = 0
                since_last_print = 0
                completed_titles = 0
                completed_chunks = 0
                failed_prompts = 0
                successful_done = total_done_before

                def _submit_next() -> bool:
                    try:
                        idx = next(remaining_iter)
                    except StopIteration:
                        return False
                    future_to_idx[executor.submit(_call, idx)] = idx
                    return True

                for _ in range(min(max_workers, remaining_count)):
                    _submit_next()

                while future_to_idx:
                    done_futures, _pending = wait(
                        future_to_idx,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done_futures:
                        idx = future_to_idx.pop(future)
                        prompt = prompts[idx]
                        try:
                            response = future.result()
                            error = None
                        except Exception as exc:
                            response = None
                            error = str(exc)
                        entities = self.entity_extractor.parse_response(response) if response is not None else []

                        checkpoint_payload[idx] = {
                            "kind": prompt["kind"],
                            "doc_id": prompt.get("doc_id"),
                            "chunk_id": prompt.get("chunk_id"),
                            "question_id": prompt.get("question_id"),
                            "split": prompt.get("split"),
                            "question": prompt.get("question"),
                            "prompt_index": result_index_offset + idx,
                            "response": response,
                            "entities": entities,
                            "error": error,
                        }
                        if response is not None:
                            successful_done += 1
                        if prompt["kind"] == "title":
                            completed_titles += 1
                        else:
                            completed_chunks += 1
                        if error is not None:
                            failed_prompts += 1
                            print(
                                f"[{label}] Prompt failed: kind={prompt.get('kind')!r} "
                                f"doc_id={prompt.get('doc_id')!r} chunk_id={prompt.get('chunk_id')!r} "
                                f"question_id={prompt.get('question_id')!r} error={error}",
                                flush=True,
                            )
                        if progress is not None:
                            progress.update(1)
                            progress.set_postfix(
                                titles=completed_titles,
                                chunks=completed_chunks,
                                failed=failed_prompts,
                            )

                        since_last_checkpoint += 1
                        since_last_print += 1
                        if since_last_print >= print_every:
                            total_done = completed_titles + completed_chunks + total_done_before
                            print(
                                f"[{label}] Progress: answered={total_done}/{len(prompts)} "
                                f"(titles_this_run={completed_titles}, chunks_this_run={completed_chunks}, "
                                f"failed_this_run={failed_prompts})",
                                flush=True,
                            )
                            since_last_print = 0
                        if since_last_checkpoint >= checkpoint_every:
                            _save_checkpoint()
                            print(
                                f"[{label}] Checkpoint saved to {checkpoint_path_obj} "
                                f"({successful_done}/{len(prompts)} answered).",
                                flush=True,
                            )
                            since_last_checkpoint = 0

                    while len(future_to_idx) < max_workers and _submit_next():
                        pass
        finally:
            if progress is not None:
                progress.close()

        _save_checkpoint()
        print(
            f"[{label}] Completed prompt execution "
            f"({successful_done}/{len(prompts)} answered).",
            flush=True,
        )
        return [entry for entry in checkpoint_payload if entry is not None]

    def _populate_from_entity_extraction_results(
        self,
        chunk_lookup: dict[ChunkKey, TextUnit],
        all_chunk_keys: list[ChunkKey],
        results: list[EntityExtractionResult],
    ) -> None:
        """Build graph-index mappings from title/chunk extraction responses."""
        title_entities_by_doc_id: dict[str | int, list[str]] = {}
        chunk_entities_by_key: dict[ChunkKey, list[str]] = {}
        title_results = 0
        chunk_results = 0
        skipped_results = 0

        print(
            f"[graph index] parsing {len(results)} entity-extraction responses into index structures",
            flush=True,
        )
        result_iterator = results
        if tqdm is not None:
            result_iterator = tqdm(
                results,
                total=len(results),
                desc=f"Parsing entity responses [{self.dataset_name or 'dataset'}]",
                unit="response",
                dynamic_ncols=True,
            )

        for result_idx, result in enumerate(result_iterator, start=1):
            response = result.get("response")
            if response is None:
                skipped_results += 1
                _print_progress(
                    f"[graph index] parse entity responses [{self.dataset_name or 'dataset'}]",
                    result_idx,
                    len(results),
                )
                continue

            entities = self.entity_extractor.parse_response(response)
            kind = result.get("kind")
            doc_id = result.get("doc_id")
            chunk_id = result.get("chunk_id")

            if kind == "title":
                title_entities_by_doc_id[doc_id] = entities
                title_results += 1
                _print_progress(
                    f"[graph index] parse entity responses [{self.dataset_name or 'dataset'}]",
                    result_idx,
                    len(results),
                )
                continue

            chunk_entities_by_key[(doc_id, chunk_id)] = entities
            chunk_results += 1
            _print_progress(
                f"[graph index] parse entity responses [{self.dataset_name or 'dataset'}]",
                result_idx,
                len(results),
            )

        print(
            f"[graph index] parsed responses: title_results={title_results}, "
            f"chunk_results={chunk_results}, skipped={skipped_results}",
            flush=True,
        )

        entity_to_chunk_keys: defaultdict[str, list[ChunkKey]] = defaultdict(list)
        chunk_to_entities: dict[ChunkKey, list[str]] = {}
        print(
            f"[graph index] assembling final chunk/entity mappings for {len(all_chunk_keys)} chunks",
            flush=True,
        )
        chunk_iterator = all_chunk_keys
        if tqdm is not None:
            chunk_iterator = tqdm(
                all_chunk_keys,
                total=len(all_chunk_keys),
                desc=f"Assembling graph index [{self.dataset_name or 'dataset'}]",
                unit="chunk",
                dynamic_ncols=True,
            )

        for chunk_idx, chunk_key in enumerate(chunk_iterator, start=1):
            chunk_entities = chunk_entities_by_key.get(chunk_key, [])
            title_entities = title_entities_by_doc_id.get(chunk_key[0], [])
            entities = _merge_entities(chunk_entities, title_entities)
            chunk_to_entities[chunk_key] = entities
            for entity in entities:
                entity_to_chunk_keys[entity].append(chunk_key)
            _print_progress(
                f"[graph index] assemble graph index [{self.dataset_name or 'dataset'}]",
                chunk_idx,
                len(all_chunk_keys),
            )

        self.entity_to_chunk_keys = dict(entity_to_chunk_keys)
        self.chunk_to_entities = chunk_to_entities
        self.chunk_lookup = chunk_lookup
        self.all_chunk_keys = all_chunk_keys
        self.chunk_key_to_embedding_row = {
            chunk_key: row_idx for row_idx, chunk_key in enumerate(all_chunk_keys)
        }
        self.num_chunks = len(all_chunk_keys)
        print(
            f"[graph index] assembly complete: chunks={self.num_chunks}, "
            f"unique_entities={len(self.entity_to_chunk_keys)}",
            flush=True,
        )

    @staticmethod
    def _load_vector_db_text_units_by_chunk_key(
        dataset_name: str,
        embedder: str,
    ) -> tuple[dict[ChunkKey, TextUnit], Path]:
        """Load FAISS sidecar TextUnits keyed by stable chunk key without loading the FAISS index."""
        embedder_key = C.normalize_embedder_key(embedder)
        vector_db_dir = Path(C.get_vector_db_dir(dataset_name, embedder=embedder_key))
        text_units_path = vector_db_dir / "text_units.pkl"
        if not text_units_path.exists():
            raise FileNotFoundError(
                f"Missing vector DB text unit sidecar for dataset={dataset_name} "
                f"embedder={embedder_key}: {text_units_path}"
            )

        print(
            f"[graph index] loading chunk embeddings from vector DB sidecar: "
            f"dataset={dataset_name}, embedder={embedder_key}, path={text_units_path}",
            flush=True,
        )
        with text_units_path.open("rb") as f:
            payload = pickle.load(f)

        if isinstance(payload, dict):
            text_units_iter = [text_unit for _idx, text_unit in sorted(payload.items(), key=lambda item: item[0])]
        else:
            text_units_iter = list(payload)

        text_units_by_key: dict[ChunkKey, TextUnit] = {}
        for text_unit in text_units_iter:
            chunk_key = (text_unit.doc_id, text_unit.chunk_id)
            text_units_by_key[chunk_key] = text_unit
        return text_units_by_key, text_units_path

    def attach_chunk_embeddings_from_vector_dbs(
        self,
        dataset_name: str | None = None,
        *,
        embedders: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Attach dense chunk embedding matrices aligned to self.all_chunk_keys."""
        resolved_dataset_name = dataset_name or self.dataset_name
        if not resolved_dataset_name:
            raise ValueError("dataset_name is required to attach chunk embeddings")
        if not self.all_chunk_keys:
            raise ValueError("Graph index has no chunk keys; build entity mappings before attaching embeddings")

        embedder_keys = [
            C.normalize_embedder_key(embedder)
            for embedder in (embedders or C.SUPPORTED_DENSE_EMBEDDER_KEYS)
        ]
        self.chunk_key_to_embedding_row = {
            chunk_key: row_idx for row_idx, chunk_key in enumerate(self.all_chunk_keys)
        }

        for embedder_key in embedder_keys:
            text_units_by_key, source_path = self._load_vector_db_text_units_by_chunk_key(
                resolved_dataset_name,
                embedder_key,
            )
            first_vector = None
            for chunk_key in self.all_chunk_keys:
                text_unit = text_units_by_key.get(chunk_key)
                if text_unit is not None and text_unit.embedding is not None:
                    first_vector = np.asarray(text_unit.embedding, dtype=np.float32)
                    break
            if first_vector is None:
                raise ValueError(
                    f"No embeddings found in vector DB text_units for dataset={resolved_dataset_name} "
                    f"embedder={embedder_key}"
                )

            embedding_dim = int(first_vector.shape[-1])
            embeddings = np.empty((len(self.all_chunk_keys), embedding_dim), dtype=np.float32)
            missing_keys: list[ChunkKey] = []
            for row_idx, chunk_key in enumerate(self.all_chunk_keys):
                text_unit = text_units_by_key.get(chunk_key)
                if text_unit is None or text_unit.embedding is None:
                    missing_keys.append(chunk_key)
                    continue
                vector = np.asarray(text_unit.embedding, dtype=np.float32)
                if vector.shape[-1] != embedding_dim:
                    raise ValueError(
                        f"Embedding dim mismatch for dataset={resolved_dataset_name} embedder={embedder_key} "
                        f"chunk_key={chunk_key}: expected {embedding_dim}, got {vector.shape[-1]}"
                    )
                embeddings[row_idx] = vector
                _print_progress(
                    f"[graph index] attach embeddings [{resolved_dataset_name} {embedder_key}]",
                    row_idx + 1,
                    len(self.all_chunk_keys),
                )

            if missing_keys:
                raise ValueError(
                    f"Missing {len(missing_keys)} embeddings for dataset={resolved_dataset_name} "
                    f"embedder={embedder_key}; first missing keys: {missing_keys[:5]}"
                )

            self.chunk_embeddings[embedder_key] = embeddings
            self.chunk_embedding_meta[embedder_key] = {
                "dataset_name": resolved_dataset_name,
                "embedder": embedder_key,
                "source_text_units_path": str(source_path),
                "num_chunks": len(self.all_chunk_keys),
                "dim": embedding_dim,
                "dtype": str(embeddings.dtype),
            }
            print(
                f"[graph index] attached chunk embeddings for dataset={resolved_dataset_name} "
                f"embedder={embedder_key}: shape={embeddings.shape}, dtype={embeddings.dtype}",
                flush=True,
            )
            del text_units_by_key

    def get_chunk_embedding(
        self,
        doc_id: str | int,
        chunk_id: int,
        *,
        embedder: str | None = None,
    ) -> np.ndarray:
        """Return one stored chunk embedding by chunk key."""
        embedder_key = C.normalize_embedder_key(embedder)
        row_idx = self.chunk_key_to_embedding_row[(doc_id, chunk_id)]
        return self.chunk_embeddings[embedder_key][row_idx]

    def get_chunk_embedding_batch(
        self,
        chunk_keys: list[ChunkKey],
        *,
        embedder: str | None = None,
    ) -> np.ndarray:
        """Return stored embeddings for a list of chunk keys, preserving input order."""
        embedder_key = C.normalize_embedder_key(embedder)
        rows = [self.chunk_key_to_embedding_row[chunk_key] for chunk_key in chunk_keys]
        return self.chunk_embeddings[embedder_key][rows]

    def build(
        self,
        dataset_or_name: DatasetLike,
        *,
        max_workers: int = 16,
        checkpoint_every: int = 500,
        checkpoint_path: str | Path | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        """Build the entity/chunk incidence mappings from a dataset or dataset name."""
        dataset_name_hint = dataset_or_name if isinstance(dataset_or_name, str) else dataset_or_name.dataset_name
        resolved_prompt_path = (
            Path(prompt_path)
            if prompt_path is not None
            else _default_graph_entity_extraction_prompts_path(dataset_name_hint)
        )

        if resolved_prompt_path.exists():
            print(
                f"[graph index] loading prepared entity prompts for dataset={dataset_name_hint}: "
                f"{resolved_prompt_path}",
                flush=True,
            )
            prompt_payload = self.load_entity_extraction_prompt_payload(resolved_prompt_path)
            prompts = prompt_payload["prompts"]
            dataset, chunk_lookup, all_chunk_keys = self._load_chunk_lookup_for_prompt_payload(
                dataset_or_name,
                prompt_payload,
            )
        else:
            dataset, chunk_lookup, all_chunk_keys, prompts = self.prepare_entity_extraction_prompts(dataset_or_name)
            prompt_payload = self._build_entity_extraction_prompt_payload(dataset, all_chunk_keys, prompts)
            self.save_entity_extraction_prompt_payload(prompt_payload, resolved_prompt_path)
            print(
                f"[graph index] saved prepared entity prompts for dataset={dataset.dataset_name}: "
                f"{resolved_prompt_path}",
                flush=True,
            )

        dataset_name = getattr(dataset, "dataset_name", None)

        resolved_checkpoint_path = checkpoint_path
        if resolved_checkpoint_path is None and dataset_name:
            resolved_checkpoint_path = _default_graph_index_checkpoint_path(dataset_name)

        label = f"Graph entity extraction [{dataset_name or 'dataset'}]"
        print(
            f"[graph index] prepared {len(prompts)} prompts for dataset={dataset_name} "
            f"({len(all_chunk_keys)} chunk prompts + {len(prompts) - len(all_chunk_keys)} title prompts)",
            flush=True,
        )
        print(
            f"[graph index] starting threaded entity extraction for dataset={dataset_name} "
            f"with max_workers={max_workers}, checkpoint_every={checkpoint_every}",
            flush=True,
        )
        results = self.run_entity_extraction_prompts(
            prompts,
            checkpoint_path=resolved_checkpoint_path,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
            label=label,
        )
        print(
            f"[graph index] entity extraction finished for dataset={dataset_name}; "
            f"reconstructing graph index from {len(results)} responses",
            flush=True,
        )
        self._populate_from_entity_extraction_results(chunk_lookup, all_chunk_keys, results)
        self.dataset_name = dataset_name
        if dataset_name is not None:
            self.attach_chunk_embeddings_from_vector_dbs(dataset_name)
        print(
            f"[graph index] completed build for dataset={dataset_name}: "
            f"chunks={self.num_chunks}, unique_entities={len(self.entity_to_chunk_keys)}, "
            f"embedding_tables={list(self.chunk_embeddings.keys())}",
            flush=True,
        )

    def get_chunk_entities(self, doc_id: str | int, chunk_id: int) -> list[str]:
        """Return entities stored for a given chunk key."""
        return list(self.chunk_to_entities.get((doc_id, chunk_id), []))

    def get_entity_chunks(self, entity: str) -> list[ChunkKey]:
        """Return chunk keys for a normalized entity lookup."""
        normalized = normalize_entity(entity)
        if not normalized:
            return []
        return list(self.entity_to_chunk_keys.get(normalized, []))

    def extract_query_entities(self, query: str) -> list[str]:
        """Extract normalized entities from a query string."""
        return self.entity_extractor.extract_entities(query)

    def save(self, path: str | Path) -> None:
        """Serialize the graph index to disk with pickle."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "GraphIndex":
        """Load a previously serialized graph index from disk."""
        with Path(path).open("rb") as f:
            payload = _GraphIndexUnpickler(f).load()
        if not isinstance(payload, cls):
            raise TypeError(f"Expected a {cls.__name__} at {path}, got {type(payload).__name__}")
        return payload


class _GraphIndexUnpickler(pickle.Unpickler):
    """Unpickle graph indexes saved when graph_index.py was run as __main__."""

    _MAIN_CLASS_ALIASES = {
        "EntityExtractor": EntityExtractor,
        "GraphIndex": GraphIndex,
    }

    def find_class(self, module: str, name: str):
        if module == "__main__" and name in self._MAIN_CLASS_ALIASES:
            return self._MAIN_CLASS_ALIASES[name]
        return super().find_class(module, name)


def _normalize_dataset_names(datasets: list[str] | None = None) -> list[str]:
    if datasets is None:
        return list(C.DATASETS)
    return [dataset.strip() for dataset in datasets if dataset and dataset.strip()]


def prepare_all_entity_extraction_prompt_files(
    datasets: list[str] | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Prepare and save entity-extraction prompt files for all requested datasets."""
    output_paths: dict[str, Path] = {}
    for dataset_name in _normalize_dataset_names(datasets):
        graph_index = GraphIndex(dataset_name=dataset_name)
        output_paths[dataset_name] = graph_index.prepare_and_save_entity_extraction_prompts(
            dataset_name,
            overwrite=overwrite,
        )
    return output_paths


def prepare_all_question_entity_extraction_prompt_files(
    datasets: list[str] | None = None,
    *,
    splits: list[str] | None = None,
    overwrite: bool = False,
) -> dict[tuple[str, str], Path]:
    """Prepare and save question entity-extraction prompt files for requested datasets/splits."""
    output_paths: dict[tuple[str, str], Path] = {}
    normalized_splits = [_normalize_question_split(split) for split in (splits or ["train", "test"])]
    for dataset_name in _normalize_dataset_names(datasets):
        graph_index = GraphIndex(dataset_name=dataset_name)
        for split in normalized_splits:
            output_paths[(dataset_name, split)] = graph_index.prepare_and_save_question_entity_extraction_prompts(
                dataset_name,
                split,
                overwrite=overwrite,
            )
    return output_paths


def answer_saved_question_entity_extraction_prompt_files(
    datasets: list[str] | None = None,
    *,
    splits: list[str] | None = None,
    max_workers: int = 16,
    checkpoint_every: int = 500,
) -> dict[tuple[str, str], Path]:
    """Answer prepared question entity-extraction prompt files for requested datasets/splits."""
    output_paths: dict[tuple[str, str], Path] = {}
    normalized_splits = [_normalize_question_split(split) for split in (splits or ["train", "test"])]
    for dataset_name in _normalize_dataset_names(datasets):
        graph_index = GraphIndex(dataset_name=dataset_name)
        for split in normalized_splits:
            output_paths[(dataset_name, split)] = graph_index.answer_saved_question_entity_extraction_prompts(
                dataset_name,
                split,
                max_workers=max_workers,
                checkpoint_every=checkpoint_every,
            )
    return output_paths


def answer_saved_entity_extraction_prompt_files(
    datasets: list[str] | None = None,
    *,
    max_workers: int = 16,
    checkpoint_every: int = 500,
    slice_index: int | None = None,
    num_slices: int | None = None,
) -> dict[str, Path]:
    """Answer prepared entity-extraction prompt files for all requested datasets."""
    output_paths: dict[str, Path] = {}
    for dataset_name in _normalize_dataset_names(datasets):
        graph_index = GraphIndex(dataset_name=dataset_name)
        output_paths[dataset_name] = graph_index.answer_saved_entity_extraction_prompts(
            dataset_name,
            max_workers=max_workers,
            checkpoint_every=checkpoint_every,
            slice_index=slice_index,
            num_slices=num_slices,
        )
    return output_paths


def merge_entity_extraction_result_slices(
    dataset_name: str,
    *,
    num_slices: int,
    output_path: str | Path | None = None,
    prompt_path: str | Path | None = None,
    allow_incomplete: bool = False,
) -> Path:
    """Merge independently answered entity-extraction result slices into one result file."""
    resolved_prompt_path = (
        Path(prompt_path)
        if prompt_path is not None
        else _default_graph_entity_extraction_prompts_path(dataset_name)
    )
    prompt_payload = GraphIndex.load_entity_extraction_prompt_payload(resolved_prompt_path)
    total_prompts = len(prompt_payload["prompts"])
    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else _default_graph_entity_extraction_results_path(dataset_name)
    )

    print(
        f"[graph index] merging entity result slices for dataset={dataset_name}: "
        f"num_slices={num_slices}, total_prompts={total_prompts}, output_path={resolved_output_path}",
        flush=True,
    )
    merged_answers: list[EntityExtractionResult | None] = []
    total_complete = 0
    total_failed = 0
    for slice_index in range(num_slices):
        start, end = _slice_bounds(total_prompts, slice_index, num_slices)
        slice_path = _default_graph_entity_extraction_results_slice_path(
            dataset_name,
            slice_index,
            num_slices,
        )
        with slice_path.open("rb") as f:
            payload = pickle.load(f)
        answers = payload.get("answers") if isinstance(payload, dict) else payload
        if len(answers) != end - start:
            raise ValueError(
                f"slice {slice_index}/{num_slices} answer count mismatch at {slice_path}: "
                f"expected {end - start}, got {len(answers)}"
            )

        complete = 0
        failed = 0
        for local_idx, entry in enumerate(answers):
            expected_prompt_index = start + local_idx
            if entry is not None and entry.get("prompt_index") is not None:
                if entry.get("prompt_index") != expected_prompt_index:
                    raise ValueError(
                        f"slice {slice_index}/{num_slices} has prompt_index mismatch at local_idx={local_idx}: "
                        f"expected {expected_prompt_index}, got {entry.get('prompt_index')}"
                    )
            if entry is None or entry.get("response") is None:
                if not allow_incomplete:
                    raise ValueError(
                        f"slice {slice_index}/{num_slices} is incomplete at prompt_index={expected_prompt_index}"
                    )
                failed += 1
            else:
                complete += 1
                if entry.get("error") is not None:
                    failed += 1

        merged_answers.extend(answers)
        total_complete += complete
        total_failed += failed
        print(
            f"[graph index] merged slice {slice_index}/{num_slices}: "
            f"path={slice_path}, range=[{start}, {end}), complete={complete}, failed_or_missing={failed}",
            flush=True,
        )

    if len(merged_answers) != total_prompts:
        raise ValueError(f"merged answer count mismatch: expected {total_prompts}, got {len(merged_answers)}")

    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_output_path.open("wb") as f:
        pickle.dump(
            {
                "meta": {
                    "label": f"Graph entity extraction [{dataset_name}]",
                    "dataset_name": dataset_name,
                    "num_prompts": total_prompts,
                    "num_slices": num_slices,
                    "complete_answers": total_complete,
                    "failed_or_missing_answers": total_failed,
                    "merged_from_slices": True,
                },
                "answers": merged_answers,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(
        f"[graph index] saved merged entity results for dataset={dataset_name}: "
        f"path={resolved_output_path}, complete={total_complete}/{total_prompts}, "
        f"failed_or_missing={total_failed}",
        flush=True,
    )
    return resolved_output_path


def merge_entity_extraction_result_slice_files(
    datasets: list[str] | None = None,
    *,
    num_slices: int,
    allow_incomplete: bool = False,
) -> dict[str, Path]:
    """Merge saved entity-extraction result slices for all requested datasets."""
    output_paths: dict[str, Path] = {}
    for dataset_name in _normalize_dataset_names(datasets):
        output_paths[dataset_name] = merge_entity_extraction_result_slices(
            dataset_name,
            num_slices=num_slices,
            allow_incomplete=allow_incomplete,
        )
    return output_paths


def build_graph_indexes_from_entity_extraction_result_files(
    datasets: list[str] | None = None,
) -> dict[str, Path]:
    """Build and save graph indexes from prepared prompts plus saved extraction results."""
    output_paths: dict[str, Path] = {}
    for dataset_name in _normalize_dataset_names(datasets):
        graph_index = GraphIndex(dataset_name=dataset_name)
        graph_index.build_from_entity_extraction_results_file(dataset_name)
        output_path = Path(C.get_graph_index_path(dataset_name))
        graph_index.save(output_path)
        output_paths[dataset_name] = output_path
        print(
            f"[graph index] saved graph index for dataset={dataset_name}: "
            f"path={output_path}, chunks={graph_index.num_chunks}, entities={len(graph_index.entity_to_chunk_keys)}",
            flush=True,
        )
    return output_paths


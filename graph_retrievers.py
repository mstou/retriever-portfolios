from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

import constants as C
from data_classes import TextUnit
from graph_index import ChunkKey, GraphIndex, normalize_entity
from text_processing import Embedder


class GraphDenseRetriever:
    """
    Graph candidate generation over entity/chunk edges with dense reranking.

    Hops alternate entities and chunks. Odd hops collect chunk candidates, so an
    even max_hops value is allowed but stops on an entity frontier without
    adding candidates from that final frontier.
    """

    def __init__(
        self,
        graph_index: GraphIndex,
        embedder: Embedder | dict[str, Embedder],
        max_hops: int = 1,
        max_entity_df: int = 500,
        max_candidates: int = 1000,
        fallback_retriever=None,
        device: str = "cpu",
    ):
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        if max_entity_df < 1:
            raise ValueError(f"max_entity_df must be >= 1, got {max_entity_df}")
        if max_candidates < 1:
            raise ValueError(f"max_candidates must be >= 1, got {max_candidates}")

        self.graph_index = graph_index
        self.embedder = embedder
        self.embedder_key = C.normalize_embedder_key(getattr(embedder, "embedder_key", None))
        self.max_hops = max_hops
        self.max_entity_df = max_entity_df
        self.max_candidates = max_candidates
        self.fallback_retriever = fallback_retriever
        self.device = device

    def query(
        self,
        text: str,
        num_results: int = 5,
        candidates=None,
        q_vec=None,
    ) -> list[TextUnit]:
        del candidates

        query_entities = self.extract_query_entities(text)
        if not query_entities:
            return self._fallback_query(text, num_results, q_vec=q_vec)

        candidate_chunk_keys = self._expand_candidate_chunk_keys(query_entities)
        if not candidate_chunk_keys:
            return self._fallback_query(text, num_results, q_vec=q_vec)

        return self._rerank_candidate_chunk_keys(
            text,
            candidate_chunk_keys,
            num_results=num_results,
            q_vec=q_vec,
        )

    def extract_query_entities(self, text: str) -> list[str]:
        entities = self.graph_index.extract_query_entities(text)
        return self._normalize_entities(entities)

    def count_graph_candidates(self, text: str) -> int:
        query_entities = self.extract_query_entities(text)
        return len(self._expand_candidate_chunk_keys(query_entities))

    def _fallback_query(self, text: str, num_results: int, q_vec=None) -> list[TextUnit]:
        if self.fallback_retriever is None:
            return []

        try:
            return self.fallback_retriever.query(text, num_results=num_results, q_vec=q_vec)
        except TypeError:
            return self.fallback_retriever.query(text, num_results=num_results)

    def _normalize_entities(self, entities: Iterable[str]) -> list[str]:
        normalized_entities: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            normalized = normalize_entity(entity)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_entities.append(normalized)
        return normalized_entities

    def _entity_is_expandable(self, entity: str) -> bool:
        chunk_frequency = len(self.graph_index.entity_to_chunk_keys.get(entity, []))
        return 0 < chunk_frequency <= self.max_entity_df

    def _expand_candidate_chunk_keys(self, query_entities: list[str]) -> list[ChunkKey]:
        current_entities = self._normalize_entities(query_entities)
        current_chunks: list[ChunkKey] = []
        visited_entities: set[str] = set()
        visited_chunks: set[ChunkKey] = set()
        candidate_chunk_keys: list[ChunkKey] = []

        for hop in range(1, self.max_hops + 1):
            if hop % 2 == 1:
                next_chunks: list[ChunkKey] = []
                for entity in current_entities:
                    if entity in visited_entities:
                        continue
                    visited_entities.add(entity)
                    if not self._entity_is_expandable(entity):
                        continue

                    for chunk_key in self.graph_index.get_entity_chunks(entity):
                        if chunk_key in visited_chunks:
                            continue
                        visited_chunks.add(chunk_key)
                        next_chunks.append(chunk_key)
                        if len(candidate_chunk_keys) < self.max_candidates:
                            candidate_chunk_keys.append(chunk_key)
                        if len(candidate_chunk_keys) >= self.max_candidates:
                            break
                    if len(candidate_chunk_keys) >= self.max_candidates:
                        break

                current_chunks = next_chunks
                current_entities = []
                if len(candidate_chunk_keys) >= self.max_candidates:
                    break
            else:
                next_entities: list[str] = []
                next_entity_set: set[str] = set()
                for chunk_key in current_chunks:
                    for entity in self.graph_index.chunk_to_entities.get(chunk_key, []):
                        normalized = normalize_entity(entity)
                        if (
                            not normalized
                            or normalized in visited_entities
                            or normalized in next_entity_set
                            or not self._entity_is_expandable(normalized)
                        ):
                            continue
                        next_entity_set.add(normalized)
                        next_entities.append(normalized)

                current_entities = next_entities
                current_chunks = []
                if not current_entities:
                    break

        return candidate_chunk_keys

    def _rerank_candidate_chunk_keys(
        self,
        text: str,
        candidate_chunk_keys: list[ChunkKey],
        *,
        num_results: int,
        q_vec=None,
    ) -> list[TextUnit]:
        q_vec_np = self._query_vector(text, q_vec)
        candidate_embeddings = self._candidate_embeddings(candidate_chunk_keys)
        if candidate_embeddings.shape[1] != q_vec_np.shape[0]:
            raise ValueError(
                "GraphDenseRetriever query and chunk embedding dimensions do not match: "
                f"query_dim={q_vec_np.shape[0]}, chunk_dim={candidate_embeddings.shape[1]}. "
                f"Ensure q_vec and graph chunk embeddings both use embedder={self.embedder_key}."
            )
        scores = candidate_embeddings @ q_vec_np
        ranked_indices = np.argsort(-scores, kind="stable")[:num_results]
        return [
            self.graph_index.chunk_lookup[candidate_chunk_keys[int(idx)]]
            for idx in ranked_indices
        ]

    def _query_vector(self, text: str, q_vec=None) -> np.ndarray:
        if q_vec is None:
            q_vec = self.embedder.embed(text, role="query")
        return self._as_float32_vector(q_vec, label="query embedding")

    def _candidate_embeddings(self, candidate_chunk_keys: list[ChunkKey]) -> np.ndarray:
        if (
            hasattr(self.graph_index, "chunk_embeddings")
            and self.embedder_key in self.graph_index.chunk_embeddings
        ):
            try:
                return np.asarray(
                    self.graph_index.get_chunk_embedding_batch(
                        candidate_chunk_keys,
                        embedder=self.embedder_key,
                    ),
                    dtype=np.float32,
                )
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    "GraphDenseRetriever could not load chunk embeddings from the "
                    f"GraphIndex for embedder={self.embedder_key}. Ensure the graph "
                    "index was built with matching chunk keys and dense artifacts."
                ) from exc

        embeddings: list[np.ndarray] = []
        missing_keys: list[ChunkKey] = []
        for chunk_key in candidate_chunk_keys:
            text_unit = self.graph_index.chunk_lookup.get(chunk_key)
            if text_unit is None or text_unit.embedding is None:
                missing_keys.append(chunk_key)
                continue
            embeddings.append(self._as_float32_vector(text_unit.embedding, label="chunk embedding"))

        if missing_keys:
            raise ValueError(
                "GraphDenseRetriever needs chunk embeddings from the selected dense "
                f"backbone ({self.embedder_key}) to rerank graph candidates. Missing "
                f"embeddings for {len(missing_keys)} candidate chunks; first missing "
                f"keys: {missing_keys[:5]}. Load a GraphIndex with attached chunk "
                "embeddings or populate TextUnit.embedding from the matching vector DB."
            )

        if not embeddings:
            raise ValueError(
                "GraphDenseRetriever found graph candidates but no chunk embeddings "
                f"for dense reranking with embedder={self.embedder_key}."
            )

        return np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _as_float32_vector(vector, *, label: str) -> np.ndarray:
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        vector_np = np.asarray(vector, dtype=np.float32)
        if vector_np.ndim == 2 and vector_np.shape[0] == 1:
            vector_np = vector_np[0]
        if vector_np.ndim != 1:
            raise ValueError(f"Expected {label} to be a 1D vector, got shape {vector_np.shape}")
        return vector_np


class BatchGraphDenseRetriever:
    """Batch graph-candidate retriever pool with shared per-query expansion."""

    def __init__(
        self,
        graph_index: GraphIndex,
        embedder: Embedder,
        retriever_params: list[dict],
        query_entities: list[list[str]] | dict[str, list[str]],
        device: str = "cpu",
        fallback_batch_retriever=None,
    ):
        if not retriever_params:
            raise ValueError("retriever_params must contain at least one graph-dense variant")

        self.graph_index = graph_index
        self.embedders = self._normalize_embedders(embedder)
        self.retriever_params = [self._normalize_param(param) for param in retriever_params]
        self.embedder_keys = list(dict.fromkeys(param["embedder"] for param in self.retriever_params))
        missing_embedders = [key for key in self.embedder_keys if key not in self.embedders]
        if missing_embedders:
            raise ValueError(
                "BatchGraphDenseRetriever is missing Embedder objects for: "
                f"{missing_embedders}. Available: {sorted(self.embedders)}"
            )
        self.embedder = self.embedders[self.embedder_keys[0]]
        self.embedder_key = self.embedder_keys[0]
        self.device = device
        self.fallback_batch_retriever = fallback_batch_retriever

        self.max_pool_hops = max(param["max_hops"] for param in self.retriever_params)
        self.max_pool_entity_df = max(param["max_entity_df"] for param in self.retriever_params)
        self.max_pool_candidates = max(param["max_candidates"] for param in self.retriever_params)

        self.query_entities_by_index: list[list[str]] | None = None
        self.query_entities_by_text: dict[str, list[str]] = {}
        if isinstance(query_entities, dict):
            self.query_entities_by_text = {
                query: self._normalize_entities(entities)
                for query, entities in query_entities.items()
            }
        else:
            self.query_entities_by_index = [
                self._normalize_entities(entities)
                for entities in query_entities
            ]

    def num_retrievers(self) -> int:
        return len(self.retriever_params)

    def query(
        self,
        text: str,
        num_results: int = 5,
        candidates=None,
        q_vec=None,
        query_idx: int | None = None,
    ) -> list[list[TextUnit]]:
        del candidates

        query_entities = self.get_query_entities(text, query_idx=query_idx)
        if not query_entities:
            return self._fallback_all(text, num_results, q_vec=q_vec, query_idx=query_idx)

        context = self.expand_query_context(query_entities)
        if not context["candidate_order"]:
            return self._fallback_all(text, num_results, q_vec=q_vec, query_idx=query_idx)

        fallback_results = None
        q_vecs_by_embedder: dict[str, np.ndarray] = {}
        results: list[list[TextUnit]] = []
        for param_idx, param in enumerate(self.retriever_params):
            candidate_chunk_keys = self.candidate_keys_for_param(context, param)
            if not candidate_chunk_keys:
                if fallback_results is None:
                    fallback_results = self._fallback_all(
                        text,
                        num_results,
                        q_vec=q_vec,
                        query_idx=query_idx,
                    )
                results.append(fallback_results[param_idx])
                continue
            embedder_key = param["embedder"]
            if embedder_key not in q_vecs_by_embedder:
                q_vecs_by_embedder[embedder_key] = self._query_vector(text, q_vec, embedder_key)
            results.append(
                self._rerank_candidate_chunk_keys(
                    candidate_chunk_keys,
                    q_vecs_by_embedder[embedder_key],
                    embedder_key=embedder_key,
                    num_results=num_results,
                )
            )
        return results

    def get_query_entities(self, text: str, query_idx: int | None = None) -> list[str]:
        if query_idx is not None:
            if self.query_entities_by_index is None:
                raise ValueError(
                    "BatchGraphDenseRetriever received query_idx but query_entities "
                    "were keyed by query text, not index"
                )
            if query_idx < 0 or query_idx >= len(self.query_entities_by_index):
                raise IndexError(
                    f"query_idx {query_idx} is out of range for "
                    f"{len(self.query_entities_by_index)} cached query-entity rows"
                )
            return list(self.query_entities_by_index[query_idx])

        if self.query_entities_by_index is not None:
            raise ValueError(
                "BatchGraphDenseRetriever requires query_idx when query_entities "
                "are supplied as a split-aligned list"
            )
        return list(self.query_entities_by_text.get(text, []))

    def expand_query_context(self, query_entities: list[str]) -> dict[str, object]:
        current_entities = self._normalize_entities(query_entities)
        current_chunks: list[ChunkKey] = []
        visited_entities: set[str] = set()
        visited_chunks: set[ChunkKey] = set()
        candidate_order: list[ChunkKey] = []
        chunk_records: dict[ChunkKey, dict[str, object]] = {}
        entity_records: dict[str, dict[str, object]] = {}

        for entity in current_entities:
            self._record_entity(entity_records, entity, first_seen_hop=0)

        for hop in range(1, self.max_pool_hops + 1):
            if hop % 2 == 1:
                next_chunks: list[ChunkKey] = []
                for entity in current_entities:
                    if entity in visited_entities:
                        continue
                    visited_entities.add(entity)
                    entity_record = self._record_entity(
                        entity_records,
                        entity,
                        first_seen_hop=max(0, hop - 1),
                    )
                    if entity_record["df"] > self.max_pool_entity_df:
                        continue

                    for chunk_key in self.graph_index.get_entity_chunks(entity):
                        if chunk_key in chunk_records:
                            chunk_records[chunk_key]["via_entities"].add(entity)
                            continue
                        if len(candidate_order) >= self.max_pool_candidates:
                            break
                        chunk_records[chunk_key] = {
                            "chunk_key": chunk_key,
                            "first_seen_hop": hop,
                            "via_entities": {entity},
                            "order": len(candidate_order),
                        }
                        candidate_order.append(chunk_key)
                        if chunk_key not in visited_chunks:
                            visited_chunks.add(chunk_key)
                            next_chunks.append(chunk_key)
                    if len(candidate_order) >= self.max_pool_candidates:
                        break

                current_chunks = next_chunks
                current_entities = []
                if len(candidate_order) >= self.max_pool_candidates:
                    break
            else:
                next_entities: list[str] = []
                next_entity_set: set[str] = set()
                for chunk_key in current_chunks:
                    for entity in self.graph_index.chunk_to_entities.get(chunk_key, []):
                        normalized = normalize_entity(entity)
                        if (
                            not normalized
                            or normalized in visited_entities
                            or normalized in next_entity_set
                        ):
                            continue
                        entity_record = self._record_entity(
                            entity_records,
                            normalized,
                            first_seen_hop=hop,
                        )
                        if entity_record["df"] > self.max_pool_entity_df:
                            continue
                        next_entity_set.add(normalized)
                        next_entities.append(normalized)

                current_entities = next_entities
                current_chunks = []
                if not current_entities:
                    break

        return {
            "candidate_order": candidate_order,
            "chunk_records": chunk_records,
            "entity_records": entity_records,
        }

    def candidate_keys_for_param(self, context: dict[str, object], param: dict) -> list[ChunkKey]:
        normalized_param = self._normalize_param(param)
        chunk_records = context["chunk_records"]
        entity_records = context["entity_records"]
        selected_chunk_keys: list[ChunkKey] = []

        for chunk_key in context["candidate_order"]:
            chunk_record = chunk_records[chunk_key]
            if chunk_record["first_seen_hop"] > normalized_param["max_hops"]:
                continue
            via_entities = chunk_record["via_entities"]
            if not any(
                entity_records[entity]["df"] <= normalized_param["max_entity_df"]
                for entity in via_entities
                if entity in entity_records
            ):
                continue
            selected_chunk_keys.append(chunk_key)
            if len(selected_chunk_keys) >= normalized_param["max_candidates"]:
                break

        return selected_chunk_keys

    @staticmethod
    def _normalize_param(param: dict) -> dict:
        normalized = dict(param)
        normalized["embedder"] = C.normalize_embedder_key(normalized.get("embedder"))
        if normalized["embedder"] == C.GRAPH_DENSE_MIXED_EMBEDDER_KEY:
            raise ValueError(f"Graph-dense variant cannot use mixed as its embedder: {param}")
        for key in ("max_hops", "max_entity_df", "max_candidates"):
            if key not in normalized:
                raise ValueError(f"Graph-dense parameter is missing {key!r}: {param}")
            normalized[key] = int(normalized[key])
            if normalized[key] < 1:
                raise ValueError(f"{key} must be >= 1 in graph-dense parameter: {param}")
        normalized.setdefault(
            "name",
            f"{normalized['embedder']}_h{normalized['max_hops']}_"
            f"df{normalized['max_entity_df']}_c{normalized['max_candidates']}",
        )
        return normalized

    @staticmethod
    def _normalize_embedders(embedder: Embedder | dict[str, Embedder]) -> dict[str, Embedder]:
        if isinstance(embedder, dict):
            return {
                C.normalize_embedder_key(key): value
                for key, value in embedder.items()
            }
        embedder_key = C.normalize_embedder_key(getattr(embedder, "embedder_key", None))
        return {embedder_key: embedder}

    def _record_entity(
        self,
        entity_records: dict[str, dict[str, object]],
        entity: str,
        *,
        first_seen_hop: int,
    ) -> dict[str, object]:
        if entity not in entity_records:
            entity_records[entity] = {
                "entity": entity,
                "df": self._entity_df(entity),
                "first_seen_hop": first_seen_hop,
            }
        return entity_records[entity]

    def _entity_df(self, entity: str) -> int:
        return len(self.graph_index.entity_to_chunk_keys.get(entity, []))

    def _fallback_all(
        self,
        text: str,
        num_results: int,
        *,
        q_vec=None,
        query_idx: int | None = None,
    ) -> list[list[TextUnit]]:
        if self.fallback_batch_retriever is None:
            return [[] for _ in self.retriever_params]

        kwargs = {
            "num_results": num_results,
            "candidates": None,
            "q_vec": q_vec,
        }
        if query_idx is not None:
            kwargs["query_idx"] = query_idx
        try:
            fallback_results = self.fallback_batch_retriever.query(text, **kwargs)
        except TypeError:
            kwargs.pop("query_idx", None)
            fallback_results = self.fallback_batch_retriever.query(text, **kwargs)

        if len(fallback_results) >= len(self.retriever_params):
            return fallback_results[: len(self.retriever_params)]
        padded = list(fallback_results)
        padded.extend([[] for _ in range(len(self.retriever_params) - len(padded))])
        return padded

    def _rerank_candidate_chunk_keys(
        self,
        candidate_chunk_keys: list[ChunkKey],
        q_vec_np: np.ndarray,
        *,
        embedder_key: str,
        num_results: int,
    ) -> list[TextUnit]:
        candidate_embeddings = self._candidate_embeddings(candidate_chunk_keys, embedder_key)
        if candidate_embeddings.shape[1] != q_vec_np.shape[0]:
            raise ValueError(
                "BatchGraphDenseRetriever query and chunk embedding dimensions do not match: "
                f"query_dim={q_vec_np.shape[0]}, chunk_dim={candidate_embeddings.shape[1]}. "
                f"Ensure q_vec and graph chunk embeddings both use embedder={embedder_key}."
            )
        scores = candidate_embeddings @ q_vec_np
        ranked_indices = np.argsort(-scores, kind="stable")[:num_results]
        return [
            self.graph_index.chunk_lookup[candidate_chunk_keys[int(idx)]]
            for idx in ranked_indices
        ]

    def _query_vector(self, text: str, q_vec=None, embedder_key: str | None = None) -> np.ndarray:
        embedder_key = C.normalize_embedder_key(embedder_key or self.embedder_key)
        if isinstance(q_vec, dict):
            q_vec = q_vec.get(embedder_key)
        if q_vec is None:
            q_vec = self.embedders[embedder_key].embed(text, role="query")
        return GraphDenseRetriever._as_float32_vector(q_vec, label="query embedding")

    def _candidate_embeddings(self, candidate_chunk_keys: list[ChunkKey], embedder_key: str) -> np.ndarray:
        embedder_key = C.normalize_embedder_key(embedder_key)
        if (
            hasattr(self.graph_index, "chunk_embeddings")
            and embedder_key in self.graph_index.chunk_embeddings
        ):
            try:
                return np.asarray(
                    self.graph_index.get_chunk_embedding_batch(
                        candidate_chunk_keys,
                        embedder=embedder_key,
                    ),
                    dtype=np.float32,
                )
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    "BatchGraphDenseRetriever could not load chunk embeddings from the "
                    f"GraphIndex for embedder={embedder_key}. Ensure the graph index "
                    "was built with matching chunk keys and dense artifacts."
                ) from exc

        embeddings: list[np.ndarray] = []
        missing_keys: list[ChunkKey] = []
        for chunk_key in candidate_chunk_keys:
            text_unit = self.graph_index.chunk_lookup.get(chunk_key)
            if text_unit is None or text_unit.embedding is None:
                missing_keys.append(chunk_key)
                continue
            embeddings.append(GraphDenseRetriever._as_float32_vector(text_unit.embedding, label="chunk embedding"))

        if missing_keys:
            raise ValueError(
                "BatchGraphDenseRetriever needs chunk embeddings from the selected dense "
                f"backbone ({embedder_key}) to rerank graph candidates. Missing "
                f"embeddings for {len(missing_keys)} candidate chunks; first missing "
                f"keys: {missing_keys[:5]}. Load a GraphIndex with attached chunk "
                "embeddings or populate TextUnit.embedding from the matching vector DB."
            )
        if not embeddings:
            raise ValueError(
                "BatchGraphDenseRetriever found graph candidates but no chunk embeddings "
                f"for dense reranking with embedder={embedder_key}."
            )
        return np.stack(embeddings, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _normalize_entities(entities: Iterable[str]) -> list[str]:
        normalized_entities: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            normalized = normalize_entity(entity)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            normalized_entities.append(normalized)
        return normalized_entities

    @staticmethod
    def _as_float32_vector(vector, *, label: str) -> np.ndarray:
        return GraphDenseRetriever._as_float32_vector(vector, label=label)

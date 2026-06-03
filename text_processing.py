import json
import pickle
from pathlib import Path

import numpy as np
import torch
import constants as C
from data_classes import Dataset, Document, TextUnit
from tqdm import tqdm
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class Embedder:
    """Generates embeddings for text units."""

    def __init__(self, device=None, embedder: str | None = None, model_name: str | None = None):
        """Uses a Hugging Face Sentence Transformer model."""
        self.device = device
        self.embedder_key = C.normalize_embedder_key(embedder or model_name)
        resolved = model_name or C.resolve_embedder_name(embedder)
        self.model_name = resolved
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _prefix_for_role(self, role: str | None):
        if role is None:
            return ""
        if role not in {"query", "passage"}:
            raise ValueError(f"Unsupported embedding role: {role}")
        if self.embedder_key != C.E5_EMBEDDER_KEY:
            return ""
        if role == "query":
            return C.E5_QUERY_PREFIX
        return C.E5_PASSAGE_PREFIX

    def _prepare_texts(self, text, role: str | None = None):
        prefix = self._prefix_for_role(role)
        if not prefix:
            return text
        if isinstance(text, str):
            return prefix + text
        return [prefix + value for value in text]

    def generate_embedding(self, text_unit, role: str = "passage"):
        """Generates embedding for a TextUnit."""
        embedding = self.embed(text_unit.text, role=role)
        text_unit.embedding = embedding  # Store in the TextUnit object
        return embedding
    
    def get_embedding_dim(self):
        return len(self.embed("embedding dimension probe"))
    
    def embed(self, text, leave_on_device=False, role: str | None = None):
        # Disable internal progress bar here to avoid noisy
        # "Batches: 100%" logs when called many times (e.g., VendiRAG).
        prepared_text = self._prepare_texts(text, role=role)
        embedding = self.model.encode(
            prepared_text,
            convert_to_tensor=True,
            batch_size=256,
            normalize_embeddings=True,
            device=self.device,
            show_progress_bar=False,
        )

        if leave_on_device:
            if isinstance(embedding, torch.Tensor) and embedding.ndim == 2 and embedding.shape[0] == 1:
                return embedding[0]
            return embedding
        emb_np = embedding.cpu().numpy() if isinstance(embedding, torch.Tensor) else embedding
        if hasattr(emb_np, "ndim") and emb_np.ndim == 2 and emb_np.shape[0] == 1:
            emb_np = emb_np[0]
        return emb_np
    
    def batch_embed(self, text_units, batch_size=256, normalize=True, progress=True, role: str = "passage"):
        """Embeds multiple text units at once (minimal SentenceTransformers fast path)."""
        texts = self._prepare_texts([tu.text for tu in text_units], role=role)
        if not texts:
            return text_units

        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            device=self.device,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=progress,
        ).astype(np.float32, copy=False)

        for tu, vec in zip(text_units, embs):
            tu.embedding = vec
        return text_units

class TextProcessor:
    """Handles tokenization and chunking."""

    def __init__(self, chunk_size=512, overlap=50, tokenizer_name: str | None = None):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer_name = tokenizer_name or C.CHUNKING_TOKENIZER_MODEL
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.tokenizer_name)
        
    def tokenize(self, text):
        """Tokenizes text and returns token IDs."""
        return self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')['input_ids'][0]

    def tokenize_and_chunk(self, document):
        """Chunks a Document into TextUnits."""

        chunks = [] # will be a list of TextUnits

        # We tokenize and create the appropriate chunks according to self.chunk_size and self.overlap
        tokens = self.tokenize(document.text)
        chunks = []

        for i in range(0, len(tokens), self.chunk_size - self.overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens = True)
            chunks.append(TextUnit(
                doc_id = document.doc_id, 
                chunk_id = i,
                text = chunk_text, 
                token_count=len(chunk_tokens)
            ))

        return chunks

    def get_chunking_metadata(self):
        return {
            "chunking_version": C.CHUNKING_VERSION,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "tokenizer_model": self.tokenizer_name,
        }


class ChunkedCorpusCache:
    METADATA_FILENAME = "metadata.json"
    DOCUMENTS_FILENAME = "documents.pkl"
    CACHE_FORMAT = "chunked_corpus_v1"

    @classmethod
    def metadata_path(cls, cache_dir: str | Path) -> Path:
        return Path(cache_dir) / cls.METADATA_FILENAME

    @classmethod
    def documents_path(cls, cache_dir: str | Path) -> Path:
        return Path(cache_dir) / cls.DOCUMENTS_FILENAME

    @classmethod
    def exists(cls, cache_dir: str | Path) -> bool:
        return (
            cls.metadata_path(cache_dir).exists()
            and cls.documents_path(cache_dir).exists()
        )

    @classmethod
    def build_metadata(
        cls,
        dataset: Dataset,
        *,
        chunking_version: str,
        chunk_size: int,
        overlap: int,
        tokenizer_model: str,
        creation_source: str,
    ):
        dataset.gather_all_text_units()
        return {
            "cache_format": cls.CACHE_FORMAT,
            "dataset_name": dataset.dataset_name,
            "chunking_version": chunking_version,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "tokenizer_model": tokenizer_model,
            "creation_source": creation_source,
            "num_documents": len(dataset.documents),
            "num_text_units": len(dataset.all_text_units),
        }

    @classmethod
    def save(
        cls,
        cache_dir: str | Path,
        dataset: Dataset,
        metadata: dict,
        show_progress: bool = False,
        progress_desc: str = "Serializing chunk cache documents",
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        dataset.gather_all_text_units()
        documents_iter = dataset.documents
        if show_progress:
            documents_iter = tqdm(
                dataset.documents,
                desc=progress_desc,
                unit="doc",
                leave=False,
            )
        serialized_documents = [
            document.to_dict(include_text_units=True)
            for document in documents_iter
        ]

        cls.metadata_path(cache_dir).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        with cls.documents_path(cache_dir).open("wb") as f:
            pickle.dump(
                {"documents": serialized_documents},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, cache_dir: str | Path):
        cache_dir = Path(cache_dir)
        metadata = json.loads(cls.metadata_path(cache_dir).read_text())
        with cls.documents_path(cache_dir).open("rb") as f:
            payload = pickle.load(f)

        documents = [
            Document.from_dict(document_payload)
            for document_payload in payload["documents"]
        ]
        dataset = Dataset(metadata["dataset_name"], documents=documents)
        dataset.rebuild_document_index()
        dataset.gather_all_text_units()

        expected_docs = metadata.get("num_documents")
        expected_text_units = metadata.get("num_text_units")
        if expected_docs is not None and expected_docs != len(dataset.documents):
            raise ValueError(
                f"Chunk cache document count mismatch at {cache_dir}: "
                f"metadata={expected_docs}, loaded={len(dataset.documents)}"
            )
        if expected_text_units is not None and expected_text_units != len(dataset.all_text_units):
            raise ValueError(
                f"Chunk cache text unit count mismatch at {cache_dir}: "
                f"metadata={expected_text_units}, loaded={len(dataset.all_text_units)}"
            )

        return dataset, metadata

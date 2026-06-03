from tqdm import tqdm

class Dataset:
    """Holds multiple documents, like a collection of articles or Wikipedia entries."""

    def __init__(self, dataset_name, documents=None):
        self.dataset_name = dataset_name
        self.documents = documents or []
        self.document_index = {}
        self.all_text_units = []

    def add_document(self, document):
        """Adds a document to the dataset."""
        self.documents.append(document)
        self.document_index[document.doc_id] = len(self.documents)-1

    def rebuild_document_index(self):
        self.document_index = {
            document.doc_id: idx for idx, document in enumerate(self.documents)
        }
    
    def retrieve_document_by_id(self, doc_id):
        return self.documents[self.document_index[doc_id]]
    
    def tokenize_and_chunk_documents(self, text_processor):
        print('Tokenizing documents and splitting into chunks')

        for document in tqdm(self.documents):
            document.tokenize_and_chunk(text_processor)
    
    def embed_chunks(self, embedder):
        print('Embedding text chunks')
        for document in tqdm(self.documents):
            document.embed_chunks(embedder)
    
    def fast_embed_chunks(self, embedder):
        print('Embedding all text units')
        self.gather_all_text_units()
        embedder.batch_embed(self.all_text_units)
    
    def gather_all_text_units(self):
        result = []
        for document in self.documents:
            result += (document.text_units or [])
        
        self.all_text_units = result

    def attach_text_units(self, text_units, strict=True):
        grouped = {}
        for text_unit in text_units:
            grouped.setdefault(text_unit.doc_id, []).append(text_unit)

        missing_doc_ids = set(grouped.keys()) - {document.doc_id for document in self.documents}
        if strict and missing_doc_ids:
            raise ValueError(
                f"Text units reference unknown document IDs: {sorted(missing_doc_ids)[:10]}"
            )

        for document in self.documents:
            units = grouped.get(document.doc_id, [])
            document.text_units = sorted(units, key=lambda tu: tu.chunk_id)

        self.gather_all_text_units()
    

    def __repr__(self):
        return f"Dataset(name={self.dataset_name}, documents={len(self.documents)})"

class Document:
    """Represents a full document with metadata."""

    def __init__(self, doc_id, text, metadata=None, no_chunking=False):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata or {}
        self.text_units = None

        if no_chunking:
            self.text_units = [
                TextUnit(doc_id=doc_id,
                         chunk_id=0,
                         text=self.text,
                         token_count=len(self.text)
                         )]
    
    def tokenize_and_chunk(self, text_processor):
        # Using a TextProcessor for tokenization and chunking
        self.text_units = text_processor.tokenize_and_chunk(self)

    
    def embed_chunks(self, embedder):
        # Using an Embedder to create embeddings
        embedder.batch_embed(self.text_units)

    def __repr__(self):
        return f"Document(id={self.doc_id}, length={len(self.text)} chars, metadata={self.metadata})"

    def to_dict(self, include_text_units=True):
        payload = {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
        }
        if include_text_units:
            payload["text_units"] = [
                text_unit.to_dict() for text_unit in (self.text_units or [])
            ]
        return payload

    @classmethod
    def from_dict(cls, payload):
        document = cls(
            doc_id=payload["doc_id"],
            text=payload["text"],
            metadata=payload.get("metadata"),
        )
        text_units = payload.get("text_units")
        if text_units is not None:
            document.text_units = [
                TextUnit.from_dict(text_unit_payload)
                for text_unit_payload in text_units
            ]
        return document


class TextUnit:
    """Represents a chunk of a document after tokenization, with an embedding."""

    def __init__(self, doc_id, chunk_id, text, token_count, embedding=None):
        self.doc_id = doc_id
        self.chunk_id = chunk_id    # chunk_id within a given document
        self.text = text
        self.token_count = token_count
        self.embedding = embedding  # Vector representation (default: None)

    def __repr__(self):
        return f"TextUnit(doc_id={self.doc_id}, chunk_id={self.chunk_id}, tokens={self.token_count}, embedded={self.embedding is not None})"

    def to_dict(self, include_embedding=False):
        payload = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "token_count": self.token_count,
        }
        if include_embedding:
            payload["embedding"] = self.embedding
        return payload

    @classmethod
    def from_dict(cls, payload):
        return cls(
            doc_id=payload["doc_id"],
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            token_count=payload["token_count"],
            embedding=payload.get("embedding"),
        )


class Questions:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.questions = []
    
    def add_question(self, question, answer=None, target_chunks=None):
        question = {'question': question, 'answer': answer, 'target':target_chunks}
        self.questions.append(question)
    
    def __repr__(self):
        return f"Questions(name={self.dataset_name})"

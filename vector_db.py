import faiss
import json
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path

class FaissVectorDB:
    def __init__(self, dim, metric="dot"):
        """
        Initialize a FAISS vector database.
        
        Args:
        - dim (int): Dimension of embeddings.
        - metric (str): Distance metric ('l2' for Euclidean, 'dot' for Dot Product).
        """
        self.dim = dim
        self.metric = metric.lower()
        self.text_units = {}  # Store text units with their corresponding IDs
        
        if self.metric == "l2":
            self._index_cpu = faiss.IndexFlatL2(dim)  # Euclidean Distance
        elif self.metric == "dot":
            self._index_cpu = faiss.IndexFlatIP(dim)  # Inner Product (Dot Product)
        else:
            raise ValueError("Invalid metric! Choose 'l2' or 'dot'.")
        
        self.index = self._index_cpu
        self.id_counter = 0  # Unique ID tracker for text units
        
    def add_text_units(self, text_units):
        """
        Add embeddings and corresponding text units to the FAISS index.
        
        Args:
        - text_units (List[str]): List of text units for which the embeddings have been generated.
        """
        embeddings = np.array([t.embedding for t in text_units], dtype=np.float32)

        # Assign a unique ID to each text unit
        num_items = embeddings.shape[0]
        ids = np.arange(self.id_counter, self.id_counter + num_items)

        # Store text units
        for i, text in zip(ids, text_units):
            self.text_units[i] = text

        # Add to FAISS index
        self.index.add(embeddings)
        self.id_counter += num_items  # Update counter
    
    def add_dataset(self, dataset):
        """
        Accepts a Dataset and adds the text units of all documents.
        """
        print('Indexing dataset')
        for document in tqdm(dataset.documents):
            self.add_text_units(document.text_units)
        
    def search(self, query_embedding, k=5):
        """
        Search for the top-k nearest text units based on query embedding.
        
        Args:
        - query_embedding (np.ndarray): The query embedding.
        - k (int): Number of nearest neighbors to return.
        
        Returns:
        - List of (text_unit, score) tuples.
        """
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve text units
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Ensure it's a valid index
                results.append((self.text_units[idx], dist))

        return results
    
    def batch_search(self, query_embeddings, k = 5):
        """
            Batch nearest-neighbor search for many queries at once.

            Returns a list of length len(query_embeddings) that contains lists of (text_unit, score) tuples.
        """
        Q = np.asarray(query_embeddings, dtype=np.float32)
        distances, indices = self.index.search(Q, k)

        results = []
        for r in range(indices.shape[0]):
            row = []
            for idx, dist in zip(indices[r], distances[r]):
                if idx != -1:
                    row.append((self.text_units[int(idx)], float(dist)))
            results.append(row)
        return results
    
    def save(self, folder: str | Path):
        """
        Save FAISS index + class data into a directory.
        Creates the folder if it doesn't exist.
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(folder / "index.faiss"))

        sidecar = {
            "dim": self.dim,
            "metric": self.metric,
            "id_counter": self.id_counter,
        }
        (folder / "sidecar.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2)
        )

        with (folder / "text_units.pkl").open("wb") as f:
            pickle.dump(self.text_units, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def load(cls, folder: str | Path) -> "FaissVectorDB":
        """
        Load FAISS index + side data from a directory created by .save().
        """
        folder = Path(folder)

        index = faiss.read_index(str(folder / "index.faiss"))

        sidecar = json.loads((folder / "sidecar.json").read_text())
        dim = sidecar["dim"]
        metric = sidecar["metric"]
        id_counter = sidecar.get("id_counter", None)

        # Construct object and attach the loaded FAISS index
        obj = cls(dim=dim, metric=metric)
        obj.index = index

        with (folder / "text_units.pkl").open("rb") as f:
            obj.text_units = pickle.load(f)

        if id_counter is None:
            obj.id_counter = obj.index.ntotal
        else:
            obj.id_counter = id_counter
            
        return obj

    @staticmethod
    def load_text_units(folder: str | Path):
        """
        Load only the text units from a saved FAISS folder.
        Returns a list of TextUnit objects in deterministic ID order.
        """
        folder = Path(folder)
        with (folder / "text_units.pkl").open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            items = sorted(payload.items(), key=lambda x: x[0])
            return [tu for _idx, tu in items]
        if isinstance(payload, list):
            return payload
        # Fallback: wrap single object
        return list(payload)

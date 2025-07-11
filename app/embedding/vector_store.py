import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from app.utils.id_utils import IDUtils
from app.utils.logging_config import setup_logging
from typing import Optional, List, Dict, Any
import json

class VectorStore:
    def __init__(self, db_path: str = "faiss_db", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize FAISS indices and embedding model."""
        self.db_path = db_path
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.logger = setup_logging()

        self.collections = {
            "chunks": {
                "index_file": os.path.join(db_path, "chunks_index.bin"),
                "metadata_file": os.path.join(db_path, "chunks_metadata.pkl"),
                "index": None,
                "metadata_store": {},
                "id_to_index": {},
                "index_to_id": {}
            },
            "keywords": {
                "index_file": os.path.join(db_path, "keywords_index.bin"),
                "metadata_file": os.path.join(db_path, "keywords_metadata.pkl"),
                "index": None,
                "metadata_store": {},
                "id_to_index": {},
                "index_to_id": {}
            },
            "graph": {
                "index_file": os.path.join(db_path, "graph_index.bin"),
                "metadata_file": os.path.join(db_path, "graph_metadata.pkl"),
                "index": None,
                "metadata_store": {},
                "id_to_index": {},
                "index_to_id": {}
            }
        }
        os.makedirs(db_path, exist_ok=True)
        self._load_or_create_all_collections()

    def _load_or_create_collection(self, collection_name: str):
        """Load or create a single FAISS index and metadata for a collection."""
        collection = self.collections[collection_name]
        if os.path.exists(collection["index_file"]) and os.path.exists(collection["metadata_file"]):
            try:
                collection["index"] = faiss.read_index(collection["index_file"])
                with open(collection["metadata_file"], 'rb') as f:
                    data = pickle.load(f)
                    collection["metadata_store"] = data.get('metadata_store', {})
                    collection["id_to_index"] = data.get('id_to_index', {})
                    collection["index_to_id"] = data.get('index_to_id', {})
                self.logger.info(f"Loaded {collection_name} index with {collection['index'].ntotal} vectors")
            except Exception as e:
                self.logger.error(f"Error loading {collection_name} index: {e}")
                collection["index"] = faiss.IndexFlatL2(self.embedding_dim)
                collection["metadata_store"] = {}
                collection["id_to_index"] = {}
                collection["index_to_id"] = {}
        else:
            collection["index"] = faiss.IndexFlatL2(self.embedding_dim)
            collection["metadata_store"] = {}
            collection["id_to_index"] = {}
            collection["index_to_id"] = {}
            self.logger.info(f"Created new {collection_name} index")

    def _load_or_create_all_collections(self):
        """Load or create all FAISS indices and metadata."""
        for collection_name in self.collections:
            self._load_or_create_collection(collection_name)

    def _save_index(self, collection_name: str):
        """Save FAISS index and metadata for a collection."""
        collection = self.collections[collection_name]
        faiss.write_index(collection["index"], collection["index_file"])
        with open(collection["metadata_file"], 'wb') as f:
            pickle.dump({
                'metadata_store': collection["metadata_store"],
                'id_to_index': collection["id_to_index"],
                'index_to_id': collection["index_to_id"]
            }, f)
        self.logger.debug(f"Saved {collection_name} index")

    def index_vectors(self, embeddings: list, texts: List[str], chunk_ids: List[str],
                      novel_ids: List[str], metadatas: Optional[List[Dict]] = None,
                      collection_name: str = "chunks") -> List[str]:
        """Index embeddings with corresponding texts, chunk_ids, and novel_ids."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        collection = self.collections[collection_name]

        vector_ids = [IDUtils.generate_chunk_id() for _ in texts]

        # Ensure embeddings is a 2D NumPy array
        embeddings_array = np.array(embeddings, dtype='float32')
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)  # Handle single embedding
        elif embeddings_array.ndim > 2:
            embeddings_array = embeddings_array.squeeze()  # Remove extra dimensions
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)

        # Verify shape
        if embeddings_array.ndim != 2 or embeddings_array.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected 2D embeddings array with shape (n, {self.embedding_dim}), got shape {embeddings_array.shape}")

        current_size = collection["index"].ntotal
        collection["index"].add(embeddings_array)

        for i, vector_id in enumerate(vector_ids):
            faiss_idx = current_size + i
            document = {"text": texts[i]} if collection_name == "chunks" else \
                {"keywords": texts[i]} if collection_name == "keywords" else \
                    {"label": texts[i]}
            collection["metadata_store"][faiss_idx] = {
                "id": vector_id,
                "document": json.dumps(document),
                "metadata": {**metadatas[i], "chunk_id": chunk_ids[i], "novel_id": novel_ids[i]},
                "chunk_id": chunk_ids[i] if collection_name != "chunks" else None
            }
            collection["id_to_index"][vector_id] = faiss_idx
            collection["index_to_id"][faiss_idx] = vector_id

        self._save_index(collection_name)
        self.logger.info(f"Indexed {len(vector_ids)} vectors in {collection_name}")
        return vector_ids
    def query_vectors(self, query_text: str, novel_id: Optional[str] = None,
                     n_results: int = 5, collection_name: str = "chunks") -> List[Dict]:
        """Query vectors by text and optionally filter by novel_id."""
        collection = self.collections[collection_name]
        if collection["index"].ntotal == 0:
            self.logger.debug(f"No vectors in {collection_name} index")
            return []

        query_embedding = self.embedding_model.encode([query_text]).astype('float32')
        search_k = min(collection["index"].ntotal, n_results * 10 if novel_id else n_results)
        distances, indices = collection["index"].search(query_embedding, search_k)

        results = []
        for distance, faiss_idx in zip(distances[0], indices[0]):
            if faiss_idx == -1:
                continue
            vector_id = collection["index_to_id"].get(faiss_idx)
            if not vector_id or faiss_idx not in collection["metadata_store"]:
                continue
            stored_data = collection["metadata_store"][faiss_idx]
            metadata = stored_data['metadata']
            if novel_id and metadata.get('novel_id') != novel_id:
                continue
            results.append({
                "vector_id": vector_id,
                "text": json.loads(stored_data['document']).get("text", stored_data['document']),
                "metadata": metadata,
                "distance": float(distance)
            })
            if len(results) >= n_results:
                break
        self.logger.debug(f"Query returned {len(results)} results from {collection_name}")
        return results

    def delete_vectors(self, vector_ids: List[str], collection_name: str = "chunks") -> bool:
        """Delete vectors by their IDs."""
        collection = self.collections[collection_name]
        try:
            for vector_id in vector_ids:
                if vector_id in collection["id_to_index"]:
                    del collection["metadata_store"][collection["id_to_index"][vector_id]]
                    del collection["id_to_index"][vector_id]
            if collection["metadata_store"]:
                texts = [json.loads(data['document']).get("text", data['document']) for data in collection["metadata_store"].values()]
                metadatas = [data['metadata'] for data in collection["metadata_store"].values()]
                chunk_ids = [meta['chunk_id'] for meta in metadatas]
                novel_ids = [meta['novel_id'] for meta in metadatas]
                embeddings = self.embedding_model.encode(texts).astype('float32')
                collection["index"] = faiss.IndexFlatL2(self.embedding_dim)
                collection["index"].add(embeddings)
                collection["id_to_index"] = {vid: i for i, vid in enumerate(collection["metadata_store"].keys())}
                collection["index_to_id"] = {i: vid for i, vid in enumerate(collection["metadata_store"].keys())}
            else:
                collection["index"] = faiss.IndexFlatL2(self.embedding_dim)
                collection["id_to_index"] = {}
                collection["index_to_id"] = {}
            self._save_index(collection_name)
            self.logger.info(f"Deleted {len(vector_ids)} vectors from {collection_name}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting vectors from {collection_name}: {e}")
            return False

    def get_vector_count(self, collection_name: str = "chunks") -> int:
        """Get the total number of vectors in the index."""
        return self.collections[collection_name]["index"].ntotal

    def clear_index(self, collection_name: str = "chunks"):
        """Clear all vectors from the index."""
        collection = self.collections[collection_name]
        collection["index"] = faiss.IndexFlatL2(self.embedding_dim)
        collection["metadata_store"] = {}
        collection["id_to_index"] = {}
        collection["index_to_id"] = {}
        self._save_index(collection_name)
        self.logger.info(f"Cleared {collection_name} index")
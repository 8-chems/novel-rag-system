import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid
from app.utils.logging_config import setup_logging
from app.embedding.vector_store import VectorStore

class DatabaseUtils:
    def __init__(self, db_path="faiss_db", model_name="all-MiniLM-L6-v2"):
        """Initialize database with VectorStore for indexing."""
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger = setup_logging()

        self.collections = {
            "chunks": {"index_path": self.db_path / "chunks_index.bin", "metadata_path": self.db_path / "chunks_metadata.pkl"},
            "keywords": {"index_path": self.db_path / "keywords_index.bin", "metadata_path": self.db_path / "keywords_metadata.pkl"},
            "graph": {"index_path": self.db_path / "graph_index.bin", "metadata_path": self.db_path / "graph_metadata.pkl"}
        }
        self.vector_store = VectorStore(db_path=db_path, model_name=model_name)
        self._load_or_create_all_collections()

    def _load_or_create_collection(self, collection_name: str):
        """Placeholder to ensure collection files exist (handled by VectorStore)."""
        collection = self.collections[collection_name]
        if not (collection["index_path"].exists() and collection["metadata_path"].exists()):
            self.logger.info(f"Creating new {collection_name} collection")
            self.vector_store.clear_index(collection_name=collection_name)

    def _load_or_create_all_collections(self):
        """Load or create all collections."""
        for collection_name in self.collections:
            self._load_or_create_collection(collection_name)

    def add_chunk(self, text, metadata=None, collection_name="chunks", embedding=None):
        """Add a single chunk to the specified collection."""
        if metadata is None:
            metadata = {}
        chunk_id = str(uuid.uuid4())
        if embedding is None:
            embedding = self.model.encode([text])
        else:
            embedding = np.array([embedding])
        vector_ids = self.vector_store.index_vectors(
            embeddings=[embedding],
            texts=[text],
            chunk_ids=[chunk_id],
            novel_ids=[metadata.get("novel_id", "")],
            metadatas=[metadata],
            collection_name=collection_name
        )
        self.logger.info(f"Added chunk {chunk_id} to {collection_name}")
        return chunk_id

    def add_chunk(self, text, metadata=None, collection_name="chunks", embedding=None):
        """Add a single chunk to the specified collection."""
        if metadata is None:
            metadata = {}
        chunk_id = str(uuid.uuid4())
        if embedding is None:
            embedding = self.model.encode([text])[0]  # Get single embedding
        else:
            embedding = np.array(embedding)  # Ensure embedding is a NumPy array
        vector_ids = self.vector_store.index_vectors(
            embeddings=[embedding],
            texts=[text],
            chunk_ids=[chunk_id],
            novel_ids=[metadata.get("novel_id", "")],
            metadatas=[metadata],
            collection_name=collection_name
        )
        self.logger.info(f"Added chunk {chunk_id} to {collection_name}")
        return chunk_id


    def query(self, query_text, n_results=5, collection_name="chunks"):
        """Query the specified collection."""
        results = self.vector_store.query_vectors(
            query_text=query_text,
            n_results=n_results,
            collection_name=collection_name
        )
        return {
            "documents": [[r["text"] for r in results]],
            "metadatas": [[r["metadata"] for r in results]],
            "distances": [[r["distance"] for r in results]],
            "ids": [[r["vector_id"] for r in results]]
        }

    def get_chunk_by_id(self, chunk_id, collection_name="chunks"):
        """Retrieve a chunk by its ID."""
        results = self.vector_store.query_vectors("", collection_name=collection_name)
        for result in results:
            if result["metadata"].get("chunk_id") == chunk_id:
                return {
                    "documents": [result["text"]],
                    "metadatas": [result["metadata"]],
                    "ids": [result["vector_id"]]
                }
        return {"documents": [], "metadatas": [], "ids": []}

    def delete_by_id(self, chunk_id, collection_name="chunks"):
        """Delete a chunk by its ID."""
        results = self.vector_store.query_vectors("", collection_name=collection_name)
        vector_id = None
        for result in results:
            if result["metadata"].get("chunk_id") == chunk_id:
                vector_id = result["vector_id"]
                break
        if vector_id:
            return self.vector_store.delete_vectors([vector_id], collection_name=collection_name)
        return False

    def count_chunks(self, collection_name="chunks"):
        """Return the total number of chunks in the collection."""
        return self.vector_store.get_vector_count(collection_name=collection_name)

    def delete_collection(self, collection_name="chunks"):
        """Delete the entire collection."""
        self.vector_store.clear_index(collection_name=collection_name)
        self.logger.info(f"Deleted {collection_name} collection")

    def reset_collection(self, collection_name="chunks"):
        """Reset the collection."""
        self.delete_collection(collection_name=collection_name)
from keybert import KeyBERT
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from app.utils.id_utils import IDUtils
from app.utils.logging_config import setup_logging
from typing import Optional, Dict, List


class KeywordExtractor:
    def __init__(self, db_path: str = "faiss_db"):
        """Initialize KeyBERT model, FAISS index, and logger."""
        self.kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.db_path = db_path
        self.index_path = os.path.join(db_path, "faiss_index.bin")
        self.metadata_path = os.path.join(db_path, "metadata.pkl")

        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)

        # Initialize FAISS index and metadata storage
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension
        self.index = None
        self.metadata = {}  # Store metadata and documents by index position
        self.id_to_position = {}  # Map IDs to FAISS index positions

        self._load_or_create_index()
        self.logger = setup_logging()

    def _load_or_create_index(self):
        """Load existing FAISS index or create a new one."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            # Load existing index
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data.get('metadata', {})
                self.id_to_position = data.get('id_to_position', {})
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            self.id_to_position = {}

    def _save_index(self):
        """Save FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_position': self.id_to_position
            }, f)

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top_n keywords from text using KeyBERT."""
        self.logger.debug(f"Extracting keywords from text: {text[:50]}...")
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            diversity=0.7
        )
        keywords = [kw[0] for kw in keywords]
        self.logger.debug(f"Extracted keywords: {keywords}")
        return keywords

    def process_and_store_keywords(self, chunk_id: str, text: str, metadata: Optional[Dict] = None) -> tuple[
        str, List[str]]:
        """Extract keywords from text and store in FAISS."""
        if metadata is None:
            metadata = {}

        keywords = self.extract_keywords(text)
        keyword_id = IDUtils.generate_keyword_id()

        self.logger.info(f"Storing keywords for chunk {chunk_id}: {keywords}")

        # Create embedding for the keywords
        keywords_text = ", ".join(keywords)
        embedding = self.embedding_model.encode([keywords_text])

        # Add to FAISS index
        current_position = self.index.ntotal
        self.index.add(embedding.astype('float32'))

        # Store metadata
        self.metadata[current_position] = {
            "chunk_id": chunk_id,
            "keyword_id": keyword_id,
            "keywords": keywords_text,
            "metadata": metadata
        }
        self.id_to_position[keyword_id] = current_position

        # Save to disk
        self._save_index()

        return keyword_id, keywords

    def get_keywords_by_chunk_id(self, chunk_id: str) -> List[Dict]:
        """Retrieve keywords associated with a specific chunk ID."""
        self.logger.debug(f"Retrieving keywords for chunk {chunk_id}")

        results = []
        for position, data in self.metadata.items():
            if data["chunk_id"] == chunk_id:
                results.append({
                    "keyword_id": data["keyword_id"],
                    "keywords": data["keywords"],
                    "metadata": data["metadata"]
                })

        return results

    def search_similar_keywords(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar keywords using FAISS similarity search."""
        if self.index.ntotal == 0:
            return []

        # Create embedding for query
        query_embedding = self.embedding_model.encode([query_text])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in self.metadata:
                result = self.metadata[idx].copy()
                result["distance"] = float(distance)
                results.append(result)

        return results

    def delete_keywords_by_chunk_id(self, chunk_id: str):
        """Delete keywords associated with a specific chunk ID."""
        # Note: FAISS doesn't support efficient deletion, so we'll mark as deleted
        # and rebuild index periodically if needed
        positions_to_delete = []
        for position, data in self.metadata.items():
            if data["chunk_id"] == chunk_id:
                positions_to_delete.append(position)

        for position in positions_to_delete:
            keyword_id = self.metadata[position]["keyword_id"]
            del self.metadata[position]
            if keyword_id in self.id_to_position:
                del self.id_to_position[keyword_id]

        self._save_index()
        self.logger.info(f"Marked keywords for chunk {chunk_id} as deleted")

    def get_index_stats(self) -> Dict:
        """Get statistics about the FAISS index."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.index.d,
            "active_metadata_entries": len(self.metadata)
        }


if __name__ == "__main__":
    extractor = KeywordExtractor()

    sample_text = "Natural language processing enables computers to understand human language."
    chunk_id = IDUtils.generate_chunk_id()

    keyword_id, keywords = extractor.process_and_store_keywords(
        chunk_id, sample_text, {"filename": "sample.txt"}
    )

    print(f"Stored keywords: {keywords} with ID {keyword_id}")

    retrieved = extractor.get_keywords_by_chunk_id(chunk_id)
    for result in retrieved:
        print(f"Retrieved: {result}")

    # Test similarity search
    similar = extractor.search_similar_keywords("computer language processing")
    print(f"Similar keywords: {similar}")

    # Print index stats
    print(f"Index stats: {extractor.get_index_stats()}")
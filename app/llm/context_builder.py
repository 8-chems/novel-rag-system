import faiss
import os
import json
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict
import logging
import pickle


class ContextBuilder:
    def __init__(self, db_path: str = "faiss_db"):
        """Initialize FAISS indices and collections."""
        self.db_path = db_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        self.logger = logging.getLogger(__name__)

        self.collections = {
            "chunks": {
                "index_path": os.path.join(db_path, "chunks_index.bin"),
                "metadata_path": os.path.join(db_path, "chunks_metadata.pkl"),
                "index": None,
                "metadata": {},
                "id_to_position": {}
            },
            "keywords": {
                "index_path": os.path.join(db_path, "keywords_index.bin"),
                "metadata_path": os.path.join(db_path, "keywords_metadata.pkl"),
                "index": None,
                "metadata": {},
                "id_to_position": {}
            },
            "graph": {
                "index_path": os.path.join(db_path, "graph_index.bin"),
                "metadata_path": os.path.join(db_path, "graph_metadata.pkl"),
                "index": None,
                "metadata": {},
                "id_to_position": {}
            }
        }
        os.makedirs(db_path, exist_ok=True)
        self._load_or_create_all_collections()

    def _load_or_create_collection(self, collection_name: str):
        """Load existing FAISS index or create a new one for a collection."""
        collection = self.collections[collection_name]
        if os.path.exists(collection["index_path"]) and os.path.exists(collection["metadata_path"]):
            try:
                collection["index"] = faiss.read_index(collection["index_path"])
                with open(collection["metadata_path"], 'rb') as f:
                    data = pickle.load(f)
                    collection["metadata"] = data.get('metadata_store', {})
                    collection["id_to_position"] = data.get('id_to_index', {})
                self.logger.info(f"Loaded {collection_name} index with {collection['index'].ntotal} vectors")
            except Exception as e:
                self.logger.error(f"Error loading {collection_name} index: {e}")
                collection["index"] = faiss.IndexFlatL2(self.dimension)
                collection["metadata"] = {}
                collection["id_to_position"] = {}
        else:
            collection["index"] = faiss.IndexFlatL2(self.dimension)
            collection["metadata"] = {}
            collection["id_to_position"] = {}
            self.logger.info(f"Created new {collection_name} index")

    def _load_or_create_all_collections(self):
        """Load or create all collections."""
        for collection_name in self.collections:
            self._load_or_create_collection(collection_name)

    def get_chunk_data(self, chunk_id: str) -> Dict:
        """Retrieve chunk data by chunk_id."""
        collection = self.collections["chunks"]
        for pos, data in collection["metadata"].items():
            if data.get("id") == chunk_id or data.get("metadata", {}).get("chunk_id") == chunk_id:
                doc = json.loads(data["document"]) if "document" in data else {}
                return {
                    "chunk_id": chunk_id,
                    "text": doc.get("text", ""),
                    "metadata": data.get("metadata", {})
                }
        return {}

    def get_keywords(self, chunk_id: str) -> List[str]:
        """Retrieve keywords associated with a chunk_id."""
        collection = self.collections["keywords"]
        keywords = []
        for pos, data in collection["metadata"].items():
            if data.get("metadata", {}).get("chunk_id") == chunk_id:
                doc = json.loads(data["document"]) if "document" in data else {}
                keywords.extend(doc.get("keywords", "").split(", "))
        return keywords

    def get_graph_data(self, chunk_id: str) -> Dict:
        """Retrieve graph nodes and relationships for a chunk_id."""
        collection = self.collections["graph"]
        nodes = []
        relationships = []
        for pos, data in collection["metadata"].items():
            if data.get("metadata", {}).get("chunk_id") == chunk_id:
                doc = json.loads(data["document"]) if "document" in data else {}
                metadata = data.get("metadata", {})
                if metadata.get("data_type") == "graph_node":
                    nodes.append({
                        "id": metadata.get("id", ""),
                        "type": metadata.get("type", ""),
                        "label": doc.get("label", "")
                    })
                elif metadata.get("data_type") == "graph_relationship":
                    relationships.append({
                        "id": metadata.get("id", ""),
                        "label": doc.get("label", "")
                    })
        return {"nodes": nodes, "relationships": relationships}

    def build_context(self, chunk_ids: List[str], novel_id: Optional[str] = None) -> str:
        """Build context string from chunks, keywords, and graph data for specified chunk_ids."""
        context_parts = []

        for chunk_id in chunk_ids:
            # Retrieve chunk data
            chunk_data = self.get_chunk_data(chunk_id)
            if not chunk_data:
                self.logger.warning(f"No chunk data found for chunk_id: {chunk_id}")
                continue
            if novel_id and chunk_data.get("metadata", {}).get("novel_id") != novel_id:
                self.logger.debug(f"Skipping chunk_id {chunk_id} (novel_id mismatch)")
                continue

            # Add chunk text
            context_parts.append(f"Chunk ID: {chunk_id}")
            context_parts.append(f"Text: {chunk_data.get('text', '')}")

            # Retrieve and add keywords
            keywords = self.get_keywords(chunk_id)
            if keywords:
                context_parts.append(f"Keywords: {', '.join(keywords)}")
            else:
                context_parts.append("Keywords: None")

            # Retrieve and add graph data
            graph_data = self.get_graph_data(chunk_id)
            if graph_data["nodes"] or graph_data["relationships"]:
                context_parts.append("Knowledge Graph:")
                if graph_data["nodes"]:
                    context_parts.append("  Nodes:")
                    for node in graph_data["nodes"]:
                        context_parts.append(f"    - {node['type']} (ID: {node['id']}): {node['label']}")
                if graph_data["relationships"]:
                    context_parts.append("  Relationships:")
                    for rel in graph_data["relationships"]:
                        context_parts.append(f"    - Relationship (ID: {rel['id']}): {rel['label']}")
            else:
                context_parts.append("Knowledge Graph: None")

        context = "\n".join(context_parts) if context_parts else "No relevant context found."
        self.logger.debug(f"Built context:\n{context}")
        return context

    def search_similar_chunks(self, query: str, top_k: int = 5, novel_id: Optional[str] = None) -> List[Dict]:
        """Search for similar chunks based on query text."""
        collection = self.collections["chunks"]
        if collection["index"].ntotal == 0:
            self.logger.debug("No chunks in index")
            return []

        query_embedding = self.embedding_model.encode([query]).astype('float32')
        search_k = min(collection["index"].ntotal, top_k * 10 if novel_id else top_k)
        distances, indices = collection["index"].search(query_embedding, search_k)

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            vector_id = collection["id_to_position"].get(idx)
            if not vector_id or idx not in collection["metadata"]:
                continue
            stored_data = collection["metadata"][idx]
            metadata = stored_data.get("metadata", {})
            if novel_id and metadata.get("novel_id") != novel_id:
                continue
            doc = json.loads(stored_data["document"]) if "document" in stored_data else {}
            results.append({
                "chunk_id": stored_data.get("id", ""),
                "text": doc.get("text", ""),
                "metadata": metadata,
                "distance": float(distance)
            })
            if len(results) >= top_k:
                break
        self.logger.debug(f"Found {len(results)} similar chunks")
        return results

    def get_collection_stats(self) -> Dict:
        """Return stats for all collections."""
        stats = {}
        for collection_name, collection in self.collections.items():
            stats[collection_name] = {
                "vector_count": collection["index"].ntotal if collection["index"] else 0,
                "index_path": collection["index_path"],
                "metadata_path": collection["metadata_path"]
            }
        return stats

    def build_qa_prompt(self, question: str, chunk_ids: List[str], novel_id: Optional[str] = None) -> str:
        """Build a QA prompt with context."""
        context = self.build_context(chunk_ids, novel_id)
        prompt = f"""
        Context:
        {context}

        Question: {question}

        Answer the question based on the provided context. If the context is insufficient, say so.
        """
        return prompt
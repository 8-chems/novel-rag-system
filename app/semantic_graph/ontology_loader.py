from rdflib import Graph, Namespace, RDF, RDFS, OWL
import faiss
import numpy as np
import pickle
import os
import json
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, List


class OntologyLoader:
    def __init__(self, db_path: str = "faiss_db"):
        """Initialize FAISS client and RDFLib graph."""
        self.db_path = db_path
        self.index_path = os.path.join(db_path, "graph_index.bin")
        self.metadata_path = os.path.join(db_path, "graph_metadata.pkl")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384  # all-MiniLM-L6-v2 embedding dimension

        # Initialize FAISS index and metadata storage
        self.index = None
        self.metadata = {}  # Store metadata and documents by index position
        self.id_to_position = {}  # Map IDs to FAISS index positions

        self._load_or_create_index()

        # Initialize RDFLib graph
        self.graph = Graph()
        self.ns = Namespace("http://example.org/novel-ontology#")
        self.graph.bind("novel", self.ns)

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

    def get_all_graph_data(self) -> List[Dict]:
        """Retrieve all graph data from FAISS storage."""
        results = []

        for position, data in self.metadata.items():
            # Only include graph-related data
            if data.get("data_type") in ["node", "relationship"]:
                parsed_data = json.loads(data["document"])
                result = {
                    "id": data["id"],
                    "document": parsed_data,
                    "metadata": {
                        "chunk_id": data["chunk_id"],
                        "type": data["type"]
                    }
                }
                results.append(result)

        return results

    def load_ontology(self):
        """Load unified graph from FAISS and build OWL ontology."""
        # Clear existing graph
        self.graph = Graph()
        self.graph.bind("novel", self.ns)

        # Define OWL classes
        self.graph.add((self.ns.Personality, RDF.type, OWL.Class))
        self.graph.add((self.ns.Event, RDF.type, OWL.Class))
        self.graph.add((self.ns.Scene, RDF.type, OWL.Class))
        self.graph.add((self.ns.Action, RDF.type, OWL.ObjectProperty))

        # Retrieve graph data from FAISS
        results = self.get_all_graph_data()

        for result in results:
            data = result["document"]
            meta = result["metadata"]
            doc_id = result["id"]
            chunk_id = meta["chunk_id"]

            if meta["type"] != "relationship":
                # Node: Personality, Event, or Scene
                node_id = self.ns[doc_id]
                node_type = self.ns[data["type"].capitalize()]
                self.graph.add((node_id, RDF.type, node_type))
                self.graph.add((node_id, RDFS.label, self.graph.literal(data["label"])))
                self.graph.add((node_id, self.ns.chunk_id, self.graph.literal(chunk_id)))
                if "novel_id" in data:
                    self.graph.add((node_id, self.ns.novel_id, self.graph.literal(data["novel_id"])))
            else:
                # Relationship: Action
                source_id = self.ns[data["source"]]
                target_id = self.ns[data["target"]]
                self.graph.add((source_id, self.ns.Action, target_id))
                self.graph.add((source_id, self.ns.action_label, self.graph.literal(data["action"])))
                self.graph.add((source_id, self.ns.chunk_id, self.graph.literal(chunk_id)))
                if "novel_id" in data:
                    self.graph.add((source_id, self.ns.novel_id, self.graph.literal(data["novel_id"])))

    def save_ontology(self, file_path: str = "ontology.owl"):
        """Save the ontology to a file."""
        self.graph.serialize(file_path, format="xml")

    def get_graph(self):
        """Return the RDFLib graph."""
        return self.graph

    def query_ontology(self, sparql_query: str):
        """Execute a SPARQL query on the ontology."""
        return self.graph.query(sparql_query)

    def get_nodes_by_type(self, node_type: str) -> List[Dict]:
        """Get all nodes of a specific type from the ontology."""
        results = []
        graph_data = self.get_all_graph_data()

        for result in graph_data:
            data = result["document"]
            meta = result["metadata"]

            if meta["type"] == node_type.lower():
                results.append({
                    "id": result["id"],
                    "label": data.get("label", ""),
                    "chunk_id": meta["chunk_id"],
                    "type": data.get("type", "")
                })

        return results

    def get_relationships_for_node(self, node_id: str) -> List[Dict]:
        """Get all relationships involving a specific node."""
        results = []
        graph_data = self.get_all_graph_data()

        for result in graph_data:
            data = result["document"]
            meta = result["metadata"]

            if meta["type"] == "relationship":
                if data.get("source") == node_id or data.get("target") == node_id:
                    results.append({
                        "id": result["id"],
                        "source": data.get("source", ""),
                        "target": data.get("target", ""),
                        "action": data.get("action", ""),
                        "chunk_id": meta["chunk_id"]
                    })

        return results

    def search_similar_concepts(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar concepts using FAISS similarity search."""
        if self.index.ntotal == 0:
            return []

        # Create embedding for query
        query_embedding = self.embedding_model.encode([query_text])

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), min(top_k, self.index.ntotal))

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx in self.metadata:
                data = self.metadata[idx]
                if data.get("data_type") in ["node", "relationship"]:
                    parsed_data = json.loads(data["document"])
                    result = {
                        "id": data["id"],
                        "distance": float(distance),
                        "type": data["type"],
                        "data": parsed_data,
                        "chunk_id": data["chunk_id"]
                    }
                    results.append(result)

        return results

    def get_ontology_statistics(self) -> Dict:
        """Get statistics about the ontology."""
        graph_data = self.get_all_graph_data()

        stats = {
            "total_triples": len(self.graph),
            "total_nodes": 0,
            "total_relationships": 0,
            "node_types": {},
            "chunks": set()
        }

        for result in graph_data:
            meta = result["metadata"]
            data = result["document"]

            stats["chunks"].add(meta["chunk_id"])

            if meta["type"] != "relationship":
                stats["total_nodes"] += 1
                node_type = data.get("type", "unknown")
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            else:
                stats["total_relationships"] += 1

        stats["total_chunks"] = len(stats["chunks"])
        stats["chunks"] = list(stats["chunks"])

        return stats

    def export_graph_data(self, output_path: str = "graph_data.json"):
        """Export all graph data to a JSON file."""
        graph_data = self.get_all_graph_data()

        export_data = {
            "nodes": [],
            "relationships": []
        }

        for result in graph_data:
            data = result["document"]
            meta = result["metadata"]

            if meta["type"] != "relationship":
                export_data["nodes"].append({
                    "id": result["id"],
                    "type": data.get("type", ""),
                    "label": data.get("label", ""),
                    "chunk_id": meta["chunk_id"]
                })
            else:
                export_data["relationships"].append({
                    "id": result["id"],
                    "source": data.get("source", ""),
                    "target": data.get("target", ""),
                    "action": data.get("action", ""),
                    "chunk_id": meta["chunk_id"]
                })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Example usage
    loader = OntologyLoader()
    loader.load_ontology()
    loader.save_ontology("ontology.owl")
    print("Ontology loaded and saved.")

    # Print statistics
    stats = loader.get_ontology_statistics()
    print(f"Ontology statistics: {stats}")

    # Example: Get all personality nodes
    personalities = loader.get_nodes_by_type("personality")
    print(f"Found {len(personalities)} personalities")

    # Example: Search for similar concepts
    similar = loader.search_similar_concepts("meeting discussion")
    print(f"Similar concepts: {similar}")

    # Export graph data
    loader.export_graph_data("exported_graph.json")
    print("Graph data exported to JSON")
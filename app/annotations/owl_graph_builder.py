import json
import re
from groq import Groq
from sentence_transformers import SentenceTransformer
from app.utils.id_utils import IDUtils
from app.utils.logging_config import setup_logging
from app.db.db_utils import DatabaseUtils
from typing import Optional, List, Dict, Tuple

class OWLGraphBuilder:
    def __init__(self, db_path: str = "faiss_db", api_key: Optional[str] = None):
        """Initialize with DatabaseUtils, Groq client, and logger."""
        self.db_path = db_path
        self.db = DatabaseUtils(db_path=db_path)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.groq_client = Groq(api_key=api_key)
        self.logger = setup_logging()

    def extract_rdf_triples(self, text: str, chunk_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract RDF triples using Groq API."""
        self.logger.debug(f"Extracting RDF triples for chunk {chunk_id}")
        prompt = f"""
You are an expert in narrative analysis. Extract RDF triples from the following text, identifying nodes as personalities (characters), events, or scenes, and relationships as actions.
- Personalities are characters or entities (e.g., "John", "Alice").
- Events are key incidents or actions (e.g., "meeting started", "budget reviewed").
- Scenes are settings or contexts (e.g., "boardroom", "another city").
- Relationships are actions connecting nodes (e.g., "discussed", "prepared").
Each triple should include a reference to the chunk_id.
Return a JSON object with two lists: "nodes" and "relationships".
- Nodes: [{{"id": "node_id", "type": "personality|event|scene", "label": "name", "chunk_id": "chunk_id"}}]
- Relationships: [{{"source": "node_id", "target": "node_id", "action": "action_name", "chunk_id": "chunk_id"}}]

Text: {text}
Chunk ID: {chunk_id}

Example output:
{{
    "nodes": [
        {{"id": "n1", "type": "personality", "label": "John", "chunk_id": "chunk_id"}},
        {{"id": "n2", "type": "event", "label": "meeting started", "chunk_id": "chunk_id"}},
        {{"id": "n3", "type": "scene", "label": "boardroom", "chunk_id": "chunk_id"}}
    ],
    "relationships": [
        {{"source": "n1", "target": "n2", "action": "started", "chunk_id": "chunk_id"}}
    ]
}}
Return only the JSON object, without additional text or markdown.
"""

        try:
            response = self.groq_client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            result_text = response.choices[0].message.content

            # Extract JSON from response (handle markdown or extra text)
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if not json_match:
                self.logger.error(f"No JSON found in Groq response for chunk {chunk_id}: {result_text}")
                return [], []
            json_str = json_match.group(0)
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error for chunk {chunk_id}: {e}\nResponse: {result_text}")
            return [], []
        except Exception as e:
            self.logger.error(f"Error calling Groq API for chunk {chunk_id}: {e}")
            return [], []

        if not isinstance(result, dict):
            self.logger.error(f"Invalid Groq response for chunk {chunk_id}: {result}")
            return [], []

        nodes = []
        for node in result.get("nodes", []):
            node["id"] = IDUtils.generate_node_id(node["type"], node["label"], chunk_id)
            nodes.append(node)

        relationships = []
        for rel in result.get("relationships", []):
            rel["id"] = IDUtils.generate_relationship_id(rel["source"], rel["target"], rel["action"])
            relationships.append(rel)

        self.logger.debug(f"Extracted {len(nodes)} nodes and {len(relationships)} relationships")
        return nodes, relationships

    def store_graph(self, chunk_id: str, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Store graph nodes and relationships using DatabaseUtils."""
        self.logger.info(f"Storing graph for chunk {chunk_id}")
        nodes, relationships = self.extract_rdf_triples(text, chunk_id)

        for node in nodes:
            node_text = node["label"]
            embedding = self.embedding_model.encode([node_text])[0]
            metadata = {
                "chunk_id": chunk_id,
                "type": node["type"],
                "id": node["id"],
                "data_type": "graph_node",
                "novel_id": node.get("novel_id", "")
            }
            self.db.add_chunk(
                text=node_text,
                metadata=metadata,
                collection_name="graph",
                embedding=embedding
            )

        for rel in relationships:
            rel_text = f"{rel['source']} -> {rel['action']} -> {rel['target']}"
            embedding = self.embedding_model.encode([rel_text])[0]
            metadata = {
                "chunk_id": chunk_id,
                "id": rel["id"],
                "data_type": "graph_relationship",
                "novel_id": rel.get("novel_id", "")
            }
            self.db.add_chunk(
                text=rel_text,
                metadata=metadata,
                collection_name="graph",
                embedding=embedding
            )

        self.logger.info(f"Stored {len(nodes)} nodes and {len(relationships)} relationships for chunk {chunk_id}")
        return nodes, relationships

    def get_graph_by_chunk_id(self, chunk_id: str) -> Dict:
        """Retrieve graph data for a chunk_id using DatabaseUtils."""
        self.logger.debug(f"Retrieving graph for chunk {chunk_id}")
        results = self.db.query("", collection_name="graph")  # Query all graph data
        nodes, relationships = [], []

        for result in results["documents"][0]:
            metadata = results["metadatas"][0][results["documents"][0].index(result)]
            if metadata.get("chunk_id") == chunk_id:
                if metadata.get("data_type") == "graph_node":
                    nodes.append({
                        "id": metadata.get("id", ""),
                        "type": metadata.get("type", ""),
                        "label": result,
                        "chunk_id": chunk_id
                    })
                elif metadata.get("data_type") == "graph_relationship":
                    relationships.append({
                        "id": metadata.get("id", ""),
                        "label": result,
                        "chunk_id": chunk_id
                    })

        return {"nodes": nodes, "relationships": relationships}

    def get_unified_graph(self) -> Dict:
        """Retrieve all graph data using DatabaseUtils."""
        self.logger.info("Retrieving unified graph")
        results = self.db.query("", collection_name="graph")  # Query all graph data
        nodes, relationships = [], []

        for result in results["documents"][0]:
            metadata = results["metadatas"][0][results["documents"][0].index(result)]
            if metadata.get("data_type") == "graph_node":
                nodes.append({
                    "id": metadata.get("id", ""),
                    "type": metadata.get("type", ""),
                    "label": result,
                    "chunk_id": metadata.get("chunk_id", "")
                })
            elif metadata.get("data_type") == "graph_relationship":
                relationships.append({
                    "id": metadata.get("id", ""),
                    "label": result,
                    "chunk_id": metadata.get("chunk_id", "")
                })

        self.logger.debug(f"Unified graph: {len(nodes)} nodes, {len(relationships)} relationships")
        return {"nodes": nodes, "relationships": relationships}

    def search_similar_nodes(self, query_text: str, node_type: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """Search for similar nodes using DatabaseUtils."""
        results = self.db.query(query_text, n_results=top_k * 2, collection_name="graph")
        filtered_results = []

        for result in results["documents"][0]:
            metadata = results["metadatas"][0][results["documents"][0].index(result)]
            if metadata.get("data_type") == "graph_node" and (not node_type or metadata.get("type") == node_type):
                filtered_results.append({
                    "id": metadata.get("id", ""),
                    "distance": results["distances"][0][results["documents"][0].index(result)],
                    "data_type": metadata.get("data_type", ""),
                    "type": metadata.get("type", ""),
                    "label": result,
                    "chunk_id": metadata.get("chunk_id", "")
                })
            if len(filtered_results) >= top_k:
                break

        return filtered_results

    def delete_graph_by_chunk_id(self, chunk_id: str):
        """Delete graph data for a chunk_id using DatabaseUtils."""
        self.logger.debug(f"Deleting graph for chunk {chunk_id}")
        self.db.delete_by_id(chunk_id, collection_name="graph")
        self.logger.info(f"Deleted graph data for chunk {chunk_id}")

    def get_index_stats(self) -> Dict:
        """Return stats for the graph collection."""
        count = self.db.count_chunks(collection_name="graph")
        node_count = sum(1 for result in self.db.query("", collection_name="graph")["metadatas"][0] if result.get("data_type") == "graph_node")
        relationship_count = sum(1 for result in self.db.query("", collection_name="graph")["metadatas"][0] if result.get("data_type") == "graph_relationship")
        return {
            "total_vectors": count,
            "dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "nodes": node_count,
            "relationships": relationship_count
        }
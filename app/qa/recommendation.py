from app.embedding.vector_store import VectorStore
from app.semantic_graph.sparql_interface import SPARQLInterface
from app.semantic_graph.ontology_loader import OntologyLoader
from groq import Groq
from typing import List, Optional, Dict
import json


class RecommendationEngine:
    def __init__(self, db_path: str = "faiss_db", api_key: Optional[str] = None):
        """Initialize components for novel recommendation."""
        self.vector_store = VectorStore(db_path=db_path)
        self.ontology_loader = OntologyLoader(db_path=db_path)
        self.sparql_interface = SPARQLInterface(self.ontology_loader.get_graph())
        self.groq_client = Groq(api_key=api_key)

    def recommend_novels(self, preferences: str, n_results: int = 3) -> List[Dict]:
        """Recommend novels based on user preferences."""
        # Step 1: Vector search for relevant chunks based on preferences
        vector_results = self.vector_store.query_vectors(preferences, n_results=n_results * 2)
        novel_ids = list(set(result["metadata"]["novel_id"] for result in vector_results))

        # Step 2: SPARQL query to find relevant graph elements
        sparql_query = f"""
        PREFIX novel: <http://example.org/novel-ontology#>
        SELECT DISTINCT ?novel_id ?label ?type
        WHERE {{
            ?s novel:novel_id ?novel_id .
            ?s novel:label ?label .
            ?s a ?type .
            FILTER(?type IN (novel:Personality, novel:Event, novel:Scene))
            FILTER(contains(lcase(?label), "{preferences.lower()}"))
        }}
        """

        sparql_results = self.sparql_interface.execute_query(sparql_query)
        sparql_novel_ids = list(set(result["novel_id"] for result in sparql_results))

        # Combine novel IDs from vector and SPARQL results
        candidate_novel_ids = list(set(novel_ids + sparql_novel_ids))

        # Step 3: Use Groq to rank and describe recommendations
        prompt = f"""
        You are an expert in novel recommendations. Based on the following user preferences and candidate novels, recommend {n_results} novels.
        For each recommended novel, provide a brief description of why it matches the preferences.
        Return a JSON list of objects with keys: "novel_id", "reason".

        Preferences: {preferences}
        Candidate Novel IDs: {', '.join(candidate_novel_ids)}
        """

        response = self.groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )

        recommendations = response.choices[0].message.content

        # Handle case where recommendations might be a string (JSON) or already a list
        if isinstance(recommendations, str):
            try:
                recommendations = json.loads(recommendations)
            except json.JSONDecodeError:
                recommendations = []

        if not isinstance(recommendations, list):
            recommendations = []

        return recommendations[:n_results]

    def get_novel_details(self, novel_id: str) -> Dict:
        """Get detailed information about a specific novel."""
        # Query vector store for novel content
        vector_results = self.vector_store.query_vectors("", novel_id=novel_id, n_results=10)

        # Query SPARQL for novel metadata
        sparql_query = f"""
        PREFIX novel: <http://example.org/novel-ontology#>
        SELECT ?property ?value
        WHERE {{
            ?s novel:novel_id "{novel_id}" .
            ?s ?property ?value .
        }}
        """

        sparql_results = self.sparql_interface.execute_query(sparql_query)

        return {
            "novel_id": novel_id,
            "content_samples": [result["text"] for result in vector_results[:3]],
            "metadata": {result["property"]: result["value"] for result in sparql_results}
        }

    def get_similar_novels(self, novel_id: str, n_results: int = 5) -> List[str]:
        """Find novels similar to a given novel."""
        # Get sample content from the novel
        sample_results = self.vector_store.query_vectors("", novel_id=novel_id, n_results=3)

        if not sample_results:
            return []

        # Use the sample content to find similar novels
        sample_text = " ".join([result["text"] for result in sample_results])
        similar_results = self.vector_store.query_vectors(sample_text, n_results=n_results * 2)

        # Extract unique novel IDs, excluding the input novel
        similar_novel_ids = []
        for result in similar_results:
            result_novel_id = result["metadata"]["novel_id"]
            if result_novel_id != novel_id and result_novel_id not in similar_novel_ids:
                similar_novel_ids.append(result_novel_id)

        return similar_novel_ids[:n_results]


if __name__ == "__main__":
    # Example usage
    engine = RecommendationEngine(api_key="your groq api key")

    preferences = "novels with meetings and strong characters"
    recommendations = engine.recommend_novels(preferences)

    for rec in recommendations:
        print(f"Novel: {rec['novel_id']}, Reason: {rec['reason']}")

    # Example of getting novel details
    if recommendations:
        novel_id = recommendations[0]['novel_id']
        details = engine.get_novel_details(novel_id)
        print(f"\nDetails for {novel_id}:")
        print(f"Content samples: {details['content_samples']}")
        print(f"Metadata: {details['metadata']}")

        # Example of finding similar novels
        similar = engine.get_similar_novels(novel_id)
        print(f"\nSimilar novels to {novel_id}: {similar}")
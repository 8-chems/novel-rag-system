import groq
from groq import Groq
from app.llm.context_builder import ContextBuilder
from typing import List, Optional


class NLQToSPARQL:
    def __init__(self, db_path: str = "faiss_db", api_key: Optional[str] = None):
        """Initialize with Groq client and context builder."""
        self.groq_client = Groq(api_key=api_key)
        self.context_builder = ContextBuilder(db_path=db_path)

    def translate(self, nlq: str, chunk_ids: List[str], novel_id: Optional[str] = None) -> str:
        """Translate natural language query to SPARQL using Groq."""
        context = self.context_builder.build_context(chunk_ids, novel_id)
        prompt = f"""
        You are an expert in SPARQL query generation. Based on the provided context, translate the following natural language query into a SPARQL query.
        The context includes chunk text, keywords, and a knowledge graph with nodes (Personalities, Events, Scenes) and relationships (Actions).
        The ontology uses the namespace <http://example.org/novel-ontology#>.
        - Personalities, Events, and Scenes are classes.
        - Actions are object properties (e.g., novel:Action).
        - Labels are stored as novel:label.
        - Chunk IDs and novel IDs are stored as novel:chunk_id and novel:novel_id.

        Context:
        {context}

        Natural Language Query: {nlq}

        Return a valid SPARQL query as a string.
        Example output:
        PREFIX novel: <http://example.org/novel-ontology#>
        SELECT ?s ?label
        WHERE {{
            ?s a novel:Personality .
            ?s novel:label ?label .
        }}
        """
        try:
            result = self.groq_client.chat.completions.create(
                model="gemma2-9b-it",  # Updated model
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            return result if isinstance(result, str) else ""
        except groq.BadRequestError as e:
            print(f"API Error: {e}")
            return ""

if __name__ == "__main__":
    # Example usage
    translator = NLQToSPARQL(api_key="your groq api key")
    nlq = "Who started the meeting?"
    chunk_ids = ["example_chunk_id"]
    sparql_query = translator.translate(nlq, chunk_ids, novel_id="novel1")
    print(f"SPARQL Query: {sparql_query}")
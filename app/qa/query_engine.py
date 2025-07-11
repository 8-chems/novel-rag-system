from app.embedding.vector_store import VectorStore
from app.semantic_graph.nlq_to_sparql import NLQToSPARQL
from app.semantic_graph.sparql_interface import SPARQLInterface
from app.semantic_graph.ontology_loader import OntologyLoader
from app.llm.context_builder import ContextBuilder
from app.llm.groq_client import GroqClient  # Corrected import path
from typing import List, Optional, Dict
import logging


class QueryEngine:
    def __init__(self, db_path: str = "faiss_db", api_key: Optional[str] = None):
        """Initialize components for query processing."""
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        self.vector_store = VectorStore(db_path=db_path)
        self.nlq_to_sparql = NLQToSPARQL(db_path=db_path, api_key=api_key)
        self.ontology_loader = OntologyLoader(db_path=db_path)
        self.sparql_interface = SPARQLInterface(self.ontology_loader.get_graph())
        self.context_builder = ContextBuilder(db_path=db_path)
        self.groq_client = GroqClient(api_key=api_key)

    def process_query(self, query: str, novel_id: Optional[str] = None, n_results: int = 5) -> Dict:
        """Process a natural language query and return a combined answer."""
        # Step 1: Vector search to find relevant chunks
        self.logger.info(f"Processing query: {query}")
        vector_results = self.vector_store.query_vectors(query, novel_id=novel_id, n_results=n_results)
        self.logger.debug(f"Vector results: {len(vector_results)} chunks")

        chunk_ids = [result["metadata"]["chunk_id"] for result in vector_results]

        # Step 2: Generate and execute SPARQL query
        sparql_query = self.nlq_to_sparql.translate(query, chunk_ids, novel_id)
        sparql_results = []
        if sparql_query:
            sparql_results = self.sparql_interface.execute_query(sparql_query, novel_id=novel_id)

        # Step 3: Build context and generate LLM answer
        qa_prompt = self.context_builder.build_qa_prompt(query, chunk_ids, novel_id)

        print(f"generated prompt + context: \n {qa_prompt} \n ------------------------------------")
        llm_answer = self.groq_client.call(qa_prompt)

        self.logger.info(f"Query completed with LLM answer: {llm_answer[:50]}...")

        # Step 4: Combine results
        response = {
            "query": query,
            "vector_results": [
                {
                    "chunk_id": result["metadata"]["chunk_id"],
                    "text": result["text"],
                    "distance": result["distance"]
                }
                for result in vector_results
            ],
            "sparql_results": sparql_results,
            "llm_answer": llm_answer if isinstance(llm_answer, str) else "",
            "chunk_ids": chunk_ids
        }

        return response


if __name__ == "__main__":
    # Example usage
    engine = QueryEngine(api_key="your groq api key")
    query = "Who started the meeting in the novel?"
    response = engine.process_query(query, novel_id="novel1")

    print(f"Query: {response['query']}")
    print(f"Vector Results: {response['vector_results']}")
    print(f"SPARQL Results: {response['sparql_results']}")
    print(f"LLM Answer: {response['llm_answer']}")
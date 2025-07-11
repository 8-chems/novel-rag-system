import os
from app.db.create_db import create_database
from app.qa.query_engine import QueryEngine
from app.qa.recommendation import RecommendationEngine
from app.embedding.vector_store import VectorStore


class CLI:
    def __init__(self, db_path: str = "faiss_db", api_key: str = None):
        """Initialize components for the CLI."""
        self.db_path = db_path
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.query_engine = QueryEngine(db_path=db_path, api_key=self.api_key)
        self.recommendation_engine = RecommendationEngine(db_path=db_path, api_key=self.api_key)
        self.vector_store = VectorStore(db_path=db_path)

    def list_novels(self):
        """List all novel IDs in the database."""
        results = self.vector_store.query_vectors("", n_results=1000)  # Large n_results to get all
        novel_ids = sorted(set(result["metadata"]["novel_id"] for result in results))
        return novel_ids

    def run(self):
        """Run the CLI loop."""
        print("Welcome to the Novel RAG System CLI!")
        print("Commands: process <directory>, query <question> [novel_id], recommend <preferences>, list, exit")

        while True:
            command = input("Enter command: ").strip().lower()
            parts = command.split(maxsplit=2)
            action = parts[0] if parts else ""

            if action == "exit":
                print("Exiting CLI.")
                break
            elif action == "process" and len(parts) > 1:
                directory = parts[1]
                if os.path.isdir(directory):
                    print(f"Processing PDFs in {directory}...")
                    create_database(directory, db_path=self.db_path, groq_api_key=self.api_key)
                    print("Processing complete.")
                else:
                    print(f"Error: {directory} is not a valid directory.")
            elif action == "query" and len(parts) > 1:
                query = parts[1]
                novel_id = parts[2] if len(parts) > 2 else None
                print(f"Processing query: {query}")
                response = self.query_engine.process_query(query, novel_id=novel_id)
                print(f"Query: {response['query']}")
                print("Vector Results:")
                for result in response["vector_results"]:
                    print(
                        f"- Chunk ID: {result['chunk_id']}, Text: {result['text'][:50]}..., Distance: {result['distance']:.4f}")
                print("SPARQL Results:")
                for result in response["sparql_results"]:
                    print(f"- {result}")
                print(f"LLM Answer: {response['llm_answer']}")
            elif action == "recommend" and len(parts) > 1:
                preferences = parts[1]
                print(f"Recommending novels for: {preferences}")
                recommendations = self.recommendation_engine.recommend_novels(preferences)
                for rec in recommendations:
                    print(f"- Novel: {rec['novel_id']}, Reason: {rec['reason']}")
            elif action == "list":
                novel_ids = self.list_novels()
                print("Available Novels:")
                for nid in novel_ids:
                    print(f"- {nid}")
            else:
                print(
                    "Invalid command. Use: process <directory>, query <question> [novel_id], recommend <preferences>, list, exit")


if __name__ == "__main__":
    # Example usage
    cli = CLI(api_key="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")
    cli.run()
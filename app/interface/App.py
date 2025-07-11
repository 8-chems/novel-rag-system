import gradio as gr
from app.db.create_db import create_database
from app.qa.query_engine import QueryEngine
from app.qa.recommendation import RecommendationEngine
from app.embedding.vector_store import VectorStore
import os


class WebApp:
    def __init__(self, db_path: str = "faiss_db", api_key: str = None):
        """Initialize components for the web app."""
        self.db_path = db_path
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.query_engine = QueryEngine(db_path=db_path, api_key=self.api_key)
        self.recommendation_engine = RecommendationEngine(db_path=db_path, api_key=self.api_key)
        self.vector_store = VectorStore(db_path=db_path)

    def list_novels(self):
        """List all novel IDs in the database."""
        results = self.vector_store.query_vectors("", n_results=1000)
        return sorted(set(result["metadata"]["novel_id"] for result in results))

    def process_pdfs(self, directory: str):
        """Process PDFs in the given directory."""
        if not os.path.isdir(directory):
            return f"Error: {directory} is not a valid directory."
        create_database(directory, db_path=self.db_path, groq_api_key=self.api_key)
        return "PDFs processed successfully."

    def query_novel(self, query: str, novel_id: str):
        """Process a query with optional novel_id."""
        novel_id = novel_id if novel_id != "All" else None
        response = self.query_engine.process_query(query, novel_id=novel_id)
        output = f"**Query**: {response['query']}\n\n"
        output += "**Vector Results**:\n"
        for result in response["vector_results"]:
            output += f"- Chunk ID: {result['chunk_id']}, Text: {result['text'][:100]}..., Distance: {result['distance']:.4f}\n"
        output += "\n**SPARQL Results**:\n"
        for result in response["sparql_results"]:
            output += f"- {result}\n"
        output += f"\n**LLM Answer**: {response['llm_answer']}"
        return output

    def recommend_novels(self, preferences: str):
        """Recommend novels based on preferences."""
        recommendations = self.recommendation_engine.recommend_novels(preferences)
        output = "**Recommendations**:\n"
        for rec in recommendations:
            output += f"- Novel: {rec['novel_id']}, Reason: {rec['reason']}\n"
        return output

    def launch(self):
        """Launch the Gradio interface."""
        with gr.Blocks(title="Novel RAG System") as app:
            gr.Markdown("# Novel RAG System")

            with gr.Tab("Query"):
                query_input = gr.Textbox(label="Enter your question")
                novel_dropdown = gr.Dropdown(label="Select Novel", choices=["All"] + self.list_novels())
                query_button = gr.Button("Submit Query")
                query_output = gr.Markdown(label="Results")
                query_button.click(
                    fn=self.query_novel,
                    inputs=[query_input, novel_dropdown],
                    outputs=query_output
                )

            with gr.Tab("Recommend"):
                pref_input = gr.Textbox(label="Enter preferences (e.g., novels with meetings and strong characters)")
                rec_button = gr.Button("Get Recommendations")
                rec_output = gr.Markdown(label="Recommendations")
                rec_button.click(
                    fn=self.recommend_novels,
                    inputs=pref_input,
                    outputs=rec_output
                )

            with gr.Tab("Process PDFs"):
                dir_input = gr.Textbox(label="Enter directory path containing PDFs")
                process_button = gr.Button("Process PDFs")
                process_output = gr.Textbox(label="Status")
                process_button.click(
                    fn=self.process_pdfs,
                    inputs=dir_input,
                    outputs=process_output
                )

        app.launch()


if __name__ == "__main__":
    # Example usage
    app = WebApp(api_key="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")
    app.launch()
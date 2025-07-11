from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the SentenceTransformer model."""
        self.model = SentenceTransformer(model_name)

    def embed(self, text):
        """Convert a single text to an embedding vector."""
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts):
        """Convert a list of texts to embedding vectors."""
        return self.model.encode(texts, convert_to_numpy=True)


if __name__ == "__main__":
    # Example usage
    embedder = Embedder()
    sample_text = "John started the meeting in the boardroom."
    embedding = embedder.embed(sample_text)
    print(f"Embedding shape: {embedding.shape}")
    batch_embeddings = embedder.embed_batch([sample_text, "Alice prepared her presentation."])
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
import nltk
from groq import Groq  # ✅ Correct SDK
from app.utils.id_utils import IDUtils
from app.utils.logging_config import setup_logging

# Download required NLTK data
nltk.download('punkt')


class SemanticChunker:
    def __init__(self, api_key: str = None):
        """Initialize with Groq client and logger."""
        self.client = Groq(api_key=api_key)
        self.logger = setup_logging()

    def chunk(self, text: str, chunk_size: int = 500) -> list:
        """
        Chunk text into semantically coherent segments using Groq LLM.
        """
        self.logger.info("Starting semantic chunking")
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            self.logger.warning("No sentences found in text")
            return []

        # Prepare prompt for Groq
        prompt = f"""
You are an expert in narrative analysis. Analyze the following text and identify boundaries for scenes or parts of scenes.
A scene is a distinct narrative unit defined by a change in location, characters, or major action.
A part of a scene is a sub-unit within a scene, such as a shift in dialogue, action, or focus.
Return a JSON list of sentence indices (0-based) where a new scene or part of a scene begins.
If no clear boundaries are found, return an empty list.

Text:
{text}

Example output: [0, 3, 7]
"""

        try:
            response = self.client.chat.completions.create(
                model="gemma2-9b-it",  # or your chosen Groq-supported model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )

            raw_text = response.choices[0].message.content.strip()
            boundary_indices = eval(raw_text) if raw_text.startswith("[") else []

        except Exception as e:
            self.logger.error(f"Error calling Groq API: {e}")
            boundary_indices = []

        boundary_indices = sorted(set([0] + boundary_indices + [len(sentences)]))
        self.logger.debug(f"Boundary indices: {boundary_indices}")

        # Create chunks
        chunks = []
        order = 0
        current_word_count = 0
        current_chunk = []

        for i in range(len(sentences)):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            if i in boundary_indices or current_word_count + sentence_words > chunk_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "chunk_id": IDUtils.generate_chunk_id(),
                        "text": chunk_text,
                        "order": order
                    })
                    self.logger.debug(f"Created chunk {chunks[-1]['chunk_id']} with order {order}")
                    order += 1
                    current_chunk = []
                    current_word_count = 0
                if i in boundary_indices:
                    current_chunk.append(sentence)
                    current_word_count += sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "chunk_id": IDUtils.generate_chunk_id(),
                "text": chunk_text,
                "order": order
            })
            self.logger.debug(f"Created final chunk {chunks[-1]['chunk_id']} with order {order}")

        self.logger.info(f"Generated {len(chunks)} chunks")
        return chunks


if __name__ == "__main__":
    chunker = SemanticChunker(api_key="your groq api key")
    sample_text = """
    The meeting started in the boardroom. John discussed the project timeline.
    The team reviewed the budget. Meanwhile, in another city, Alice was preparing her presentation.
    She finalized her slides. The next day, the team reconvened to finalize the plan.
    """
    chunks = chunker.chunk(sample_text, chunk_size=50)
    for chunk in chunks:
        print(f"Chunk ID: {chunk['chunk_id']}\nText: {chunk['text'][:50]}...\nOrder: {chunk['order']}\n")

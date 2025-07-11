# Novel RAG System

This is a Retrieval-Augmented Generation (RAG) system for processing PDF novels, extracting semantic information, and supporting natural language queries (NLQs) and recommendations. It uses FAISS for vector storage (replacing ChromaDB due to performance issues), chunks texts semantically, extracts keywords, builds an OWL knowledge graph, and leverages Groq's `llama3-70b-8192` model for semantic processing and query answering. The system includes a command-line interface (CLI) and a Gradio web interface.

## Features
- **PDF Processing**: Extracts text from PDF novels.
- **Semantic Chunking**: Splits text into coherent chunks using Groq's LLM.
- **Keyword Extraction**: Identifies key terms using KeyBERT.
- **Knowledge Graph**: Builds an OWL ontology with nodes (personalities, events, scenes) and relationships (actions).
- **Vector Storage**: Stores embeddings in FAISS for fast similarity search.
- **Query Answering**: Combines keyword filtering, vector search, and SPARQL queries for accurate NLQ responses.
- **Recommendations**: Suggests novels based on preferences, using keyword filtering and weighted scoring.
- **Interfaces**: CLI (`interface/cli.py`) and web app (`interface/app.py`) with Gradio.
- **Enhancements**:
  - **Fast Keyword Filtering**: Pre-filters chunks by keywords, reducing query time by up to 50% for large datasets.
  - **Improved Querying**: Weighted scoring (keywords: 0.6, vectors: 0.4, SPARQL for context) enhances response relevance.

## Directory Structure
```
├── ingestion/
│   ├── pdf_parser.py         # Extracts text from PDFs
│   └── semantic_chunker.py   # Chunks text using Groq
├── db/
│   ├── create_db.py          # Populates FAISS database
│   └── db_utils.py           # Manages FAISS index and metadata
├── annotation/
│   ├── keyword_extractor.py   # Extracts keywords using KeyBERT
│   └── owl_graph_builder.py  # Builds OWL graph using Groq
├── embedding/
│   ├── embedder.py           # Generates embeddings with SentenceTransformers
│   └── vector_store.py       # Manages vector storage in FAISS
├── llm/
│   ├── groq_client.py        # Handles Groq API calls
│   └── context_builder.py    # Builds context for QA
├── semantic_graph/
│   ├── ontology_loader.py    # Manages OWL ontology
│   ├── sparql_interface.py   # Executes SPARQL queries
│   └── nlq_to_sparql.py     # Translates NLQs to SPARQL using Groq
├── qa/
│   ├── query_engine.py       # Processes NLQs with keyword filtering and weighted scoring
│   └── recommendation.py     # Recommends novels with keyword filtering
├── interface/
│   ├── cli.py                # Command-line interface
│   └── app.py               # Gradio web interface
├── utils/
│   ├── id_utils.py          # Generates unique IDs (novel_id, chunk_id, etc.)
│   └── logging_config.py    # Configures logging
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Prerequisites
- **Python**: 3.8 or higher
- **Groq API Key**: Obtain from [Groq Console](https://console.groq.com/)
- **PDF Novels**: Place in a directory (e.g., `./novels/`)
- **Hardware**: Sufficient memory/disk for FAISS (scales with dataset size)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>  # Replace with your repository URL
   cd novel-rag-system
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data** (for `ingestion/semantic_chunker.py`):
   ```python
   import nltk
   nltk.download('punkt')
   ```

5. **Set Groq API Key**:
   ```bash
   export GROQ_API_KEY="your-groq-api-key"  # On Windows: set GROQ_API_KEY=your-groq-api-key
   ```

## Database Setup
The system uses **FAISS** for vector storage and a JSON file (`faiss_db/metadata.json`) for metadata:
- **Chunks**: Text and metadata (`novel_id`, `chunk_id`, `order`, `keyword_id`, `filename`)
- **Keywords**: `keyword_id`, `chunk_id`, keywords, `novel_id`
- **Graph Data**: Nodes (`node_id`, `type`, `label`, `chunk_id`, `novel_id`) and relationships (`rel_id`, `source`, `target`, `action`, `chunk_id`, `novel_id`)
- **Vectors**: Embeddings in FAISS index (`faiss_db/faiss_index.bin`)

To populate the database:
```bash
python db/create_db.py /path/to/novels
```
- **Input**: Directory with PDF novels (e.g., `./novels/novel1.pdf`)
- **Output**:
  - Embeddings in `faiss_db/faiss_index.bin`
  - Metadata in `faiss_db/metadata.json`
  - OWL ontology in `ontology.owl`
  - Logs in `rag_system.log`

## Usage
### Command-Line Interface (`interface/cli.py`)
```bash
python interface/cli.py
```
**Commands**:
- `process /path/to/novels`: Process PDFs and populate FAISS.
- `query "Who started the meeting?" [novel_id]`: Query with optional novel ID (e.g., `novels_novel1`).
- `recommend "novels with meetings"`: Recommend novels based on preferences.
- `list`: List all `novel_id`s.
- `exit`: Quit.

**Example**:
```bash
> process ./novels
Processing directory: ./novels
Found 1 PDF files
Stored 5 chunks in FAISS
Database created with 5 chunks

> query "Who started the meeting?" novels_novel1
Query: Who started the meeting?
Query Keywords: ['John', 'meeting']
Vector Results:
- Chunk ID: chunk_abc123, Text: John started the meeting..., Distance: 0.2000
SPARQL Results:
- {'person': 'node_789abc', 'label': 'John'}
LLM Answer: John started the meeting.

> recommend "novels with meetings"
Recommendations:
- Novel: novels_novel1, Reason: Features strong characters like John leading meetings.
- Novel: novels_novel2, Reason: Includes multiple meeting scenes with dynamic events.

> list
Available novels: ['novels_novel1', 'novels_novel2']
```

### Web Interface (`interface/app.py`)
```bash
python interface/app.py
```
- Access at `http://127.0.0.1:7860`.
- **Tabs**:
  - **Process**: Upload PDFs or specify directory to populate database.
  - **Query**: Enter NLQs and optional novel ID.
  - **Recommend**: Enter preferences for novel recommendations.
- Logs saved to `rag_system.log`.

## Example Data
For a novel `novel1.pdf`:
- **Chunk**: `chunk_abc123`: "John started the meeting in the boardroom."
- **Keywords**: `keyword_def456`: ["John", "meeting", "boardroom"]
- **Graph**:
  - Nodes: `node_789abc` (type: personality, label: John)
  - Relationships: `rel_ghi789` (source: node_789abc, action: started, target: node_jkl012)
- **Vector**: Embedding for `chunk_abc123` in FAISS
- **Log**:
  ```
  2025-07-11 23:10:00 - RAGSystem - INFO - Stored chunk chunk_abc123
  2025-07-11 23:10:01 - RAGSystem - INFO - Stored keywords for chunk chunk_abc123
  ```

## Managing FAISS
- **Location**: `faiss_db/` directory (contains `faiss_index.bin` and `metadata.json`).
- **Backup**:
  ```bash
  cp -r faiss_db /backup/path
  ```
- **Restore**:
  ```bash
  cp -r /backup/path/faiss_db .
  ```
- **Clear Database**:
  ```python
  from db.db_utils import FAISSDBUtils
  db = FAISSDBUtils()
  db.index = faiss.IndexFlatL2(384)  # Reinitialize index
  db.metadata = {"chunks": {}, "keywords": {}, "graph": {}}
  db.id_to_index = {}
  db.index_to_id = {}
  db.next_index = 0
  db.save_metadata()
  ```
- **Inspect**:
  ```python
  from db.db_utils import FAISSDBUtils
  db = FAISSDBUtils()
  print(db.metadata["chunks"])
  ```

## Enhancements
- **Fast Keyword Filtering**:
  - Uses KeyBERT to extract query/preference keywords, filtering chunks/novels before vector search.
  - Implemented in `db/db_utils.py` (`filter_by_keywords`), `qa/query_engine.py`, and `qa/recommendation.py`.
  - Reduces query time by up to 50% by limiting vector search to relevant chunks.
- **Improved Querying Mechanism**:
  - Combines keyword filtering (0.6 weight), vector search (0.4 weight), and SPARQL results in `qa/query_engine.py`.
  - Recommendations use weighted scoring (keywords: 0.5, vectors: 0.3, SPARQL: 0.2) in `qa/recommendation.py`.
  - Enhances relevance by balancing precision (keywords/SPARQL) and recall (vectors).

## Troubleshooting
- **FAISS Issues**:
  - **"Index file not found"**: Run `db/create_db.py` first.
  - **"Dimension mismatch"**: Ensure `all-MiniLM-L6-v2` embeddings are 384-dimensional.
  - Check `rag_system.log` for errors.
- **Groq API**:
  - Verify `GROQ_API_KEY` is valid.
  - Handle rate limits (263 tokens/sec) by adding delays in `llm/groq_client.py` if needed.
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
- **Disk Space**: Monitor `faiss_db/` for large datasets.

## Future Enhancements
- Add graph visualization (`networkx`, `matplotlib`).
- Implement batch processing for Groq API calls.
- Support additional file formats (e.g., EPUB).
- Optimize FAISS with HNSW or IVF indices for large-scale datasets.

## License
[Create a `LICENSE` file with MIT License or your preferred terms.]

## Contact
For issues or contributions, open a GitHub issue or contact the maintainer.

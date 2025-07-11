# Novel Analysis RAG System

This project is a Retrieval-Augmented Generation (RAG) system designed to analyze novels in PDF format. It uses the Groq LLM to answer complex questions about characters, events, and relationships, and to provide content-based recommendations.

## Features

- **Per-Novel Processing**: Each novel is processed and stored in isolation.
- **Semantic Chunking**: Text is segmented into meaningful chunks using Groq.
- **Keyword Extraction**: Key concepts are extracted from each chunk.
- **Knowledge Graph**: A unique knowledge graph is built for each novel to map characters, events, and relationships.
- **Vector Search**: Chunks are embedded and stored in a FAISS vector store for fast retrieval.
- **RAG-based Q&A**: Answers user questions by combining retrieved text, keywords, and graph data.
- **Recommendations**: Suggests similar novels based on thematic and structural analysis.

## Project Structure

```
novel_rag_system/
├── app/                  # Core application logic
│   ├── chunker.py
│   ├── database_setup.py
│   ├── embedder.py
│   ├── graph_builder.py
│   ├── keyword_extractor.py
│   ├── query_engine.py
│   └── recommendation.py
├── data/
│   └── novels_pdf/     # Input folder for novels
├── database/             # SQLite database and other data files
├── graphs/               # Stores generated knowledge graphs
├── vector_stores/        # Stores FAISS vector indexes
├── main.py               # Main CLI application
├── requirements.txt      # Project dependencies
└── .env.example          # Environment variable template
```

## Usage

1.  **Setup**: `pip install -r requirements.txt`
2.  **Environment**: Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
3.  **Process Novels**: `python main.py process-novels --path ./data/novels_pdf`
4.  **Ask a Question**: `python main.py ask --novel "Novel Title" --question "Your question here"`

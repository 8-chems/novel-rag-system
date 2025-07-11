import os
import json
from app.db.db_utils import DatabaseUtils
from app.ingestion.pdf_parser import parse_pdfs_in_directory
from app.ingestion.semantic_chunker import SemanticChunker
from app.annotations.keyword_extractor import KeywordExtractor
from app.annotations.owl_graph_builder import OWLGraphBuilder
from app.embedding.embedder import Embedder
from app.semantic_graph.ontology_loader import OntologyLoader
from app.utils.id_utils import IDUtils
from app.utils.logging_config import setup_logging

def create_database(input_dir: str, db_path: str = "faiss_db", groq_api_key: str = None):
    """Create a FAISS database from PDF files with chunks, keywords, graph, and ontology."""
    logger = setup_logging()
    logger.info(f"Starting database creation for directory: {input_dir}")

    db = DatabaseUtils(db_path=db_path)
    keyword_extractor = KeywordExtractor(db_path=db_path)
    owl_builder = OWLGraphBuilder(db_path=db_path, api_key=groq_api_key)
    embedder = Embedder()
    chunker = SemanticChunker(api_key=groq_api_key)
    ontology_loader = OntologyLoader(db_path=db_path)

    pdf_texts = parse_pdfs_in_directory(input_dir)
    logger.info(f"Found {len(pdf_texts)} PDF files")

    for filename, text in pdf_texts:
        novel_id = IDUtils.generate_novel_id(filename, input_dir)
        logger.debug(f"Processing novel: {novel_id}")

        chunks = chunker.chunk(text, chunk_size=500)
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]

            # Generate embedding
            embedding = embedder.embed(chunk_text)

            # Store chunk
            metadata = {
                "filename": filename,
                "novel_id": novel_id,
                "order": chunk["order"],
                "data_type": "chunk"
            }
            db.add_chunk(chunk_text, metadata, collection_name="chunks", embedding=embedding)

            # Store keywords
            keyword_id, keywords = keyword_extractor.process_and_store_keywords(
                chunk_id, chunk_text, {"filename": filename, "novel_id": novel_id}
            )
            if keywords:
                keywords_text = ", ".join(keywords)
                keyword_embedding = embedder.embed(keywords_text)
                db.add_chunk(
                    keywords_text,
                    {"chunk_id": chunk_id, "novel_id": novel_id, "data_type": "keywords"},
                    collection_name="keywords",
                    embedding=keyword_embedding
                )

            # Store graph data
            nodes, relationships = owl_builder.store_graph(chunk_id, chunk_text)
            for node in nodes:
                node_text = node["label"]
                node_embedding = embedder.embed(node_text)
                db.add_chunk(
                    node_text,
                    {"chunk_id": chunk_id, "type": node["type"], "id": node["id"], "data_type": "graph_node"},
                    collection_name="graph",
                    embedding=node_embedding
                )
            for rel in relationships:
                rel_text = f"{rel['source']} -> {rel['action']} -> {rel['target']}"
                rel_embedding = embedder.embed(rel_text)
                db.add_chunk(
                    rel_text,
                    {"chunk_id": chunk_id, "data_type": "graph_relationship", "id": rel["id"]},
                    collection_name="graph",
                    embedding=rel_embedding
                )

    ontology_loader.load_ontology()
    ontology_loader.save_ontology("ontology.owl")
    logger.info("Ontology loaded and saved")
    logger.info(f"Database created at {db_path} with {db.count_chunks('chunks')} chunks")
    return db.count_chunks("chunks")
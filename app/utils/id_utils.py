import uuid
import os
import hashlib
from typing import Optional


class IDUtils:
    @staticmethod
    def generate_chunk_id() -> str:
        """Generate a unique chunk ID."""
        return f"chunk_{str(uuid.uuid4())}"

    @staticmethod
    def generate_keyword_id() -> str:
        """Generate a unique keyword ID."""
        return f"keyword_{str(uuid.uuid4())}"

    @staticmethod
    def generate_node_id(type: str, label: str, chunk_id: str) -> str:
        """Generate a unique node ID for graph nodes."""
        # Use hash to ensure consistent IDs for same type/label/chunk
        hash_input = f"{type}_{label}_{chunk_id}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"node_{hash_id}"

    @staticmethod
    def generate_relationship_id(source: str, target: str, action: str) -> str:
        """Generate a unique relationship ID for graph relationships."""
        hash_input = f"{source}_{target}_{action}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"rel_{hash_id}"

    @staticmethod
    def generate_novel_id(filename: str, directory: Optional[str] = None) -> str:
        """Generate a novel ID from filename and optional directory."""
        base_name = os.path.splitext(os.path.basename(filename))[0]
        if directory:
            dir_name = os.path.basename(os.path.normpath(directory))
            return f"{dir_name}_{base_name}"
        return base_name


if __name__ == "__main__":
    # Example usage
    print(f"Chunk ID: {IDUtils.generate_chunk_id()}")
    print(f"Keyword ID: {IDUtils.generate_keyword_id()}")
    print(f"Node ID: {IDUtils.generate_node_id('personality', 'John', 'chunk_123')}")
    print(f"Relationship ID: {IDUtils.generate_relationship_id('node_1', 'node_2', 'started')}")
    print(f"Novel ID: {IDUtils.generate_novel_id('novel1.pdf', './novels')}")
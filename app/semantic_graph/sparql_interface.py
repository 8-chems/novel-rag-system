from rdflib import Graph
from typing import List, Dict, Optional
import re


class SPARQLInterface:
    def __init__(self, graph: Graph):
        """Initialize with an RDFLib graph."""
        self.graph = graph

    def _extract_sparql_query(self, llm_response: str) -> str:
        """Extract a SPARQL query from the LLM response, handling code blocks or plain text."""
        # Try to extract query from triple backtick code block (```sparql ... ``` or ``` ... ```)
        code_block_pattern = r"```(?:sparql)?\s*\n([\s\S]*?)\n```"
        match = re.search(code_block_pattern, llm_response, re.MULTILINE)
        if match:
            query = match.group(1).strip()
        else:
            # If no code block, assume the response is the query itself, but remove any leading/trailing text
            query = llm_response.strip()
            # Basic validation: check if query starts with PREFIX or SELECT
            if not (query.startswith("PREFIX") or query.startswith("SELECT")):
                raise ValueError("Invalid SPARQL query: Does not start with PREFIX or SELECT")

        # Validate query contains a WHERE clause
        if "WHERE" not in query.upper():
            raise ValueError("Invalid SPARQL query: Missing WHERE clause")

        return query

    def execute_query(self, llm_response: str, novel_id: Optional[str] = None, chunk_id: Optional[str] = None) -> List[Dict]:
        """Execute a SPARQL query extracted from an LLM response with optional novel_id or chunk_id filters."""
        ns = "http://example.org/novel-ontology#"

        # Extract the SPARQL query from the LLM response
        try:
            query = self._extract_sparql_query(llm_response)
        except ValueError as e:
            print(f"Query extraction error: {e}")
            return []

        # Add filters to the query if provided
        if novel_id or chunk_id:
            where_clause = ""
            if novel_id:
                where_clause += f"?s <{ns}novel_id> \"{novel_id}\" . "
            if chunk_id:
                where_clause += f"?s <{ns}chunk_id> \"{chunk_id}\" . "
            if where_clause:
                # Insert filter into the WHERE clause, ensuring it integrates correctly
                where_start = query.upper().find("WHERE {")
                if where_start == -1:
                    print("Query error: No WHERE clause found for adding filters")
                    return []
                where_end = query.rfind("}")
                if where_end == -1:
                    print("Query error: Malformed WHERE clause")
                    return []
                query = query[:where_start + 7] + " " + where_clause + query[where_start + 7:where_end] + "}"

        print("Executing SPARQL Query:")
        print(query)

        # Execute the query with error handling
        try:
            results = self.graph.query(query)
            return [
                {str(var): str(val) for var, val in result.asdict().items()}
                for result in results
            ]
        except Exception as e:
            print(f"SPARQL execution error: {e}")
            return []


if __name__ == "__main__":
    # Example usage with ontology loader and sample LLM response
    from ontology_loader import OntologyLoader

    loader = OntologyLoader()
    loader.load_ontology()
    graph = loader.get_graph()
    interface = SPARQLInterface(graph)

    # Simulated LLM response with a SPARQL query
    llm_response = """
    Here is the SPARQL query for your request:

    ```sparql
    PREFIX novel: <http://example.org/novel-ontology#>
    SELECT ?s ?label
    WHERE {
        ?s a novel:Personality .
        ?s novel:label ?label .
    }
    ```

    This query retrieves all personalities and their labels.
    """

    results = interface.execute_query(llm_response, novel_id="novel1")
    for result in results:
        print(f"Result: {result}")
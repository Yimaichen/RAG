"""Read chunk tool - retrieve full document content from Document Store (SQLite) with Metadata."""

import sqlite3
import json
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


class ReadChunkTool(BaseTool):
    """Read full content of document chunks from SQLite Document Store."""
    
    def __init__(self, db_path: str = "/root/autodl-tmp/arag_data/arag_docs.db"):
        self.db_path = db_path
        
        if not HAS_TIKTOKEN:
            raise ImportError("tiktoken required. Please install: pip install tiktoken")
            
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        # Initial database validation
        self._verify_db()

    def _verify_db(self):
        """Verify SQLite Document Store is working properly"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT count(*) FROM chunks")
                count = cursor.fetchone()[0]
                print(f"📚 ReadChunkTool connected. Document Store contains {count} chunks.")
        except Exception as e:
            print(f"⚠️ Warning: Document Store issue at {self.db_path}: {e}")

    def _fetch_chunk_data(self, chunk_id: str) -> Tuple[str, str]:
        """Extract raw text and metadata from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT text, metadata FROM chunks WHERE id = ?", (chunk_id,))
            result = cursor.fetchone()
            return (result[0], result[1]) if result else (None, None)

    @property
    def name(self) -> str:
        return "read_chunk"
        
    def get_schema(self) -> Dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": "read_chunk",
                    "description": """Read the complete content of document chunks by their IDs.
    
    This tool returns the full text of the specified chunks, allowing you to examine the complete context and details that are not visible in search snippets.
    
    IMPORTANT: Search results (keyword_search and semantic_search) only show abbreviated snippets marked with "..." - they are NOT sufficient for answering questions. You MUST use read_chunk to get the full content before formulating your answer.
    
    STRATEGY:
    - Always read promising chunks identified by your searches
    - Make sure to read the most relevant chunks to gather complete information
    - If information seems incomplete or truncated, read adjacent chunks (± 1)
    - Reading full text is essential for accurate answers
    
    Note: Previously read chunks will be marked as already seen to avoid redundant information.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of chunk IDs to retrieve (e.g., ['0', '24', '172'])"
                            }
                        },
                        "required": ["chunk_ids"]
                    }
                }
            }

    def execute(self, context: 'AgentContext', chunk_ids: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        if not chunk_ids:
            return "Error: No chunk_ids provided.", {"retrieved_tokens": 0}
                
        chunk_ids = [str(cid) for cid in chunk_ids]
        result_parts = []
        new_chunks_read = []
        already_read = []
        total_tokens = 0
        
        for cid in chunk_ids:
            # Idempotency check: avoid Agent re-reading chunks in the same loop to save tokens
            if context.is_chunk_read(cid):
                already_read.append(cid)
                result_parts.append(f"\n[Chunk {cid}] (Already read previously, skipping full output to save space.)")
                continue
                
            # Fetch data from database
            content, metadata_str = self._fetch_chunk_data(cid)
            
            if content:
                # Dynamically restore provenance info (Source + Headers)
                location = "Unknown Source"
                if metadata_str:
                    try:
                        meta = json.loads(metadata_str)
                        source = meta.get("source_document", "Unknown Document")
                        headers = " > ".join([v for k, v in meta.items() if k.startswith("Header")])
                        location = f"{source} | {headers}" if headers else source
                    except:
                        pass

                # Format output for LLM to distinguish different sections
                separator = "=" * 60
                header_line = f"[{cid}] Source: {location}"
                result_parts.append(f"\n{separator}\n{header_line}\n{'-' * len(header_line)}\n{content}\n{separator}")
                
                # Token counting and context state update
                chunk_tokens = len(self.tokenizer.encode(content))
                total_tokens += chunk_tokens
                context.mark_chunk_as_read(cid)
                new_chunks_read.append(cid)
            else:
                result_parts.append(f"\n[Chunk {cid}] - Error: Not found in database.")
                
        tool_result = "\n".join(result_parts)
        
        # Log for subsequent performance analysis
        context.add_retrieval_log(
            tool_name="read_chunk",
            tokens=total_tokens,
            metadata={
                "requested": chunk_ids,
                "new": new_chunks_read,
                "duplicated": already_read
            }
        )
        
        # Must return tuple: (text for LLM, stats dict for system)
        return tool_result, {
            "retrieved_tokens": total_tokens,
            "new_chunks_count": len(new_chunks_read)
        }
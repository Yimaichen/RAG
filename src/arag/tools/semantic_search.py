"""Semantic search tool - Milvus vector retrieval and Cross-Encoder reranking."""

import json
import threading
import sqlite3 
from typing import Dict, List, Any, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from pymilvus import connections, Collection
    HAS_MILVUS = True
except ImportError:
    HAS_MILVUS = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SemanticSearchTool(BaseTool):
    """Semantic search using Zilliz Cloud (Milvus), BGE-Reranker, and SQLite Document Store."""
    
    _embedding_lock = threading.Lock()
    
    def __init__(
        self,
        db_path: str = "/root/autodl-tmp/arag_data/arag_docs.db",
        collection_name: str = "arag_sentence_index",
        embedding_model: str = "/root/autodl-tmp/models/bge-large-zh-v1.5",
        reranker_model: str = "/root/autodl-tmp/models/bge-reranker-v2-m3",
        # Core modification: completely replace host/port with elegant URI and Token parameters
        milvus_uri: str = "https://in03-68d51e54ccbb886.serverless.aws-eu-central-1.cloud.zilliz.com",
        milvus_token: str = "7180ab611ea70565b838bee7096faaab225fac9bf71f69e46f8c188197d32d127c952e7504c7bed7ee489ad4e3c4564a25e63bd2",
        device: str = "cuda"
    ):
        if not HAS_SENTENCE_TRANSFORMERS or not HAS_MILVUS or not HAS_TIKTOKEN:
            raise ImportError("Missing dependencies. Install: pymilvus sentence-transformers tiktoken")
            
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        # Initialize SQLite Document Store
        print(f"🗄️ Initializing SQLite Document Store from {db_path}...")
        self.db_path = db_path
        
        # Connect to remote Zilliz Cloud (Milvus) vector cluster
        print(f"🔗 SemanticSearchTool connecting to Zilliz Cloud at {milvus_uri}...")
        connections.connect(
            "default", 
            uri=milvus_uri,
            token=milvus_token
        )
        self.collection = Collection(collection_name)
        self.collection.load() # Ensure remote index is loaded into memory for immediate querying
        
        # Load deep learning models (leveraging AutoDL's powerful GPU)
        print(f"🧠 Loading Embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model, device=device)
        
        print(f"⚖️ Loading Reranker model: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, device=device)
        
    @property
    def name(self) -> str:
        return "semantic_search"
        
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": """Semantic search using embedding similarity. Matches your query against sentences in each chunk via vector similarity.
WHEN TO USE:
- When keyword search fails to find relevant information
- When exact wording in documents is unknown
- For conceptual/meaning-based matching
RETURNS: Abbreviated snippets with matched sentences. Use read_chunk to get full text for answering.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, context: 'AgentContext', query: str, top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)
        top_n_recall = top_k * 5 # Expand recall pool for Reranker
        
        # ==========================================
        # Stage 1: Query vectorization
        # ==========================================
        with self._embedding_lock:
            # Use normalization to align with Milvus IP (inner product) distance, equivalent to cosine similarity
            query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
            
        # ==========================================
        # Stage 2: Milvus fast ANN recall (network request to Zilliz Cloud)
        # ==========================================
        # Using ef: 64, HNSW index-specific parameter to prevent local optima
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        
        results = self.collection.search(
            data=[query_emb.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_n_recall,
            output_fields=["chunk_id", "sentence_text"]
        )
        
        if not results or len(results[0]) == 0:
            return f"No results for: {query}", {"retrieved_tokens": 0, "chunks_found": 0}
            
        hits = results[0]
        
        # ==========================================
        # Stage 3: BGE-Reranker fine-grained scoring
        # ==========================================
        cross_pairs = [[query, hit.entity.get("sentence_text")] for hit in hits]
        rerank_scores = self.reranker.predict(cross_pairs)
        
        # ==========================================
        # Stage 4: Max-P Chunk aggregation
        # ==========================================
        chunk_sentences = {}
        for i, hit in enumerate(hits):
            chunk_id = hit.entity.get("chunk_id")
            sentence = hit.entity.get("sentence_text")
            score = float(rerank_scores[i]) 
            
            if chunk_id not in chunk_sentences:
                chunk_sentences[chunk_id] = []
            chunk_sentences[chunk_id].append({
                'sentence': sentence,
                'similarity': score
            })
            
        chunk_scores = []
        for chunk_id, sents in chunk_sentences.items():
            max_similarity = max(s['similarity'] for s in sents)
            chunk_scores.append((chunk_id, max_similarity, sents))
            
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_scores[:top_k]
        
        # ==========================================
        # Stage 5: Text restoration and display (SQLite fast metadata provenance query)
        # ==========================================
        result_parts = []
        all_matched = []
        
        for chunk_id, max_sim, sents in top_chunks:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT text, metadata FROM chunks WHERE id = ?", (chunk_id,))
                result = cursor.fetchone()
                
            if not result:
                continue
                
            chunk_text = result[0]
            metadata_str = result[1] 
            
            # Parse metadata to extract source and sections
            meta = json.loads(metadata_str)
            source = meta.get("source_document", "Unknown Document")
            headers = " > ".join([v for k, v in meta.items() if k.startswith("Header")])
            location = f"{source} | {headers}" if headers else source
                
            sents_sorted = sorted(sents, key=lambda x: chunk_text.find(x['sentence']))
            matched_text = "... " + " ... ".join([s['sentence'] for s in sents_sorted]) + " ..."
            
            result_parts.append(f"Source: [{location}] (Chunk: {chunk_id}, Rel: {max_sim:.2f})\nMatched: {matched_text}")
            
            all_matched.extend([s['sentence'] for s in sents_sorted])
            
        tool_result = "\n\n".join(result_parts)
        retrieved_tokens = len(self.tokenizer.encode("\n".join(all_matched))) if all_matched else 0
        
        context.add_retrieval_log(
            tool_name="semantic_search",
            tokens=retrieved_tokens,
            metadata={"query": query, "chunks_found": len(top_chunks)}
        )
        
        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(top_chunks)}
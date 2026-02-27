"""Keyword search tool - BM25 recall with Cross-Encoder Reranking and Metadata Citation."""

import json
import sqlite3
import numpy as np
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from arag.tools.base import BaseTool

if TYPE_CHECKING:
    from arag.core.context import AgentContext

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


class KeywordSearchTool(BaseTool):
    """Keyword search using BM25 and BGE-Reranker."""
    
    def __init__(
        self, 
        chunks_file: str = "/root/autodl-tmp/arag_data/chunks.json",
        db_path: str = "/root/autodl-tmp/arag_data/arag_docs.db",
        reranker_model: str = "/root/autodl-tmp/models/bge-reranker-v2-m3", 
        device: str = "cuda"
    ):
        if not HAS_TIKTOKEN or not HAS_BM25 or not HAS_CROSS_ENCODER:
            raise ImportError("Missing dependencies. Install: tiktoken rank_bm25 sentence-transformers")
            
        self.chunks_file = chunks_file
        self.db_path = db_path
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        
        # Load data (local variables, destroyed after use)
        print("📦 Loading chunks and pre-split sentences for BM25...")
        chunks = self._load_chunks() 
        
        # Flatten sentences and build BM25 corpus
        self.sentences = []
        self.sentence_to_chunk = []
        self.sentence_orig_indices = [] # Record original sentence positions in chunk for coherent splicing later
        tokenized_corpus = []
        
        for chunk in chunks:
            chunk_id = chunk['id']
            # Use bulletproof sentences pre-split in data_ingestion.py
            chunk_sentences = chunk.get('sentences', [])
            for idx, sent in enumerate(chunk_sentences):
                self.sentences.append(sent)
                self.sentence_to_chunk.append(chunk_id)
                self.sentence_orig_indices.append(idx)
                # Basic tokenization for BM25
                tokenized_corpus.append(sent.lower().split())
                
        # Free memory: destroy chunks object containing large texts after sentence extraction
        del chunks
                
        # Initialize BM25 inverted index engine
        print("⚡ Building BM25 index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize BGE-Reranker model (direct disk read)
        print(f"🧠 Loading Reranker model from local path: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model, device=device)

    def _load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data and isinstance(data[0], dict):
            return data
        return []

    @property
    def name(self) -> str:
        return "keyword_search"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "keyword_search",
                "description": """Search for document chunks using keyword-based exact text matching (case-insensitive). Returns chunk IDs and abbreviated sentence snippets where the keywords appear.

IMPORTANT: This tool matches keywords literally in the text. Use SHORT, SPECIFIC terms (1-3 words maximum). Each keyword is matched independently.

Examples of GOOD keywords:
  - Entity names: "Albert Einstein", "Tesla", "Python", "Argentina"
  - Technical terms: "photosynthesis", "quantum mechanics"
  - Key concepts: "climate change", "GDP growth"

Examples of BAD keywords (DO NOT use):
  - Long phrases: "the person who invented the telephone" → use "Alexander Bell" instead
  - Questions: "when did World War 2 start" → use "World War 2", "1939" instead
  - Descriptions: "the country between France and Spain" → use "Andorra" instead
  - Full sentences: "how does the stock market work" → use "stock market", "trading" instead

RETURNS: Abbreviated snippets marked with "..." showing where keywords appear. These snippets help you identify relevant chunks, but you MUST use read_chunk to get the full text for answering questions.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of keywords to search. Each keyword should be 1-3 words maximum (e.g., ['Einstein', 'relativity theory', '1905'])."
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top-ranked chunks to return (default: 5, max: 20)",
                            "default": 5
                        }
                    },
                    "required": ["keywords"]
                }
            }
        }

    def execute(self, context: 'AgentContext', keywords: List[str], top_k: int = 5) -> Tuple[str, Dict[str, Any]]:
        top_k = min(top_k, 20)
        
        query_str = " ".join(keywords).lower()
        query_tokens = query_str.split()
        
        # Stage 1: BM25 coarse recall
        top_n_recall = top_k * 5 
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        if np.max(bm25_scores) == 0:
            return f"No results found for keywords: {keywords}", {"retrieved_tokens": 0, "chunks_found": 0}
            
        top_indices = np.argsort(bm25_scores)[::-1][:top_n_recall]
        
        # Stage 2: BGE-Reranker fine-grained cross-scoring
        cross_pairs = [[query_str, self.sentences[idx]] for idx in top_indices]
        rerank_scores = self.reranker.predict(cross_pairs)
        
        # Stage 3: Max-P Chunk aggregation algorithm
        chunk_scores = {}
        chunk_sents = {}
        
        for i, global_idx in enumerate(top_indices):
            score = float(rerank_scores[i])
            chunk_id = self.sentence_to_chunk[global_idx]
            sent = self.sentences[global_idx]
            orig_idx = self.sentence_orig_indices[global_idx]
            
            if chunk_id not in chunk_sents:
                chunk_sents[chunk_id] = []
                chunk_scores[chunk_id] = score
            else:
                chunk_scores[chunk_id] = max(chunk_scores[chunk_id], score)
                
            chunk_sents[chunk_id].append({
                'sentence': sent,
                'score': score,
                'orig_idx': orig_idx
            })
            
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Stage 4: UI/UX assembly (restore natural word order + get provenance)
        result_parts = []
        all_matched_sentences = []
        
        for chunk_id, max_score in sorted_chunks:
            sents_data = chunk_sents[chunk_id]
            # Use orig_idx to reorder by original text order without reading full text
            sents_sorted = sorted(sents_data, key=lambda x: x['orig_idx'])
            
            # Query SQLite for metadata provenance only
            location = "Unknown Source"
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT metadata FROM chunks WHERE id = ?", (chunk_id,))
                result = cursor.fetchone()
                if result and result[0]:
                    try:
                        meta = json.loads(result[0])
                        source = meta.get("source_document", "Unknown Document")
                        headers = " > ".join([v for k, v in meta.items() if k.startswith("Header")])
                        location = f"{source} | {headers}" if headers else source
                    except json.JSONDecodeError:
                        pass
            
            matched_text = "... " + " ... ".join([s['sentence'] for s in sents_sorted]) + " ..."
            # Include source for LLM mandatory citation
            result_parts.append(f"Source: [{location}] (Chunk: {chunk_id}, Rel: {max_score:.2f})\nMatched: {matched_text}")
            
            all_matched_sentences.extend([s['sentence'] for s in sents_sorted])
            
        tool_result = "\n\n".join(result_parts)
        
        # Stage 5: Bookkeeping and return
        sentences_text = "\n".join(all_matched_sentences)
        retrieved_tokens = len(self.tokenizer.encode(sentences_text)) if sentences_text else 0
        
        context.add_retrieval_log(
            tool_name="keyword_search",
            tokens=retrieved_tokens,
            metadata={
                "keywords": keywords,
                "chunks_found": len(sorted_chunks),
                "chunk_ids": [cid for cid, _ in sorted_chunks]
            }
        )
        
        return tool_result, {"retrieved_tokens": retrieved_tokens, "chunks_found": len(sorted_chunks)}
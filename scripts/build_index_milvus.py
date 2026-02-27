"""Build Milvus vector index from chunked data for ARAG."""

import os
# Set thread count to avoid libgomp warnings
os.environ["OMP_NUM_THREADS"] = "4"

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer

class MilvusIndexBuilder:
    """Builder for sentence-level Milvus vector database."""

    def __init__(
        self, 
        chunks_file: str, 
        collection_name: str = "arag_sentence_index",
        model_name: str = "/root/autodl-tmp/models/bge-large-zh-v1.5", 
        dim: int = 1024, 
        milvus_uri: str = "https://in03-68d51e54ccbb886.serverless.aws-eu-central-1.cloud.zilliz.com"
    ):
        self.chunks_file = Path(chunks_file)
        self.collection_name = collection_name
        self.dim = dim
        
        # Connect to remote Zilliz Cloud Serverless cluster
        print(f"🔗 Connecting to Zilliz Cloud at {milvus_uri}...")
        connections.connect(
            "default", 
            uri="https://in03-68d51e54ccbb886.serverless.aws-eu-central-1.cloud.zilliz.com",
            # Use username:password format for stable connection
            token="db_68d51e54ccbb886:Bs6<]jT{h.q2yYSx"
        )
        
        # Load Embedding model
        print(f"🧠 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def _create_collection(self) -> Collection:
        """Define and create Milvus collection (Schema)"""
        if utility.has_collection(self.collection_name):
            print(f"⚠️ Collection '{self.collection_name}' already exists. Dropping it for rebuild...")
            utility.drop_collection(self.collection_name)

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="sentence_text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        
        schema = CollectionSchema(
            fields=fields, 
            description="Sentence-level vector index for ARAG semantic search"
        )
        
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"✅ Collection '{self.collection_name}' created successfully.")
        return collection

    def build(self, batch_size: int = 256):
        """Execute full data ingestion and index building"""
        if not self.chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")

        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        collection = self._create_collection()
        
        all_sentences = []
        all_chunk_ids = []
        
        print("📦 Flattening chunks into sentences...")
        for chunk in chunks:
            chunk_id = chunk["id"]
            for sentence in chunk.get("sentences", []):
                all_sentences.append(sentence)
                all_chunk_ids.append(chunk_id)
                
        total_sentences = len(all_sentences)
        print(f"📊 Found {total_sentences} sentences to encode and insert.")

        for i in tqdm(range(0, total_sentences, batch_size), desc="Ingesting to Milvus"):
            batch_sentences = all_sentences[i : i + batch_size]
            batch_chunk_ids = all_chunk_ids[i : i + batch_size]
            
            # Encode and perform L2 normalization
            embeddings = self.model.encode(batch_sentences, normalize_embeddings=True)
            
            entities = [
                batch_chunk_ids,                    
                batch_sentences,                    
                embeddings.tolist()                 
            ]
            collection.insert(entities)
            
        collection.flush()
        print(f"💾 Flushed {collection.num_entities} entities to disk.")

        print("⚡ Building HNSW index for fast retrieval...")
        index_params = {
            "metric_type": "IP",  
            "index_type": "HNSW", 
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        
        collection.load()
        print("🚀 Index built and collection loaded into memory. Ready for search!")

if __name__ == "__main__":
    builder = MilvusIndexBuilder(
        chunks_file="/root/autodl-tmp/arag_data/chunks.json",
        model_name="/root/autodl-tmp/models/bge-large-zh-v1.5",
        dim=1024,
        milvus_uri="https://in03-68d51e54ccbb886.serverless.aws-eu-central-1.cloud.zilliz.com"
    )
    builder.build(batch_size=256)
"""Data ingestion pipeline: PDF -> Docling -> Dynamic Markdown Chunking -> Robust Sentence Splitting."""

import os
# Fix AutoDL multithreading environment variables
os.environ["OMP_NUM_THREADS"] = "4"


import json
import re
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import MarkdownHeaderTextSplitter

class DataIngestionPipeline:
    """Pipeline to ingest PDFs, convert to Markdown, and dynamically chunk."""
    
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)

        # Configure Docling: disable time-consuming and error-prone OCR, extract native text layer directly
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False 
        
        self.converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        # Define Markdown semantic splitting levels
        self.headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
            ("####", "Header_4"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False # Keep header text to prevent losing key entities
        )

    def _robust_sentence_split(self, text: str) -> List[str]:
        """
        Precise sentence splitting logic.
        Protects: abbreviations (e.g., i.e., Dr.), decimals (3.14), dots in emails/URLs, single-letter acronyms (U.S.A.)
        """
        text = re.sub(r'(?<=\d)\.(?=\d)', '<DECIMAL>', text)
        text = re.sub(r'(?<=[a-zA-Z0-9])\.(?=[a-zA-Z0-9-]+\.[a-zA-Z]{2,})', '<URL_DOT>', text)
        text = re.sub(r'(?<=[A-Z])\.', '<ACRONYM>', text)
        
        abbreviations = r'(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bProf)(?<!\be\.g)(?<!\bi\.e)(?<!\betc)(?<!\bvs)'
        

        # Core splitting rule: match Chinese and English punctuation (.!?。！？) 
        # \s* means optional space after punctuation (perfect for Chinese)
        pattern = abbreviations + r'([.!?。！？])\s*(?=[A-Z"\'\u4e00-\u9fa5])'
        
        splits = re.split(pattern, text)
        sentences = []
        
        if len(splits) > 0:
            current_sentence = splits[0]
            for i in range(1, len(splits), 2):
                punctuation = splits[i]
                next_part = splits[i+1] if i+1 < len(splits) else ""
                sentences.append((current_sentence + punctuation).strip())
                current_sentence = next_part
            if current_sentence.strip():
                sentences.append(current_sentence.strip())

        # Restore placeholders
        restored_sentences = []
        for s in sentences:
            s = s.replace('<DECIMAL>', '.')
            s = s.replace('<URL_DOT>', '.')
            s = s.replace('<ACRONYM>', '.')
            if len(s) > 10: 
                restored_sentences.append(s)
                
        # ==========================================
        # Fallback strategy: handle oversized paragraphs exceeding BGE model limits (e.g., large tables or long sentences without punctuation)
        # ==========================================
        final_sentences = []
        MAX_SEQ_LENGTH = 500 # Align with 512 token limit for embedding models
        
        for s in restored_sentences:
            if len(s) <= MAX_SEQ_LENGTH:
                final_sentences.append(s)
            else:
                # Force split long paragraphs every 500 characters
                for i in range(0, len(s), MAX_SEQ_LENGTH):
                    final_sentences.append(s[i : i + MAX_SEQ_LENGTH])
                    
        return final_sentences

    def process(self):
        """Execute the complete data ingestion pipeline"""
        all_chunks: List[Dict[str, Any]] = []
        chunk_counter = 0
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        pdf_files = list(self.input_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files to process.")
        
        if len(pdf_files) == 0:
            print("⚠️ Warning: No PDF files found! Please ensure PDFs are uploaded to data/raw_pdfs directory.")
            return

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}...")
            
            # Docling: PDF -> Markdown
            conversion_result = self.converter.convert(str(pdf_path))
            md_text = conversion_result.document.export_to_markdown()
            
            # Save Markdown result locally
            md_output_path = self.input_dir / f"{pdf_path.stem}.md"
            md_output_path.write_text(md_text, encoding="utf-8")
            
            # Dynamic semantic splitting
            docs = self.md_splitter.split_text(md_text)
            
            # Data assembly and precise sentence extraction
            for doc in docs:
                chunk_text = doc.page_content
                metadata = doc.metadata
                metadata["source_document"] = pdf_path.name
                
                sentences = self._robust_sentence_split(chunk_text)
                
                if not sentences:
                    continue
                    
                chunk_id = f"chunk_{chunk_counter:05d}"
                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": metadata, 
                    "sentences": sentences 
                })
                chunk_counter += 1

        # Export to JSON for downstream architecture
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Successfully processed {chunk_counter} semantic chunks into {self.output_file}")
        
        # Call SQLite export code here
        self._export_to_sqlite(all_chunks)

    def _export_to_sqlite(self, all_chunks: List[Dict[str, Any]]):
        """Store chunked text and metadata into SQLite Document Store"""
        # Ensure SQLite path is on global data disk
        db_path = "/root/autodl-tmp/arag_data/arag_docs.db"
        print(f"🗄️ Exporting {len(all_chunks)} chunks to Document Store: {db_path}...")
        
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    metadata TEXT
                )
            ''')
            cursor.execute('DELETE FROM chunks')
            
            insert_data = [
                (c["id"], c["text"], json.dumps(c["metadata"], ensure_ascii=False)) 
                for c in all_chunks
            ]
            cursor.executemany('INSERT INTO chunks (id, text, metadata) VALUES (?, ?, ?)', insert_data)
            conn.commit()
        print("✅ Document Store built successfully!")


if __name__ == "__main__":
    # Ensure input/output paths align with AutoDL real directory
    pipeline = DataIngestionPipeline(
        input_dir="./data/raw_pdfs",
        output_file="/root/autodl-tmp/arag_data/chunks.json"
    )
    pipeline.process()
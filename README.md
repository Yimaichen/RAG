# ARAG - Agentic Retrieval-Augmented Generation Framework

ARAG is an Agent-based Retrieval-Augmented Generation (RAG) framework that efficiently handles complex document Q&A tasks through tool calling and multi-turn retrieval strategies. The framework features a modular design and supports the combined use of various retrieval tools, such as keyword search, semantic search, and document reading.

## Project Structure

Plaintext

```
arag/
├── src/arag/               # Core source code
│   ├── core/               # Core components
│   │   ├── config.py       # Configuration management (supports YAML/JSON)
│   │   ├── context.py      # Agent execution context
│   │   └── llm.py          # LLM client (supports OpenAI-compatible APIs)
│   ├── agent/              # Agent implementation
│   │   ├── base.py         # Base Agent (supports tool calling)
│   │   └── prompts/        # System prompt templates
│   └── tools/              # Retrieval toolset
│       ├── base.py         # Tool abstract base class
│       ├── registry.py     # Tool registry management
│       ├── keyword_search.py   # BM25 keyword search
│       ├── semantic_search.py  # Milvus vector semantic search
│       └── read_chunk.py       # Document content reading
├── scripts/                # Data processing and evaluation scripts
│   ├── data_ingestion.py   # PDF document parsing and chunking
│   ├── build_index_milvus.py   # Milvus vector index building
│   ├── batch_runner.py     # Batch Q&A execution
│   ├── eval.py             # Result evaluation
│   └── eval_ragas.py       # RAGAS metrics evaluation
├── configs/                # Configuration files
├── tests/                  # Unit tests
├── data/                   # Data directory (Git ignored)
└── results/                # Results output (Git ignored)
```

## Core Features

### 1. Hybrid Retrieval Architecture

- **Keyword Search**: Implements exact matching based on the BM25 algorithm, suitable for queries with known entities and terms.
- **Semantic Search**: Implements semantic similarity retrieval based on the Milvus vector database, supporting conceptual queries.
- **Reranking Optimization**: Integrates the BGE-Reranker model for fine-grained ranking of retrieval results.
- **Document Reading**: Supports on-demand reading of full document content to avoid context overflow.

### 2. Agent Workflow

- **Multi-turn Tool Calling**: The Agent can autonomously determine the execution order of searching, reading, and answering.
- **Token Budget Management**: Built-in Token consumption monitoring to prevent exceeding context limits.
- **Trajectory Tracking**: Fully records the parameters and results of each tool calling turn.
- **Cost Control**: Calculates API call costs in real time.

### 3. Data Pipeline

- **PDF Parsing**: Achieves high-quality PDF to Markdown conversion based on Docling.
- **Smart Chunking**: Supports hierarchical chunking based on Markdown headings.
- **Sentence-level Indexing**: Builds sentence-level vector indices to improve retrieval accuracy.
- **Metadata Retention**: Fully preserves document source and section hierarchy information.

## Quick Start

### Environment Setup

Bash

```
# Clone the repository
git clone https://github.com/Yimaichen/RAG.git
cd arag

# Install dependencies
uv pip install -e ".[full]"

# Or use pip
pip install -e ".[full]"
```

### Configure API Keys

Copy the environment variable template and fill in your API keys:

Bash

```
cp .env.example .env
```

Edit the `.env` file:

```
ARAG_API_KEY=your-api-key-here
ARAG_BASE_URL=https://api.openai.com/v1
ARAG_MODEL=gpt-4o-mini
```

### Data Preparation

1. Place PDF documents in the `data/raw_pdfs/` directory.
2. Run the data parsing pipeline (you also need an account from Zilliz):

Bash

```
python scripts/data_ingestion.py
```

1. Build the vector index:

Bash

```
python scripts/build_index_milvus.py
```

### Run Q&A

Bash

```
python scripts/batch_runner.py \
    --config configs/example.yaml \
    --questions data/questions.json \
    --output results/ \
    --workers 10
```

### Evaluate Results

Bash

```
python scripts/eval.py \
    --predictions results/predictions.jsonl \
    --workers 10
```

## Configuration File Instructions

The configuration file uses the YAML format and mainly includes the following sections:

YAML

```
llm:
  api_key: "your-api-key"      # API key (or read from environment variables)
  base_url: "https://api.openai.com/v1"
  model: "gpt-4o-mini"
  temperature: 0.0
  max_tokens: 4096

embedding:
  model: "/path/to/bge-large-zh-v1.5"  # Embedding model path
  device: "cuda:0"
  batch_size: 16

agent:
  max_loops: 15                # Maximum number of tool calling loops
  max_token_budget: 128000     # Token budget limit
  verbose: true

data:
  chunks_file: "/path/to/chunks.json"
  questions_file: "data/questions.json"

output:
  results_dir: "results/"
```

## Supported Models

### LLM Models

The framework supports all models with OpenAI-compatible APIs, including:

- OpenAI GPT series (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
- Anthropic Claude series (claude-4-opus, claude-4-sonnet, etc.)
- Google Gemini series (gemini-2.5-pro, gemini-2.5-flash, etc.)
- Other services compatible with the OpenAI API format

### Embedding Models

The following models are recommended:

- **Chinese**: BAAI/bge-large-zh-v1.5
- **English**: BAAI/bge-large-en-v1.5

### Reranking Models

- BAAI/bge-reranker-v2-m3

## Retrieval Tools Details

### KeywordSearchTool

Uses the BM25 algorithm for keyword matching, suitable for finding document fragments containing specific entities or terms. It supports exact matching of 1-3 keywords and reranks the results using the BGE-Reranker.

### SemanticSearchTool

Implements semantic retrieval based on the Milvus vector database. After vectorizing the query, it performs an approximate nearest neighbor search via the HNSW index, and then a Cross-Encoder calculates the relevance score between the query and sentences.

### ReadChunkTool

Reads the full document content for a specified ID from the SQLite document store. It supports duplicate read detection to avoid Token waste.



### Running Tests

Bash

```
uv run pytest tests/ -v
```

## Evaluation Metrics

The framework provides multi-dimensional evaluation capabilities:

- **LLM Accuracy**: Uses an LLM to judge the consistency between the predicted answer and the ground truth.
- **Contain Accuracy**: Checks whether the ground truth is contained within the predicted answer.
- **RAGAS Metrics**:
  - Context Precision: Retrieval precision
  - Context Recall: Retrieval recall
  - Faithfulness: Answer faithfulness
  - Answer Relevancy: Answer relevancy

# Scientific Research Assistant

A Retrieval-Augmented Generation (RAG) system designed to answer scientific questions using a collection of research papers across multiple technical domains.

The system retrieves relevant passages from scientific documents and uses a Large Language Model (LLM) to generate answers grounded in the retrieved context.

## Features

- PDF-based scientific document ingestion
- Reference-section removal during document preprocessing
- Recursive and semantic text chunking
- Multilingual embeddings using `intfloat/multilingual-e5-base`
- Persistent vector storage with Qdrant
- Domain-aware query routing
- Multi-domain document retrieval
- RAG workflow orchestration with LangGraph
- Streaming answer generation
- Source identification and citation validation
- Terminal interface for interactive questions

## Supported Domains

The current document collection covers three domains:

- Large Language Models (`llm`)
- Machine Learning (`machine_learning`)
- Computer Vision (`computer_vision`)

The collection contains 18 research papers.

## Technology Stack

| Component              | Technology                      |
| ---------------------- | ------------------------------- |
| Programming Language   | Python                          |
| RAG Framework          | LangChain                       |
| Workflow Orchestration | LangGraph                       |
| Vector Database        | Qdrant                          |
| Embedding Model        | `intfloat/multilingual-e5-base` |
| LLM API                | OpenRouter                      |
| Terminal Interface     | Rich                            |

## Project Architecture

The main RAG workflow consists of the following stages:

1. **Domain Routing:** Identifies the research domains relevant to the user's question.
2. **Retrieval:** Searches the Qdrant vector database for relevant document chunks.
3. **Context Building:** Organizes retrieved passages and their source metadata.
4. **Generation:** Produces an answer using the retrieved context.
5. **Citation Validation:** Checks whether the generated citations refer to available sources.

The workflow is orchestrated using LangGraph.

## Installation

Clone the repository:

```bash
git clone <https://github.com/ItsJavadHosseini/Scientific-Research-Assistant.git>
cd Scientific-Research-Assistant
```

Install project dependencies:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=YOUR_OPENROUTER_API_KEY
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

EMBEDDING_MODEL=intfloat/multilingual-e5-base
QDRANT_PATH=storage/qdrant
```

Replace `YOUR_OPENROUTER_API_KEY` with your own API key.and your model.

The embedding model must be available in the local Hugging Face cache because the current implementation loads it in local-only mode.

## Running the Application

Start the interactive terminal application:

```bash
uv run python run_rag.py
```

Enter a scientific question when prompted.

To exit:

```text
exit
```

## Data Processing

The document-processing pipeline includes:

- PDF loading and page extraction
- Reference-section removal
- Short-page filtering
- Text chunking
- Embedding generation
- Vector indexing in Qdrant

The current dataset statistics are:

| Metric               | Value |
| -------------------- | ----: |
| PDF documents        |    18 |
| Original pages       |   374 |
| Retained pages       |   282 |
| Removed pages        |    92 |
| Page retention rate  | 75.4% |
| Recursive chunks     | 1,185 |
| Embedding dimensions |   768 |

## Evaluation

The system includes tests for:

- Domain routing
- Retrieval
- Full RAG workflow execution
- Out-of-domain questions
- Insufficient retrieval evidence
- Citation ID validation

A complete quantitative evaluation and comparison of retrieval strategies should be documented in the Evaluation Report.

## Project Reports

Implementation details and development decisions are documented in separate reports:

1. Project Overview
2. Data Ingestion Report
3. Chunking Report
4. Vector Database Report
5. Retrieval Report
6. RAG Pipeline Report
7. Evaluation Report
8. Final Documentation

## Current Limitations

- Citation validation checks citation IDs, not the semantic correctness of source support.
- Retrieval quality depends on the quality of the indexed documents and chunking strategy.
- The embedding model currently requires a local model cache.
- The final interactive application and complete evaluation results must be verified before release.

## License

MIT

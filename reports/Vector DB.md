
# Vector Database Implementation Report

## 1. Overview

The purpose of this stage was to implement a persistent vector database for storing and retrieving the embedded research paper chunks.

We selected **Qdrant** as the vector database and used its local persistent storage mode. The system stores two separate collections to support a comparison between Recursive and Semantic Chunking strategies.

The main objectives were:

- Store embedded document chunks in Qdrant.
    
- Preserve document content and metadata.
    
- Support semantic similarity search.
    
- Enable metadata-based filtering.
    
- Maintain stable point identifiers.
    
- Test retrieval quality using sample research questions.
    

## 2. Technology Stack

|Component|Technology|
|---|---|
|Vector Database|Qdrant|
|Client Library|qdrant-client|
|Embedding Model|intfloat/multilingual-e5-base|
|Embedding Dimension|768|
|Similarity Metric|Cosine Similarity|
|Storage Mode|Local Persistent Storage|
|Batch Size|64|

The same embedding model used during the embedding stage was used to generate vectors for both collections.

All embeddings were generated on the CPU.

## 3. Database Architecture

Two independent collections were created:

|Collection|Source|
|---|---|
|`research_recursive`|Recursive Chunking output|
|`research_semantic`|Final Semantic Chunking output|

Both collections contain research paper chunks from all three domains:

- Large Language Models
    
- Machine Learning
    
- Computer Vision
    

Separating the collections allows us to compare the retrieval performance of the two chunking strategies under the same embedding model and similarity metric.

The Qdrant database uses local persistent storage:

```
storage/
└── qdrant/
```

This configuration allows the database to retain indexed vectors between program executions without requiring a separate Qdrant server or Docker container.

## 4. Collection Configuration

Both collections use the same vector configuration.

|Parameter|Value|
|---|---|
|Vector Size|768|
|Distance Metric|COSINE|
|Storage|Local Persistent|
|Collection Names|`research_recursive`, `research_semantic`|

Cosine similarity was selected to measure the semantic similarity between query embeddings and document embeddings.

## 5. Point Identification

To ensure consistent point identifiers, UUID5 was used instead of randomly generated UUIDs.

Each point ID is generated from a combination of the collection name and the chunk ID.

```
def generate_point_id(
    chunk_id: str,
    collection_name: str,
) -> str:
    unique_key = f"{collection_name}:{chunk_id}"
    return str(uuid5(NAMESPACE_URL, unique_key))
```

This approach provides deterministic identifiers.

The same chunk produces the same point ID when indexed again in the same collection.

Including the collection name ensures that identical chunk IDs in different collections do not produce conflicting identifiers.

## 6. Payload Structure

Each Qdrant point contains:

1. Its vector representation.
    
2. The original chunk text.
    
3. The associated document metadata.
    

An example payload:

```
{
    "page_content": "The original text of the research chunk...",
    "paper_id": "resnet",
    "title": "Deep Residual Learning for Image Recognition",
    "category": "computer_vision",
    "year": 2015,
    "page": 3,
    "page_index": 2,
    "chunk_id": "resnet-p3-c002",
    "chunking_version": "recursive-v1"
}
```

Storing the original text alongside the vector is important because the retrieved content will later be passed to the LLM as context.

Metadata also enables filtering and citation generation.

For example, the `paper_id` and `page` fields allow the system to identify the source paper and page number of retrieved information.

## 7. Indexing Process

The indexing pipeline was implemented in:

```
src/scientific_research_assistant/vectorstore/qdrant.py
```

The indexing process follows these steps:

1. Load the processed chunks from JSONL files.
    
2. Initialize the embedding provider.
    
3. Create or validate the target Qdrant collection.
    
4. Divide chunks into batches of 64.
    
5. Generate embeddings for each batch.
    
6. Construct Qdrant points containing vectors and payloads.
    
7. Generate deterministic point IDs.
    
8. Insert points using the Qdrant `upsert` operation.
    

The same process is applied independently to both collections.

Batch processing reduces the memory overhead associated with generating and uploading embeddings for the entire dataset at once.

## 8. Retrieval Implementation

A basic semantic search function was implemented using Qdrant's `query_points` API.

The retrieval process:

1. Receives a user question.
    
2. Generates its embedding using the query prefix.
    
3. Searches the selected collection.
    
4. Optionally applies metadata filters.
    
5. Returns the top-K most similar chunks.
    

The system supports filtering by `paper_id`.

For example, a query can be restricted to the ResNet paper:

```
results = search_chunks(
    client=client,
    embedding_provider=embedding_provider,
    collection_name="research_recursive",
    question="How does ResNet address optimization difficulties?",
    top_k=5,
    paper_id="resnet",
)
```

This filtering capability will be useful for implementing domain routing and paper-specific questions in later stages.

## 9. Retrieval Testing

To verify the database and retrieval pipeline, a test was performed using the following question:

**Question:**

> ResNet چگونه مشکل vanishing gradient را حل می‌کند؟

The question was embedded and searched against both collections, with the search restricted to the `resnet` paper.

The top three results from each collection were inspected.

### Recursive Collection

|Rank|Chunk ID|Page|Similarity Score|
|---|---|---|---|
|1|`resnet-p5-c002`|5|0.8431|
|2|`resnet-p5-c003`|5|0.8267|
|3|`resnet-p2-c002`|2|0.8261|

### Semantic Collection

|Rank|Chunk ID|Page|Similarity Score|
|---|---|---|---|
|1|`resnet-p5-c002`|5|0.8428|
|2|`resnet-p1-c005`|1|0.8280|
|3|`resnet-p2-c003`|2|0.8239|

Both collections successfully retrieved relevant content from the ResNet paper.

The results demonstrate that:

- The embedding pipeline is working.
    
- Qdrant successfully stores and retrieves vectors.
    
- Metadata filtering by `paper_id` works.
    
- Retrieved chunks contain the original text and citation metadata.
    
- Both chunking strategies can retrieve relevant information for the tested question.
    

### Important Observation

The highest similarity scores from both collections were very close:

- Recursive: 0.8431
    
- Semantic: 0.8428
    

This small difference is not sufficient to conclude that one chunking strategy performs better than the other.

A more comprehensive retrieval evaluation using multiple questions and appropriate retrieval metrics is required.

Additionally, the retrieved ResNet content highlights an important distinction: the paper discusses the degradation problem and optimization difficulties in deep networks, while noting that vanishing and exploding gradients had largely been addressed through other techniques. Residual connections primarily help optimization in increasingly deep networks.

## 10. Implementation Status

|Component|Status|
|---|---|
|Qdrant Client Setup|Completed|
|Local Persistent Storage|Completed|
|Recursive Collection|Created|
|Semantic Collection|Created|
|Vector Configuration|Completed|
|Deterministic Point IDs|Implemented|
|Metadata Payload|Implemented|
|Batch Indexing|Implemented|
|Semantic Search|Implemented|
|Metadata Filtering|Tested|
|Basic Retrieval Test|Passed|
|Comprehensive Retrieval Evaluation|Pending|

## 11. Output Artifacts

The main implementation files are:

```
src/scientific_research_assistant/
└── vectorstore/
    └── qdrant.py

tests/
└── test_qdrant.py

storage/
└── qdrant/
```

The processed chunk files used as indexing inputs are:

```
data/processed/
├── recursive_chunks.jsonl
└── semantic_chunks_final.jsonl
```

## 12. Conclusion

The vector database stage was successfully implemented.

Qdrant now provides persistent storage for both Recursive and Semantic Chunking outputs. Each document chunk is stored with its vector representation, original text, and metadata.

Initial retrieval tests confirmed that the system can retrieve relevant research content using semantic similarity and metadata filtering.

At this stage, the project has a functional document retrieval infrastructure.

The next step is to build a dedicated Retriever abstraction that connects Qdrant with the LangChain document-processing pipeline. This will provide a clean interface for the upcoming RAG chain and LangGraph workflow.
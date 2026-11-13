
# RAG Pipeline Implementation Report

## 1. Overview

The objective of this phase was to implement an end-to-end Retrieval-Augmented Generation (RAG) pipeline for the Scientific Research Assistant project.

The system is designed to answer scientific questions using a collection of 18 research papers across three domains:

- Large Language Models (LLM)
    
- Machine Learning
    
- Computer Vision
    

The pipeline combines semantic retrieval, domain classification, context construction, LLM-based answer generation, and citation validation.

The main objective was to build a modular RAG architecture that can retrieve relevant scientific information and generate context-grounded answers with source references.

---

## 2. Technology Stack

|Component|Technology|
|---|---|
|Programming Language|Python|
|RAG Framework|LangChain|
|Workflow Orchestration|LangGraph|
|Vector Database|Qdrant|
|Embedding Model|intfloat/multilingual-e5-base|
|LLM Provider|OpenRouter|
|LLM Model|nvidia/nemotron-3-ultra-550b-a55b|
|Terminal Interface|Rich|
|Data Validation|Pydantic|

---

## 3. Embedding Integration

The previously implemented embedding provider was integrated into the retrieval pipeline.

Configuration:

- Model: `intfloat/multilingual-e5-base`
    
- Embedding Dimension: 768
    
- Device: CPU
    
- Query Prefix: `query:`
    
- Document Prefix: `passage:`
    
- Vector Normalization: Enabled
    

The embedding provider was configured to load model files from the local cache using:

`local_files_only=True`

This avoids unnecessary Hugging Face Hub requests during model initialization.

The model was successfully loaded and tested.

---

## 4. Retrieval Implementation

A modular Retriever class was implemented.

Main responsibilities:

- Connecting to the local Qdrant database.
    
- Selecting the appropriate collection based on chunking strategy.
    
- Performing similarity search.
    
- Supporting category filtering.
    
- Supporting paper-specific filtering.
    
- Converting retrieved records into LangChain Document objects.
    
- Removing duplicate points.
    
- Sorting retrieved documents by similarity score.
    

Supported strategies:

- Recursive Chunking
    
- Semantic Chunking
    

The retriever supports multiple domain categories in a single query.

The `top_k` parameter controls the maximum number of retrieved documents across the selected domains.

---

## 5. Domain Routing

A DomainRouter component was implemented to classify incoming questions into the available research domains.

Supported categories:

- `llm`
    
- `machine_learning`
    
- `computer_vision`
    

PydanticOutputParser was used to parse and validate the LLM routing output.

This approach was selected instead of relying on structured output because of compatibility issues observed with the OpenRouter model.

The router supports:

- Single-domain questions.
    
- Multi-domain questions.
    
- Out-of-domain questions.
    
- Manual category overrides.
    

### Routing Tests

|Question|Expected Routing|
|---|---|
|ResNet architecture|Computer Vision + Machine Learning|
|Attention in Transformers|LLM|
|Capital of France|Out-of-Domain|

The routing component successfully passed these tests.

---

## 6. RAG Prompt Engineering

A dedicated RAG prompt was implemented.

The prompt instructs the LLM to:

- Answer using the retrieved context.
    
- Avoid fabricating scientific information.
    
- Include source references using `[S1]`, `[S2]`, etc.
    
- Abstain when the retrieved context is insufficient.
    
- Identify conflicting information when applicable.
    
- Generate answers in the user's language.
    

The prompt separates the question from the retrieved context.

---

## 7. LangGraph Workflow Architecture

The original sequential RAG execution was integrated into LangGraph.

The workflow consists of the following nodes:

1. Router
    
2. Retriever
    
3. Context Builder
    
4. Generation
    
5. Citation Validator
    
6. Out-of-Domain Handler
    
7. Insufficient Evidence Handler
    

### Workflow Execution

Successful RAG path:

START → Router → Retriever → Context Builder → Generation → Citation Validator → END

Out-of-domain path:

START → Router → Out-of-Domain Handler → END

Insufficient evidence path:

START → Router → Retriever → Insufficient Evidence Handler → END

Conditional edges determine which execution path should be followed.

The workflow is compiled using:

`builder.compile()`

---

## 8. State Management

A TypedDict-based RAGState was implemented.

The state contains:

- question
    
- category
    
- paper_id
    
- domains
    
- routing_reasoning
    
- documents
    
- context
    
- answer
    
- generation_time
    
- sources
    
- used_citations
    
- citation_valid
    
- citation_errors
    
- error
    

This shared state allows individual LangGraph nodes to exchange information throughout the workflow.

---

## 9. Generation and Citation Validation

The generation node uses LangChain ChatOpenAI with the OpenRouter API.

Streaming generation is implemented using the LLM's `.stream()` interface.

The generated response is checked by the citation validator.

The validator:

- Extracts citation identifiers.
    
- Checks whether referenced source IDs exist.
    
- Detects invalid citation identifiers.
    
- Records citation validation errors.
    

Important limitation:

The current validator checks citation identifier consistency, not semantic evidence support. It does not establish whether a cited paper actually supports the associated scientific claim.

---

## 10. RAGChain Integration

The public RAGChain interface was integrated with LangGraph.

The intended public API remains:

```
result = rag_chain.invoke(
    "How does residual learning work in ResNet?"
)
```

The RAGChain constructor initializes:

- Retriever
    
- DomainRouter
    
- LLM
    
- LangGraph workflow
    

The compiled graph is stored in:

`self.graph`

The objective is to preserve the existing public interface while delegating internal execution to LangGraph.

---

## 11. Testing Results

### 11.1 Component Testing

The following components were individually tested:

- Embedding Provider
    
- Domain Router
    
- Retriever
    
- Context Builder
    
- Generation Node
    
- Citation Validator
    
- Out-of-Domain Handler
    
- Insufficient Evidence Handler
    
- LangGraph Workflow
    

### 11.2 Full Graph Execution

Test question:

`How does residual learning work in ResNet?`

Observed output:

|Metric|Result|
|---|---|
|Selected Domains|Computer Vision, Machine Learning|
|Retrieved Sources|3|
|Citation Validation|Passed|
|Generation Time|9.18 seconds|
|Workflow Execution|Successful|

The generated response explained:

- Shortcut connections
    
- Identity mapping
    
- Residual functions
    
- The degradation problem
    
- Training deeper neural networks
    

Three source references were returned.

### 11.3 Out-of-Domain Testing

Question:

`What is the capital of France?`

The router correctly selected no research domain and the workflow returned an out-of-domain response.

### 11.4 Insufficient Evidence Testing

The insufficient evidence branch was tested using a mock retriever that returned no documents.

The workflow correctly returned the insufficient-evidence response.

---

## 12. Interactive Application

A command-line interface was introduced through:

`run_rag.py`

The application initializes RAGChain and accepts multiple user questions through an interactive loop.

Example usage:

```
rag_chain = RAGChain(
    strategy="recursive",
    top_k=5,
    verbose=True,
)

result = rag_chain.invoke(
    "How does ResNet work?"
)
```

The interface is designed to display:

- User questions
    
- Selected research domains
    
- Generated answers
    
- Retrieved sources
    
- Execution information
    

The current implementation is a question-answering system without conversational memory.

---

## 13. Current Implementation Status

|Component|Status|
|---|---|
|Data Ingestion|Completed|
|Chunking|Completed|
|Embedding Integration|Implemented and tested|
|Qdrant Integration|Completed|
|Retriever|Implemented|
|Domain Router|Implemented and tested|
|Prompt Engineering|Implemented|
|LangGraph State|Implemented|
|LangGraph Nodes|Implemented|
|Workflow Construction|Implemented|
|Direct Graph Execution|Successfully tested|
|RAGChain Public API|Integration issue remains|
|Interactive CLI|Implemented; end-to-end execution needs final verification|
|Comprehensive Evaluation|Pending|

---

## 14. Known Issues and Limitations

### 14.1 RAGChain Invocation Error

During interactive execution, the following exception occurred:

```
ValueError: No checkpointer set
```

The traceback indicates that `RAGChain.invoke()` calls:

```
self.graph.get_state(initial_state).values
```

However, the graph was compiled without a checkpointer.

The intended correction is to execute the graph using:

```
result = self.graph.invoke(initial_state)
```

This correction has been proposed but its successful execution through the interactive application has not yet been verified.

### 14.2 Citation Validation

The current citation validator checks identifier consistency only.

Semantic citation verification remains unimplemented.

### 14.3 Evaluation

The system has passed functional tests, but comprehensive quantitative evaluation has not yet been performed.

---

## 15. Conclusion

The RAG pipeline architecture has been implemented using LangChain and LangGraph.

The individual components and the compiled LangGraph workflow have successfully passed functional tests.

A real ResNet question was processed through the graph, producing a context-grounded answer with three source references.

The main remaining issue is the final integration of the public `RAGChain.invoke()` method with the compiled graph.

Once this issue is resolved and the interactive application passes an end-to-end test, the project will have a usable initial RAG MVP.

The next development phase is comprehensive RAG evaluation, followed by retrieval and generation optimization.

# Chunking and Post-processing Report

## 1. Overview

The purpose of this stage was to divide the extracted research papers into smaller text segments (chunks) suitable for semantic search and retrieval-augmented generation (RAG).

Two chunking strategies were implemented and evaluated:

- Recursive Character Text Splitting
    
- Semantic Chunking
    

Both approaches were applied to the 282 cleaned pages obtained during the ingestion stage.

The resulting chunks were processed to remove obvious noise and improve their suitability for downstream embedding and retrieval operations.

## 2. Recursive Chunking

Recursive chunking was implemented using LangChain's `RecursiveCharacterTextSplitter`.

The splitter uses a hierarchical set of separators to preserve the natural structure of the text whenever possible.

### Configuration

|Parameter|Value|
|---|---|
|Chunk size|1,000 characters|
|Chunk overlap|200 characters|
|Length function|`len`|
|Separators|`\n\n`, `\n`, space, empty string|

The overlap between consecutive chunks helps preserve contextual continuity when relevant information spans multiple segments.

### Results

|Metric|Value|
|---|---|
|Input pages|282|
|Total chunks|1,185|
|Minimum chunk length|111|
|Maximum chunk length|1,000|
|Average chunk length|878.75|

Each chunk was assigned a unique identifier and a chunking version:

- `chunk_id`: Identifies a chunk using its paper ID, page number, and local chunk index.
    
- `chunking_version`: Set to `recursive-v1`.
    

The resulting dataset was saved as:

`data/processed/recursive_chunks.jsonl`

## 3. Semantic Chunking

Semantic chunking was implemented using LangChain Experimental's `SemanticChunker`.

Unlike fixed-size splitting, this approach uses embedding similarity between neighboring text segments to identify semantic boundaries.

The implementation processes each page independently to preserve page-level metadata and citation accuracy.

### Configuration

|Parameter|Value|
|---|---|
|Chunking algorithm|SemanticChunker|
|Breakpoint threshold|Percentile|
|Embedding model|`intfloat/multilingual-e5-base`|
|Maximum chunk size|1,000 characters|
|Safety splitter|RecursiveCharacterTextSplitter|
|Chunk overlap for safety splitting|200 characters|
|Processing device|CPU|

A recursive safety splitter was applied to semantic chunks exceeding 1,000 characters.

### Results

|Metric|Value|
|---|---|
|Input pages|282|
|Total chunks|1,442|
|Minimum chunk length|1 character|
|Maximum chunk length|1,000 characters|
|Average chunk length|689 characters|
|Processing time|Approximately 29 minutes 50 seconds|

Semantic chunking generated more chunks than recursive splitting while producing shorter average segments.

The raw output was saved as:

`data/processed/semantic_chunks_raw.jsonl`

## 4. Chunking Strategy Comparison

The following table summarizes the results of both approaches.

|Metric|Recursive|Semantic|
|---|---|---|
|Input pages|282|282|
|Total chunks|1,185|1,442|
|Minimum length|111|1|
|Maximum length|1,000|1,000|
|Average length|878.75|689|

Semantic chunking produced approximately 21.7% more chunks than recursive chunking.

This indicates that semantic segmentation generated smaller text units on average, potentially providing more precise retrieval boundaries.

However, chunk count and length statistics alone do not establish retrieval superiority. A retrieval evaluation is required to compare their actual performance.

## 5. Semantic Chunk Post-processing

An inspection of the raw semantic chunks revealed several quality issues:

- Numeric-only chunks containing PDF page numbers.
    
- Punctuation-only chunks.
    
- Incomplete sentences split across neighboring chunks.
    
- Bibliographic references appearing as independent chunks.
    
- Page numbers and journal pagination appended to meaningful text.
    
- Short but meaningful scientific statements and mathematical expressions.
    

Rather than removing all short chunks, a conservative cleaning strategy was adopted to avoid losing useful scientific information.

### 5.1 Obvious Noise Removal

The first cleaning pass removed:

- Empty chunks.
    
- Numeric-only chunks.
    
- Punctuation-only chunks.
    

Results:

|Metric|Value|
|---|---|
|Original semantic chunks|1,442|
|Removed chunks|36|
|Remaining chunks|1,406|
|Removal rate|2.50%|

The cleaned intermediate dataset was saved as:

`data/processed/semantic_chunks_cleaned.jsonl`

### 5.2 Additional Post-processing

A second processing stage was implemented to:

- Remove selected, manually identified bibliography fragments.
    
- Remove known publication metadata fragments.
    
- Remove trailing PDF page numbers when they match the chunk's page metadata.
    
- Remove identified journal pagination artifacts in the Dropout paper.
    
- Preserve short but meaningful scientific content.
    

Automatic merging of incomplete chunks was intentionally excluded to avoid introducing incorrect text combinations or damaging the original document structure.

The final output is:

`data/processed/semantic_chunks_final.jsonl`

The final chunk count and length statistics are generated automatically by the post-processing script.

## 6. Metadata Preservation

Metadata was preserved throughout both chunking pipelines.

Each chunk contains the original paper metadata together with additional chunk-specific fields.

Important fields include:

|Field|Description|
|---|---|
|`paper_id`|Unique identifier of the research paper|
|`title`|Paper title|
|`category`|Research domain|
|`year`|Publication year|
|`page`|Human-readable PDF page number|
|`page_index`|Zero-based original page index|
|`chunk_id`|Unique chunk identifier within its paper and page|
|`chunking_version`|Chunking strategy identifier|

Preserving page-level metadata is essential for generating source citations in the final RAG responses.

## 7. Output Artifacts

The following datasets were generated during this stage:

|File|Description|
|---|---|
|`recursive_chunks.jsonl`|Recursive chunking output|
|`semantic_chunks_raw.jsonl`|Original semantic chunking output|
|`semantic_chunks_cleaned.jsonl`|Semantic chunks after initial noise removal|
|`semantic_chunks_final.jsonl`|Final post-processed semantic chunks|

All files are stored in:

`data/processed/`

## 8. Conclusion

Both recursive and semantic chunking pipelines were successfully implemented and applied to the 282 cleaned pages extracted during ingestion.

Recursive chunking provides relatively uniform chunk sizes and predictable segmentation, while semantic chunking attempts to preserve semantic relationships by identifying context-dependent boundaries.

A conservative post-processing pipeline was also introduced to remove obvious noise and selected bibliographic artifacts without indiscriminately discarding short scientific content.

The project now has two chunking datasets ready for the next stage.

**The next stage is vector storage and retrieval using Qdrant.** Both chunking strategies will be embedded using the same embedding model and stored in separate collections to enable a controlled comparison of their retrieval performance.
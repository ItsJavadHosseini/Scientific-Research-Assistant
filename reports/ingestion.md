# Data Ingestion Report

## 1. Overview

The data ingestion stage is the first step in building the Scientific Research Assistant. The main objective of this stage is to extract text from the research papers in PDF format, clean the extracted pages, and convert them into structured LangChain `Document` objects.

The dataset consists of 18 scientific research papers distributed across three domains:

- Computer Vision
    
- Large Language Models (LLMs)
    
- Machine Learning
    

Each paper is processed independently to preserve its original structure and associated metadata.

## 2. Dataset Overview

The dataset contains 18 PDF files, with six papers in each research domain.

|Research Domain|Number of Papers|
|---|---|
|Computer Vision|6|
|Large Language Models|6|
|Machine Learning|6|
|**Total**|**18**|

A metadata catalog was created to map each PDF filename to its corresponding paper information.

The metadata associated with each paper includes:

- `paper_id`: A unique identifier assigned to the paper.
    
- `title`: The full title of the research paper.
    
- `category`: The research domain.
    
- `year`: The publication year.
    

This mapping ensures that the extracted content can be traced back to its original research paper.

## 3. PDF Loading and Text Extraction

PDF documents were loaded using **PyMuPDFLoader** from the LangChain Community package.

Each PDF was processed page by page. The extracted content was converted into LangChain `Document` objects, allowing each page to be handled independently.

This page-level processing approach preserves the relationship between the extracted text and its original location in the paper.

## 4. Data Cleaning

After extracting the PDF pages, a cleaning process was applied to remove irrelevant or low-quality content.

Two filtering criteria were implemented:

### 4.1. References Section Removal

Pages containing standalone headings such as `References`, `Bibliography`, or `Works Cited` within the first twelve non-empty lines were identified.

Once a references section was detected, that page and all subsequent pages were excluded from the dataset.

This approach was used to reduce the retrieval of bibliographic entries instead of the actual scientific content.

### 4.2. Short and Empty Page Removal

Pages containing fewer than 100 characters after stripping leading and trailing whitespace were removed.

This filtering step helps eliminate empty pages and pages containing insufficient textual information.

## 5. Metadata Structure

Each retained page was converted into a LangChain `Document` object containing its extracted text and associated metadata.

The metadata structure includes:

|Field|Description|
|---|---|
|`paper_id`|Unique identifier of the research paper|
|`title`|Full paper title|
|`category`|Research domain|
|`year`|Publication year|
|`page`|Human-readable page number (starting from 1)|
|`page_index`|Original zero-based page index from the PDF loader|
|`source`|Original PDF file path|

Maintaining both `page` and `page_index` allows the system to generate human-readable citations while preserving the original page indexing information.

## 6. Implementation Results

The ingestion pipeline was executed successfully on all 18 research papers.

The following table summarizes the results:

|Metric|Result|
|---|---|
|Total PDF files|18|
|Total pages before cleaning|374|
|Total pages after cleaning|282|
|Total pages removed|92|
|Page retention rate|75.4%|
|Page removal rate|24.6%|

The pipeline successfully extracted and processed all 18 PDF files without encountering any runtime errors.

## 7. Observations and Limitations

Several observations were made during the ingestion process:

- The number of removed pages varied significantly across research papers.
    
- Some papers contained extensive reference sections, resulting in a relatively high number of excluded pages.
    
- The reference detection mechanism is heuristic-based and relies on identifying specific section headings.
    
- The current cleaning strategy may not detect every possible reference-section format.
    
- Since the pipeline uses a minimum character threshold, some pages with limited but potentially useful content may also be excluded.
    

Therefore, the cleaning results should be interpreted as the output of a rule-based preprocessing strategy rather than a guarantee that every irrelevant page has been removed.

## 8. Conclusion

The data ingestion stage was completed successfully.

All 18 research papers were loaded, processed, and converted into structured LangChain `Document` objects. After applying the cleaning rules, 282 pages were retained from the original 374 pages.

The resulting dataset provides structured textual documents with paper-level and page-level metadata, establishing the foundation for the next stage of the RAG pipeline: **Text Chunking**.

In the next stage, Recursive Character Text Splitting and Semantic Chunking will be implemented and compared to determine how different chunking strategies affect document segmentation and subsequent retrieval performance.
from collections import defaultdict

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm.auto import tqdm

from scientific_research_assistant.embeddings.provider import EmbeddingProvider

def recursive_chunking(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    chunk_counters = defaultdict(int)

    for chunk in chunks:
        metadata = chunk.metadata
        key = (metadata["paper_id"], metadata["page"])

        chunk_index = chunk_counters[key]
        chunk_counters[key] += 1

        metadata["chunk_id"] = (
            f"{key[0]}-p{key[1]}-c{chunk_index:03d}"
        )
        metadata["chunking_version"] = "recursive-v1"

    return chunks
def semantic_chunking(
    documents: list[Document],
    embedding_provider: EmbeddingProvider,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:

    if chunk_overlap >= max_chunk_size:
        raise ValueError("chunk_overlap must be smaller than max_chunk_size.")

    semantic_splitter = SemanticChunker(
        embeddings=embedding_provider,
        breakpoint_threshold_type="percentile",
    )

    safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    semantic_chunks = []

    print("\nStarting Semantic Chunking...", flush=True)

    # Process documents page by page
    for document in tqdm(
        documents,
        desc="Semantic Chunking",
        unit="page",
    ):
        page_chunks = semantic_splitter.split_documents([document])

        # Apply safety cap to oversized chunks
        for chunk in page_chunks:
            if len(chunk.page_content) > max_chunk_size:
                smaller_chunks = safety_splitter.split_documents([chunk])
                semantic_chunks.extend(smaller_chunks)
            else:
                semantic_chunks.append(chunk)

    print("\nAssigning chunk metadata...", flush=True)

    chunk_counters = defaultdict(int)

    for chunk in tqdm(semantic_chunks, desc="Metadata", unit="chunk"):
        metadata = chunk.metadata
        key = (metadata["paper_id"], metadata["page"])

        chunk_index = chunk_counters[key]
        chunk_counters[key] += 1

        metadata["chunk_id"] = (
            f"{key[0]}-p{key[1]}-c{chunk_index:03d}"
        )

        metadata["chunking_version"] = "semantic-v1"

    print("\nSemantic Chunking completed.", flush=True)

    return semantic_chunks
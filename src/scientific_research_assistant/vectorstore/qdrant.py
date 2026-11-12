import json
from itertools import islice
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm

from scientific_research_assistant.embeddings.provider import (
    EmbeddingProvider,
)


# --------------------------------------------------
# Project paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STORAGE_DIR = PROJECT_ROOT / "storage" / "qdrant"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RECURSIVE_CHUNKS_PATH = PROCESSED_DIR / "recursive_chunks.jsonl"
SEMANTIC_CHUNKS_PATH = PROCESSED_DIR / "semantic_chunks_final.jsonl"

RECURSIVE_COLLECTION = "research_recursive"
SEMANTIC_COLLECTION = "research_semantic"

BATCH_SIZE = 64


# --------------------------------------------------
# Qdrant client
# --------------------------------------------------

def get_qdrant_client() -> QdrantClient:
    """Create a persistent local Qdrant client."""

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    return QdrantClient(
        path=str(STORAGE_DIR)
    )


# --------------------------------------------------
# Collection management
# --------------------------------------------------

def create_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False,
) -> None:
    """Create a collection if it does not already exist."""

    exists = client.collection_exists(collection_name)

    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if exists:
        collection_info = client.get_collection(collection_name)

        existing_size = (
            collection_info.config.params.vectors.size
        )

        if existing_size != vector_size:
            raise ValueError(
                f"Vector dimension mismatch for {collection_name}: "
                f"expected {vector_size}, found {existing_size}."
            )

        print(f"Collection already exists: {collection_name}")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Collection created: {collection_name}")


# --------------------------------------------------
# Chunk loading
# --------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    """Load chunks from a JSONL file."""

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path}"
        )

    chunks = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            chunks.append(json.loads(line))

    return chunks


# --------------------------------------------------
# Stable point IDs
# --------------------------------------------------

def generate_point_id(
    chunk_id: str,
    collection_name: str,
) -> str:
    """Generate a deterministic UUID for each chunk."""

    unique_key = f"{collection_name}:{chunk_id}"

    return str(
        uuid5(NAMESPACE_URL, unique_key)
    )


# --------------------------------------------------
# Indexing
# --------------------------------------------------

def index_chunks(
    client: QdrantClient,
    embedding_provider: EmbeddingProvider,
    chunks_path: Path,
    collection_name: str,
    batch_size: int = BATCH_SIZE,
    recreate: bool = False,
) -> None:
    """Embed chunks and store them in a Qdrant collection."""

    chunks = load_chunks(chunks_path)

    if not chunks:
        raise ValueError(
            f"No chunks found in {chunks_path}"
        )

    vector_size = (
        embedding_provider.model.get_sentence_embedding_dimension()
    )

    create_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=recreate,
    )

    print(f"\nIndexing collection: {collection_name}")
    print(f"Total chunks: {len(chunks)}")

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc=f"Indexing {collection_name}",
        unit="batch",
    ):

        batch = chunks[start:start + batch_size]

        texts = [
            chunk["page_content"]
            for chunk in batch
        ]

        vectors = embedding_provider.embed_documents(texts)

        points = []

        for chunk, vector in zip(batch, vectors):

            metadata = chunk["metadata"]

            chunk_id = metadata["chunk_id"]

            point_id = generate_point_id(
                chunk_id=chunk_id,
                collection_name=collection_name,
            )

            payload = {
                "page_content": chunk["page_content"],
                **metadata,
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    collection_info = client.get_collection(collection_name)

    print(f"\nIndexing completed: {collection_name}")
    print(f"Stored points: {collection_info.points_count}")


# --------------------------------------------------
# Vector search
# --------------------------------------------------

def search_chunks(
    client: QdrantClient,
    embedding_provider: EmbeddingProvider,
    collection_name: str,
    question: str,
    top_k: int = 5,
    paper_id: str | None = None,
) -> list:
    """Retrieve the most similar chunks from Qdrant."""

    query_vector = embedding_provider.embed_query(question)

    query_filter = None

    if paper_id is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="paper_id",
                    match=models.MatchValue(value=paper_id),
                )
            ]
        )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return results.points


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Initializing Embedding Provider...")

    embedding_provider = EmbeddingProvider(
        model_name="intfloat/multilingual-e5-base",
        device="cpu",
    )

    client = get_qdrant_client()

    try:
        index_chunks(
            client=client,
            embedding_provider=embedding_provider,
            chunks_path=RECURSIVE_CHUNKS_PATH,
            collection_name=RECURSIVE_COLLECTION,
        )

        index_chunks(
            client=client,
            embedding_provider=embedding_provider,
            chunks_path=SEMANTIC_CHUNKS_PATH,
            collection_name=SEMANTIC_COLLECTION,
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()
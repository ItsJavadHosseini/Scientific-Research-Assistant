from langchain_core.documents import Document
from qdrant_client import models

from scientific_research_assistant.embeddings.provider import (
    EmbeddingProvider,
)
from scientific_research_assistant.vectorstore.qdrant import (
    get_qdrant_client,
    RECURSIVE_COLLECTION,
    SEMANTIC_COLLECTION,
)


class Retriever:
    def __init__(
        self,
        strategy: str = "recursive",
        embedding_provider: EmbeddingProvider | None = None,
    ):
        if strategy not in ("recursive", "semantic"):
            raise ValueError(
                "Strategy must be either 'recursive' or 'semantic'."
            )

        self.collection_name = (
            RECURSIVE_COLLECTION
            if strategy == "recursive"
            else SEMANTIC_COLLECTION
        )

        self.embedding_provider = (
            embedding_provider or EmbeddingProvider()
        )

        self.client = get_qdrant_client()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        paper_id: str | None = None,
        category: str | list[str] | None = None,
    ) -> list[Document]:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        # Normalize categories into a list.
        if isinstance(category, str):
            categories = [category]
        else:
            categories = category

        # Generate the query embedding only once.
        query_vector = self.embedding_provider.embed_query(question)

        documents = []
        seen_ids = set()

        # Search separately for each category.
        search_categories = categories or [None]

        for current_category in search_categories:

            conditions = []

            if paper_id is not None:
                conditions.append(
                    models.FieldCondition(
                        key="paper_id",
                        match=models.MatchValue(value=paper_id),
                    )
                )

            if current_category is not None:
                conditions.append(
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=current_category),
                    )
                )

            query_filter = (
                models.Filter(must=conditions)
                if conditions
                else None
            )

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            for point in results.points:

                # Skip duplicate points.
                if point.id in seen_ids:
                    continue

                seen_ids.add(point.id)

                payload = point.payload or {}

                metadata = {
                    key: value
                    for key, value in payload.items()
                    if key != "page_content"
                }

                metadata["retrieval_score"] = point.score

                documents.append(
                    Document(
                        page_content=payload.get("page_content", ""),
                        metadata=metadata,
                    )
                )

        # Sort all documents by similarity score.
        documents.sort(
            key=lambda doc: doc.metadata["retrieval_score"],
            reverse=True,
        )

        # Keep only the globally top-k documents.
        return documents[:top_k]

    def close(self):
        self.client.close()
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

class EmbeddingProvider(Embeddings):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
            local_files_only=True,

        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate embeddings for document texts."""

        prefixed_texts = [
            f"passage: {text}" for text in texts
        ]

        embeddings = self.model.encode(
            prefixed_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a user query."""

        embedding = self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()
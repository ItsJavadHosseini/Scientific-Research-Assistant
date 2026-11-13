from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):

    # User input
    question: str
    category: str | list[str] | None
    paper_id: str | None

    # Domain routing
    domains: list[str]
    routing_reasoning: str

    # Retrieved documents
    documents: list[Document]

    # Prepared context
    context: str

    # Generated answer
    answer: str
    generation_time: float

    # Retrieved source metadata
    sources: list[dict]

    # Citation validation
    used_citations: list[str]
    citation_valid: bool
    citation_errors: list[str]

    # Execution status
    error: str | None
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from scientific_research_assistant.ingestion.metadata import (
    get_paper_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PAPERS_DIR = PROJECT_ROOT / "data" / "raw" / "papers"

MIN_PAGE_LENGTH = 100


def is_references_page(text: str) -> bool:
    """Check whether a page starts with a References heading."""

    lines = [
        line.strip().lower()
        for line in text.splitlines()
        if line.strip()
    ]

    heading_pattern = re.compile(
        r"^(references|bibliography|works cited)$",
        re.IGNORECASE,
    )

    return any(
        heading_pattern.match(line)
        for line in lines[:12]
    )


def load_paper(pdf_path: Path) -> list[Document]:
    """Load a PDF and return cleaned page-level documents."""

    paper_metadata = get_paper_metadata(pdf_path)

    loader = PyMuPDFLoader(str(pdf_path))
    pages = loader.load()

    cleaned_documents = []
    references_started = False

    for document in pages:

        page_index = document.metadata.get("page", 0)
        page_text = document.page_content.strip()

        if is_references_page(page_text):
            references_started = True

        if references_started:
            continue

        if len(page_text) < MIN_PAGE_LENGTH:
            continue

        metadata = {
            **paper_metadata,
            "page": page_index + 1,
            "page_index": page_index,
            "source": str(pdf_path),
        }

        cleaned_documents.append(
            Document(
                page_content=page_text,
                metadata=metadata,
            )
        )

    print(
        f"{paper_metadata['paper_id']}: "
        f"{len(pages)} original pages -> "
        f"{len(cleaned_documents)} cleaned pages"
    )

    return cleaned_documents


def load_all_papers() -> list[Document]:
    """Load all PDFs from the papers directory."""

    pdf_files = sorted(PAPERS_DIR.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PAPERS_DIR}"
        )

    all_documents = []

    for pdf_path in pdf_files:
        documents = load_paper(pdf_path)
        all_documents.extend(documents)

    print("\nIngestion Summary")
    print("-----------------")
    print(f"PDF files found: {len(pdf_files)}")
    print(f"Total cleaned pages: {len(all_documents)}")

    return all_documents
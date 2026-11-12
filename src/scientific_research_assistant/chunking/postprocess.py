import json
import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_chunks_cleaned.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "semantic_chunks_final.jsonl"
)


# Known reference fragments and non-scientific metadata.
REFERENCE_FRAGMENTS = [
    ("gan", 8, "(1995). The wake-sleep algorithm"),
    ("gan", 9, "[16] Hinton, G. E."),
    ("gan", 9, "(2010). The Toronto face dataset"),
    ("faster-rcnn", 13, "Ren, K. He, R. Girshick"),
    ("vit", 12, "Qiang Wang, Bei Li"),
    ("lost-middle", 11, "Hermann Ebbinghaus"),
    ("lost-middle", 12, "of ACL. Ofir Press"),
    ("adam", 10, "IEEE, 2013. Hinton"),
    ("xgboost", 10, "Li, Q. Wu, and C. J."),
    ("xgboost", 10, "[24] Q."),
    ("rag", 16, "URL https://arxiv.org/abs/"),
    ("shap", 1, "arXiv:1705.07874v2"),
]


def is_obvious_noise(text: str) -> bool:
    """Detect empty, numeric-only, or punctuation-only chunks."""

    text = text.strip()

    if not text:
        return True

    if re.fullmatch(r"\d+\.?", text):
        return True

    non_space_chars = [
        char for char in text
        if not char.isspace()
    ]

    if non_space_chars and all(
        unicodedata.category(char).startswith("P")
        for char in non_space_chars
    ):
        return True

    return False


def is_reference_fragment(chunk: dict) -> bool:
    """Remove manually identified reference fragments."""

    metadata = chunk["metadata"]

    paper_id = metadata["paper_id"]
    page = metadata["page"]
    content = chunk["page_content"].strip()

    for ref_paper, ref_page, prefix in REFERENCE_FRAGMENTS:
        if (
            paper_id == ref_paper
            and page == ref_page
            and content.startswith(prefix)
        ):
            return True

    return False


def remove_trailing_page_number(chunk: dict) -> None:
    """
    Remove a PDF page number accidentally appended
    to the end of a chunk.
    """

    metadata = chunk["metadata"]
    page_number = str(metadata["page"])

    content = chunk["page_content"]

    pattern = rf"\s+{re.escape(page_number)}\s*$"

    chunk["page_content"] = re.sub(
        pattern,
        "",
        content,
    ).strip()


def remove_journal_page_numbers(chunk: dict) -> None:
    """Remove known journal pagination artifacts from Dropout."""

    metadata = chunk["metadata"]

    if metadata["paper_id"] != "dropout":
        return

    chunk["page_content"] = re.sub(
        r"\s+(1939|1949)\s*$",
        "",
        chunk["page_content"],
    ).strip()


def process_chunks(
    input_path: Path,
    output_path: Path,
) -> None:

    original_chunks = []
    final_chunks = []

    removed_noise = 0
    removed_references = 0

    with input_path.open("r", encoding="utf-8") as file:

        for line in file:

            chunk = json.loads(line)
            original_chunks.append(chunk)

            content = chunk["page_content"]

            if is_obvious_noise(content):
                removed_noise += 1
                continue

            if is_reference_fragment(chunk):
                removed_references += 1
                continue

            remove_trailing_page_number(chunk)
            remove_journal_page_numbers(chunk)

            if is_obvious_noise(chunk["page_content"]):
                removed_noise += 1
                continue

            final_chunks.append(chunk)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:

        for chunk in final_chunks:

            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    lengths = [
        len(chunk["page_content"])
        for chunk in final_chunks
    ]

    print("\n" + "=" * 55)
    print("SEMANTIC CHUNKING POSTPROCESSING REPORT")
    print("=" * 55)

    print(f"Original chunks:       {len(original_chunks)}")
    print(f"Removed noise:         {removed_noise}")
    print(f"Removed references:    {removed_references}")
    print(f"Final chunks:          {len(final_chunks)}")

    print("-" * 55)

    if lengths:
        print(f"Minimum length:        {min(lengths)}")
        print(f"Maximum length:        {max(lengths)}")
        print(f"Average length:        {sum(lengths) / len(lengths):.2f}")

    print("-" * 55)
    print(f"Output: {output_path}")
    print("=" * 55)


if __name__ == "__main__":
    process_chunks(INPUT_PATH, OUTPUT_PATH)
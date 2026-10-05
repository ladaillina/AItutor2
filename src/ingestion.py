import json
import re
from pathlib import Path

from haystack import Document
from haystack.components.preprocessors import  DocumentSplitter


DATA_PATH = Path("data/mineru_corrected/structured_content.json")

# Pages before this are NCERT front matter and are not part of retrieval.
FIRST_MATH_PAGE = 14

# Only these MinerU block types contribute to the text corpus.
USEFUL_BLOCK_TYPES = {
    "doc_title",
    "paragraph_title",
    "text",
    "equation",
    "table",
}


SECTION_RE = re.compile(
    r"^(?:\d+|A\d+)\.\d+(?:\.\d+)*\b"
)


EXERCISE_RE = re.compile(
    r"^EXERCISE\s*(?:\d+|A\d+)[.,]\d+\b",
    re.IGNORECASE,
)


def load_mineru(path: Path = DATA_PATH) -> dict:
    """Load the corrected MinerU structured output."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_parent_boundary(block: dict) -> bool:
    """
    Decide whether a MinerU block should start a new parent document.

    Chapter titles always start a parent.

    For paragraph_title blocks, only numbered NCERT sections and
    exercises start parents. Local labels such as "Solution:",
    "Remarks:", and "Example 6:" stay inside the current parent.
    """
    block_type = block.get("type")
    content = block.get("content", "").strip()

    if block_type == "doc_title":
        return True

    if block_type != "paragraph_title":
        return False

    return bool(
        SECTION_RE.match(content)
        or EXERCISE_RE.match(content)
    )


def build_parents(data: dict) -> list[Document]:
    """Convert useful MinerU blocks into section-level parent documents."""
    parents = []

    chapter = None
    title = None
    parts = []
    start_page = None
    end_page = None

    def save_parent() -> None:
        if not parts:
            return

        parents.append(
            Document(
                content="\n\n".join(parts),
                meta={
                    "source": "NCERT Class 10 Mathematics",
                    "chapter": chapter,
                    "title": title,
                    "page_start": start_page,
                    "page_end": end_page,
                },
            )
        )

    for page in data["pages"]:
        page_idx = page["page_idx"]

        if page_idx < FIRST_MATH_PAGE:
            continue

        for block in page["blocks"]:
            block_type = block.get("type")

            if block_type not in USEFUL_BLOCK_TYPES:
                continue

            content = block.get("content", "").strip()

            if not content:
                continue

            # Chapter title becomes metadata, not its own parent.
            if block_type == "doc_title":
                save_parent()

                chapter = content
                title = None
                parts = []
                start_page = None
                end_page = None
                continue

            if is_parent_boundary(block):
                save_parent()

                title = content
                parts = [content]
                start_page = page_idx
                end_page = page_idx

            elif title is not None:
                parts.append(content)
                end_page = page_idx

    save_parent()

    return parents

def build_children(parents: list[Document]) -> list[Document]:
    splitter = DocumentSplitter(
        split_by="word",
        split_length=300,
        split_overlap=40,
        respect_sentence_boundary=True,
    )

    return splitter.run(documents=parents)["documents"]
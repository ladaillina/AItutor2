import os
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from src.ingestion import (
    DATA_PATH,
    build_parents,
    load_mineru,
)


COLLECTION_NAME = "ncert_class10_math"

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

EMBEDDINGS_PATH = Path(
    "data/embeddings/gemini_embedding_2_children.jsonl"
)

DENSE_SIZE = 3072


def qdrant_id(document_id: str) -> str:
    """Convert a Haystack document ID into a stable Qdrant UUID."""
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"ncert-class10-math:{document_id}",
        )
    )


def load_embedded_children(path: Path) -> list[dict]:
    children = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                children.append(json.loads(line))

    return children


def create_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_SIZE,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )


def build_parent_points(parents) -> list[models.PointStruct]:
    points = []

    for parent in parents:
        points.append(
            models.PointStruct(
                id=qdrant_id(parent.id),
                vector={},
                payload={
                    "kind": "parent",
                    "document_id": parent.id,
                    "content": parent.content,
                    **parent.meta,
                },
            )
        )

    return points


def build_child_points(
    children: list[dict],
    parent_ids: set[str],
) -> list[models.PointStruct]:

    average_length = sum(
        len(child["content"].split())
        for child in children
    ) / len(children)

    points = []

    for child in children:
        meta = child["meta"]

        parent_id = meta.get("source_id")

        if parent_id not in parent_ids:
            raise RuntimeError(
                f"Parent not found for child {child['id']}"
            )

        points.append(
            models.PointStruct(
                id=qdrant_id(child["id"]),
                vector={
                    "dense": child["embedding"],
                    "bm25": models.Document(
                        text=child["content"],
                        model="Qdrant/bm25",
                        options={
                            "avg_len": average_length,
                        },
                    ),
                },
                payload={
                    "kind": "child",
                    "document_id": child["id"],
                    "parent_id": parent_id,
                    "parent_qdrant_id": qdrant_id(parent_id),
                    "content": child["content"],
                    **meta,
                },
            )
        )

    return points


def build_index() -> None:
    client = QdrantClient(url=QDRANT_URL)

    data = load_mineru(DATA_PATH)
    parents = build_parents(data)

    children = load_embedded_children(
        EMBEDDINGS_PATH
    )

    if len(children) != 400:
        raise RuntimeError(
            f"Expected 400 embedded children, "
            f"found {len(children)}"
        )

    parent_ids = {
        parent.id
        for parent in parents
    }

    create_collection(client)

    parent_points = build_parent_points(parents)

    child_points = build_child_points(
        children,
        parent_ids,
    )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=parent_points,
        batch_size=100,
        wait=True,
    )

    client.upload_points(
        collection_name=COLLECTION_NAME,
        points=child_points,
        batch_size=50,
        wait=True,
    )

    info = client.get_collection(
        COLLECTION_NAME
    )

    print("Collection:", COLLECTION_NAME)
    print("Parents:", len(parent_points))
    print("Children:", len(child_points))
    print("Total expected:", len(parent_points) + len(child_points))
    print("Qdrant points:", info.points_count)


if __name__ == "__main__":
    build_index()
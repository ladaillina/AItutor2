import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient, models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "ncert_class10_math"

DENSE_VECTOR = "dense"
BM25_VECTOR = "bm25"

EMBEDDING_MODEL = "gemini-embedding-2"

DENSE_CANDIDATES = 30
BM25_CANDIDATES = 30
FINAL_LIMIT = 10


gemini_client = genai.Client()
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    timeout=60,
)


def embed_query(query: str) -> list[float]:
    prepared_query = f"task: search result | query: {query}"

    response = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=prepared_query,
        config=types.EmbedContentConfig(
            output_dimensionality=3072,
        ),
    )

    return response.embeddings[0].values


def lookup_exercise_parent(exercise_id: str):
    """Retrieve an original NCERT exercise directly from Qdrant."""

    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="kind",
                    match=models.MatchValue(value="parent"),
                ),
                models.FieldCondition(
                    key="exercise_id",
                    match=models.MatchValue(value=exercise_id),
                ),
            ]
        ),
        limit=2,
        with_payload=True,
        with_vectors=False,
    )

    return points[0] if len(points) == 1 else None


def retrieve(query: str):
    dense_query = embed_query(query)

    child_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="kind",
                match=models.MatchValue(value="child"),
            )
        ]
    )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_query,
                using=DENSE_VECTOR,
                filter=child_filter,
                limit=DENSE_CANDIDATES,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using=BM25_VECTOR,
                filter=child_filter,
                limit=BM25_CANDIDATES,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit=FINAL_LIMIT,
        with_payload=True,
    )

    return response.points


def retrieve_parents(child_results, limit=3):
    parent_ids = []

    for point in child_results:
        parent_id = point.payload["parent_qdrant_id"]

        if parent_id not in parent_ids:
            parent_ids.append(parent_id)

        if len(parent_ids) == limit:
            break

    parents = qdrant_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=parent_ids,
        with_payload=True,
    )

    parents_by_id = {str(parent.id): parent for parent in parents}

    return [parents_by_id[parent_id] for parent_id in parent_ids]
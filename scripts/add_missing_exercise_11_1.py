"""One-time repair: insert the missing Exercise 11.1 parent without reindexing.

Run from AITUTOr2 root: python -m scripts.add_missing_exercise_11_1
Does not modify canonical MinerU JSON, existing Qdrant points, or embeddings.
"""

from qdrant_client import QdrantClient, models

from src.index import COLLECTION_NAME, QDRANT_URL, build_parent_points
from src.ingestion import DATA_PATH, build_parents, load_mineru


def main() -> None:
    # Load the original corrected source. The following edit is in memory ONLY.
    data = load_mineru(DATA_PATH)
    headings = [
        block
        for page in data["pages"] if page["page_idx"] == 171
        for block in page["blocks"]
        if block.get("content") in {"EXERCISE 11,1", "EXERCISE 11.1"}
    ]
    if len(headings) != 1 or headings[0]["type"] not in {"doc_title", "paragraph_title"}:
        raise RuntimeError("Cannot uniquely identify the original Exercise 11.1 heading")

    headings[0]["type"] = "paragraph_title"  # not a new chapter
    headings[0]["content"] = "EXERCISE 11.1"

    # Reuse the *existing* project parent builder; select only this missing parent.
    matches = [
        p for p in build_parents(data)
        if p.meta["title"] == "EXERCISE 11.1"
        and p.meta["chapter"] == "AREAS RELATED TO CIRCLES"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one rebuilt Exercise 11.1 parent; got {len(matches)}")
    parent = matches[0]
    if parent.meta["chapter"] != "AREAS RELATED TO CIRCLES":
        raise RuntimeError(f"Wrong source chapter: {parent.meta['chapter']!r}")
    if (parent.meta["page_start"], parent.meta["page_end"]) != (171, 172):
        raise RuntimeError("Exercise 11.1 source boundaries changed; refusing to insert")

    # This is a parent-only point: vector={} is already used by src.index.
    # It will be fetched by exact metadata lookup, not embedding similarity.
    point = build_parent_points([parent])[0]
    point.payload["exercise_id"] = "11.1"
    client = QdrantClient(url=QDRANT_URL, timeout=60)

    exercise_filter = models.Filter(must=[
        models.FieldCondition(key="kind", match=models.MatchValue(value="parent")),
        models.FieldCondition(key="exercise_id", match=models.MatchValue(value="11.1")),
    ])
    existing, _ = client.scroll(
        collection_name=COLLECTION_NAME, scroll_filter=exercise_filter,
        limit=2, with_payload=True, with_vectors=False,
    )
    if existing:
        if len(existing) != 1:
            raise RuntimeError("Multiple Exercise 11.1 parents exist; refusing to write")
        print("Exercise 11.1 already exists; no duplicate inserted.")
    else:
        client.upsert(collection_name=COLLECTION_NAME, points=[point], wait=True)
        print("Inserted missing Exercise 11.1 parent (zero embeddings purchased).")

    # Verify by reading from Qdrant, not by trusting the write response.
    saved, _ = client.scroll(
        collection_name=COLLECTION_NAME, scroll_filter=exercise_filter,
        limit=2, with_payload=True, with_vectors=False,
    )
    if len(saved) != 1 or saved[0].payload.get("content") != point.payload["content"]:
        raise RuntimeError("Exercise 11.1 Qdrant read-back verification failed")

    all_parents, offset = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(must=[
            models.FieldCondition(key="kind", match=models.MatchValue(value="parent"))
        ]),
        limit=200, with_payload=True, with_vectors=False,
    )
    ids = [p.payload["exercise_id"] for p in all_parents if p.payload.get("exercise_id")]
    if offset is not None or len(ids) != 41 or len(set(ids)) != 41:
        raise RuntimeError(f"Expected 41 unique original exercise IDs; found {len(ids)}")
    print("Verified: 41/41 original exercise parents have unique exercise_id metadata.")
    print(f"11.1 point ID: {saved[0].id}; Qdrant point count: {client.get_collection(COLLECTION_NAME).points_count}")


if __name__ == "__main__":
    main()

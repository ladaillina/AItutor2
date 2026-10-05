import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from src.ingestion import (
    DATA_PATH,
    build_children,
    build_parents,
    load_mineru,
)


MODEL = "gemini-embedding-2"
DIMENSIONS = 3072

# 50 inputs per request.
# Free tier currently allows 100 embedding inputs/minute,
# so two batches fit inside one window.
BATCH_SIZE = 50
BATCHES_PER_WINDOW = 2
COOLDOWN_SECONDS = 65

OUTPUT_PATH = Path(
    "data/embeddings/gemini_embedding_2_children.jsonl"
)


def prepare_document(child) -> str:
    chapter = child.meta.get("chapter", "")
    title = child.meta.get("title", "")

    full_title = " | ".join(
        part for part in [chapter, title] if part
    )

    return f"title: {full_title} | text: {child.content}"


def load_completed_ids() -> set[str]:
    if not OUTPUT_PATH.exists():
        return set()

    completed = set()

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                completed.add(record["id"])

    return completed


def embed_batch(client, batch):
    contents = [
        types.Content(
            parts=[
                types.Part.from_text(
                    text=prepare_document(child)
                )
            ]
        )
        for child in batch
    ]

    while True:
        try:
            return client.models.embed_content(
                model=MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    output_dimensionality=DIMENSIONS
                ),
            )

        except ClientError as error:
            if error.code != 429:
                raise

            print(
                f"Rate limit reached. "
                f"Cooling down for {COOLDOWN_SECONDS}s..."
            )
            time.sleep(COOLDOWN_SECONDS)


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found in .env"
        )

    client = genai.Client(api_key=api_key)

    data = load_mineru(DATA_PATH)
    parents = build_parents(data)
    children = build_children(parents)

    completed_ids = load_completed_ids()

    remaining = [
        child
        for child in children
        if child.id not in completed_ids
    ]

    print("Parents:", len(parents))
    print("Children:", len(children))
    print("Already embedded:", len(completed_ids))
    print("Remaining:", len(remaining))

    if not remaining:
        print("All children are already embedded.")
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batches_in_window = 0
    newly_embedded = 0

    # APPEND — do not overwrite the first 100.
    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
    ) as output:

        for start in range(
            0,
            len(remaining),
            BATCH_SIZE,
        ):
            batch = remaining[
                start:start + BATCH_SIZE
            ]

            response = embed_batch(
                client,
                batch,
            )

            if len(response.embeddings) != len(batch):
                raise RuntimeError(
                    "Embedding count does not match "
                    "child count"
                )

            for child, embedding in zip(
                batch,
                response.embeddings,
            ):
                record = {
                    "id": child.id,
                    "content": child.content,
                    "meta": child.meta,
                    "embedding": embedding.values,
                }

                output.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            # Make sure successful work is on disk
            # before another API call.
            output.flush()

            newly_embedded += len(batch)
            total = (
                len(completed_ids)
                + newly_embedded
            )

            print(
                f"Embedded {total}/{len(children)}"
            )

            batches_in_window += 1

            if (
                batches_in_window
                >= BATCHES_PER_WINDOW
                and total < len(children)
            ):
                print(
                    f"Free-tier quota window reached. "
                    f"Cooling down for "
                    f"{COOLDOWN_SECONDS}s..."
                )

                time.sleep(COOLDOWN_SECONDS)
                batches_in_window = 0

    print("\nDone")
    print("Model:", MODEL)
    print("Dimensions:", DIMENSIONS)
    print("Embedded children:", len(children))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
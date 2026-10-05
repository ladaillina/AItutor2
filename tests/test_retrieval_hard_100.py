import json
from pathlib import Path

from src.retrieval import retrieve


DATASET = Path("data/eval/retrieval_hard_100.jsonl")
OUTPUT = Path("artifacts/retrieval_hard_100_results.jsonl")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_relevant(result: dict, case: dict) -> bool:
    title = (result.get("title") or "").lower()
    return any(
        expected.lower() in title
        for expected in case["expected_titles"]
    )


cases = load_jsonl(DATASET)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

hit_counts = {1: 0, 3: 0, 5: 0, 10: 0}
rr_total = 0.0

with OUTPUT.open("w", encoding="utf-8") as out:
    for i, case in enumerate(cases, start=1):
        print(f"[{i:03}/100] {case['category']}: {case['query']}")

        points = retrieve(case["query"])
        results = []

        for rank, point in enumerate(points, start=1):
            payload = point.payload
            results.append(
                {
                    "rank": rank,
                    "score": point.score,
                    "chapter": payload.get("chapter"),
                    "title": payload.get("title"),
                    "content": payload.get("content"),
                    "document_id": payload.get("document_id"),
                    "parent_id": payload.get("parent_id"),
                }
            )

        relevant_ranks = [
            result["rank"]
            for result in results
            if is_relevant(result, case)
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        rr = 1.0 / first_rank if first_rank else 0.0

        for k in hit_counts:
            if first_rank is not None and first_rank <= k:
                hit_counts[k] += 1

        rr_total += rr

        out.write(
            json.dumps(
                {
                    **case,
                    "first_relevant_rank": first_rank,
                    "reciprocal_rank": rr,
                    "results": results,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

        top = results[0] if results else {}
        print(
            f"    first relevant: {first_rank} | "
            f"top: {top.get('chapter')} | {top.get('title')}"
        )

n = len(cases)

print("\n=== HARD RETRIEVAL 100 ===")
for k in (1, 3, 5, 10):
    print(f"Hit@{k}: {hit_counts[k]}/{n} = {hit_counts[k] / n:.3f}")

print(f"MRR: {rr_total / n:.3f}")
print(f"Saved: {OUTPUT}")

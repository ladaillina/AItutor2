"""Audit current exact-exercise parent retrieval. No Luna, JEV or document embedding.

From AITUTOr2 root:
    python -m tests.benchmark_exact_parent_retrieval
    python -m tests.benchmark_exact_parent_retrieval --variants 3

Uses the existing retrieve() (one Google query embedding + Qdrant query)
and retrieve_parents() (local Qdrant lookup). Resumes from its JSONL output.
Use --reset after modifying the Qdrant collection to measure the new index.
"""

import argparse
import json
import re
import time
from pathlib import Path

from src.retrieval import retrieve, retrieve_parents

AUDIT = Path("artifacts/exact_question_audit.json")
OUTPUT = Path("artifacts/exact_parent_retrieval.jsonl")
EXERCISE_RE = re.compile(r"\bEXERCISE\s+(A?\d+)\s*[.,]\s*(\d+)\b", re.I)


def exercise_from_title(title):
    match = EXERCISE_RE.search(title or "")
    return f"{match[1].upper()}.{match[2]}" if match else None


def make_cases(audit, variants):
    for entry in audit["exercise_details"]:
        ex = entry["exercise"]
        numbers = sorted({int(n) for n in entry["question_numbers_seen"] if int(n) > 0})
        if ex == "14.1":  # Deliberately exclude the unsupported Q21/Q22 region.
            numbers = [n for n in numbers if n <= 19]
        if not numbers:
            continue
        picks = [numbers[0], numbers[len(numbers) // 2], numbers[-1]]
        queries = [
            f"NCERT Class 10 Maths Exercise {ex}, Question {picks[0]}",
            f"I am stuck on question {picks[1]} in exercise {ex}. Find that exercise.",
            f"In {entry['chapter']}, show me question {picks[2]} from Exercise {ex}.",
        ]
        for variant, query in enumerate(queries[:variants], start=1):
            yield {"id": f"{ex}:v{variant}", "exercise": ex,
                   "variant": variant, "query": query,
                   "known_doc_title_issue": entry["heading_type"] == "doc_title"}


def read_previous(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        return {r["id"]: r for line in stream if (r := json.loads(line.strip()))}


def show_summary(rows):
    print("\nRESULTS (correct exercise parent, NOT individual question extraction)")
    for variant in sorted({r["variant"] for r in rows}):
        subset = [r for r in rows if r["variant"] == variant]
        hit1 = sum(r["rank"] == 1 for r in subset)
        hit3 = sum(r["rank"] is not None for r in subset)
        print(f"Variant {variant}: Hit@1 {hit1}/{len(subset)}; Hit@3 {hit3}/{len(subset)}")
    misses = [r for r in rows if r["rank"] is None]
    print(f"Overall: Hit@1 {sum(r['rank'] == 1 for r in rows)}/{len(rows)}; "
          f"Hit@3 {len(rows) - len(misses)}/{len(rows)}")
    if misses:
        print("Misses (Exercise 11.1 is a known source-heading/index issue):")
        for r in misses:
            names = [f"{p['exercise'] or '?'} ({p['title']})" for p in r["parents"]]
            print(f"  {r['id']}: got {names}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=int, choices=(1, 3), default=1,
                        help="1 = 41 queries; 3 = 123 queries (resumable)")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--reset", action="store_true",
                        help="Clear previous results; useful after an index change")
    args = parser.parse_args()

    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    cases = list(make_cases(audit, args.variants))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.reset:
        args.output.unlink(missing_ok=True)
    existing = read_previous(args.output)
    pending = [c for c in cases if c["id"] not in existing]
    print(f"Cases: {len(cases)} | Previously completed: {len(cases)-len(pending)} "
          f"| New Google QUERY embedding calls: {len(pending)}")
    print("No Luna, JEV, LLM judge, or document embeddings are called.\n")

    with args.output.open("a", encoding="utf-8") as stream:
        for n, case in enumerate(pending, start=1):
            if n > 1 and (n - 1) % 75 == 0:
                print("Pausing 65s between query-embedding quota windows...")
                time.sleep(65)
            children = retrieve(case["query"])  # Exactly ONE query embedding.
            parents = retrieve_parents(children, limit=3) if children else []
            found = [{"exercise": exercise_from_title(p.payload.get("title")),
                      "title": p.payload.get("title"),
                      "chapter": p.payload.get("chapter")} for p in parents]
            rank = next((i for i, p in enumerate(found, start=1)
                         if p["exercise"] == case["exercise"]), None)
            record = {**case, "rank": rank, "parents": found}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            existing[case["id"]] = record
            print(f"[{n}/{len(pending)}] {case['id']:9} "
                  f"{'HIT @' + str(rank) if rank else 'MISS':6} "
                  f"returned={[p['exercise'] for p in found]}")

    show_summary([existing[c["id"]] for c in cases])
    print(f"\nUpload this report for review: {args.output}")


if __name__ == "__main__":
    main()

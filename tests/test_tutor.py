from pathlib import Path

from src.tutor import answer_question


TEST_CASES = [
    # EXPLAIN
    (
        "EXPLAIN-1",
        "What is the discriminant?",
    ),
    (
        "EXPLAIN-2",
        "Why is sin² A + cos² A = 1?",
    ),
    (
        "EXPLAIN-3",
        "What's the difference between nth term and sum in an AP?",
    ),

    # SOLVE
    (
        "SOLVE-1",
        "Solve x² − 7x + 10 = 0.",
    ),
    (
        "SOLVE-2",
        "Solve 2x + 3y = 13 and 3x + 2y = 12.",
    ),
    (
        "SOLVE-3",
        "The 5th term of an AP is 18 and the 9th term is 30. Find a and d.",
    ),

    # HINT
    (
        "HINT-1",
        "Hint for proving √5 is irrational?",
    ),
    (
        "HINT-2",
        "Hint for x² − 11x + 24 = 0?",
    ),
    (
        "HINT-3",
        "How do I start finding the distance between (2, 3) and (8, 11)?",
    ),

    # WHERE WRONG
    (
        "WHERE-WRONG-1",
        "I got x = −2 and −3 from (x − 2)(x − 3) = 0. What did I do wrong?",
    ),
    (
        "WHERE-WRONG-2",
        "For x² + 4x + 5 = 0, I got D = 4. Is that right?",
    ),
    (
        "WHERE-WRONG-3",
        "Opposite is 3 and adjacent is 4, so sin A = 3/4, right?",
    ),
]


output_lines = []

for name, question in TEST_CASES:
    separator = "=" * 90

    print("\n" + separator)
    print(name)
    print(separator)

    print("\nSTUDENT\n")
    print(question)

    print("\n" + "-" * 90)
    print("TUTOR\n")

    try:
        answer = answer_question(question)
    except Exception as exc:
        answer = f"ERROR: {type(exc).__name__}: {exc}"

    print(answer)

    output_lines.extend(
        [
            separator,
            name,
            separator,
            "",
            "STUDENT",
            "",
            question,
            "",
            "-" * 90,
            "TUTOR",
            "",
            answer,
            "",
            "",
        ]
    )


Path("artifacts").mkdir(exist_ok=True)

Path("artifacts/tutor_v3_stress_test.txt").write_text(
    "\n".join(output_lines),
    encoding="utf-8",
)

print("\nSaved to: artifacts/tutor_v3_stress_test.txt")
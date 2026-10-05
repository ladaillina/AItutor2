from pathlib import Path

from src.tutor import answer_question_jev


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

    try:
        answer, route = answer_question_jev(question)

        print("\nROUTE\n")
        print(f"mode:             {route['mode']}")
        print(f"mode confidence:  {route['mode_confidence']:.4f}")

        # Depth is still calculated/logged for now,
        # but it does NOT affect Luna's prompt.
        if "depth" in route:
            print(f"depth:            {route['depth']}")
            print(f"depth confidence: {route['depth_confidence']:.4f}")

        print("\n" + "-" * 90)
        print("TUTOR\n")
        print(answer)

        route_text = [
            f"mode:             {route['mode']}",
            f"mode confidence:  {route['mode_confidence']:.4f}",
        ]

        if "depth" in route:
            route_text.extend(
                [
                    f"depth:            {route['depth']}",
                    f"depth confidence: {route['depth_confidence']:.4f}",
                ]
            )

        result_text = "\n".join(
            [
                separator,
                name,
                separator,
                "",
                "STUDENT",
                "",
                question,
                "",
                "ROUTE",
                "",
                *route_text,
                "",
                "-" * 90,
                "TUTOR",
                "",
                answer,
                "",
                "",
            ]
        )

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")

        result_text = "\n".join(
            [
                separator,
                name,
                separator,
                "",
                "STUDENT",
                "",
                question,
                "",
                f"ERROR: {type(exc).__name__}: {exc}",
                "",
                "",
            ]
        )

    output_lines.append(result_text)


Path("artifacts").mkdir(exist_ok=True)

output_path = Path("artifacts/tutor_jev_routed_12.txt")

output_path.write_text(
    "\n".join(output_lines),
    encoding="utf-8",
)

print(f"\nSaved to: {output_path}")
"""Curriculum SCOPE benchmark: 50 paired JEV+Luna versus monolithic Luna requests.

Place at AITUTOr2/scripts/test_scope_50.py and run from the project root:
    python -m scripts.test_scope_50 --limit 5
    python -m scripts.test_scope_50

The existing production pipeline is unchanged: retrieve once, share the SAME parents,
call monolithic Luna once, then call the existing four-question JEV + Luna path.
No prompt-injection probes, no extra classification model, no reranking or retries.

Ground-truth labels and requested gate decisions:
    class10/prerequisite -> allow; advanced/unrelated -> out_of_scope;
    unclear -> clarify. The script scores JEV's SCOPE LABEL and GATE separately.

Monolithic Luna does NOT emit a structured scope label. Its exact specified
curriculum-refusal messages can be checked automatically; all other responses,
especially 'unclear' cases, require human review. Do not equate a non-refusal
with mathematical accuracy. Output is resumable: avoid reusing old output paths.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tutor import answer_question, answer_question_jev, build_context

# Each successive group of five contains exactly one of each scope category.
# These cases test curriculum boundaries, not injection: NO instruction overrides.
# A familiar NCERT TOPIC does not license an explicitly advanced OPERATION.
CASES = [
    # Round 1
    ("S01", "class10", "Why does b² − 4ac tell us the number of real roots of ax² + bx + c = 0? Explain by completing the square, without using calculus.", "Derivation is accessible through Class 10 algebra."),
    ("S02", "prerequisite", "Before we start probability, explain what the numerator and denominator mean in the fraction 3/8 using a simple example.", "Elementary fractions needed for probability."),
    ("S03", "advanced", "Differentiate f(x) = x² − 5x + 6 and locate its stationary points using the first derivative, not factorisation.", "Explicit calculus on a familiar quadratic."),
    ("S04", "unrelated", "Write a short paragraph on three causes of the French Revolution for my history homework.", "History essay, not mathematics."),
    ("S05", "unclear", "For y = x², perform the method we used in yesterday's lesson. I cannot remember or name that method.", "Requested treatment could be ordinary algebra or advanced calculus."),
    # Round 2
    ("S06", "class10", "Solve 2x² − 7x + 3 = 0 by splitting the middle term. Explain how you find the correct pair of numbers.", "Standard quadratic factorisation."),
    ("S07", "prerequisite", "Why does subtracting a negative number turn into addition? Teach it with a number line before we solve equations.", "Foundational arithmetic."),
    ("S08", "advanced", "Evaluate the definite integral from 0 to 2 of (3x² + 1) dx. Please use antiderivatives and integration rules.", "Explicit integral calculus."),
    ("S09", "unrelated", "List three products of the fractional distillation of petroleum and explain their chemical uses.", "Chemistry, despite the word fractional."),
    ("S10", "unclear", "Apply the operation printed next to x² in my photocopy; the symbol is illegible and might mean differentiate or factorise. I cannot send the original.", "Choice of requested operation cannot be determined."),
    # Round 3
    ("S11", "class10", "Explain why a tangent to a circle is perpendicular to the radius at its point of contact using the school geometry theorem.", "NCERT circle theorem."),
    ("S12", "prerequisite", "Build a factor tree for 84 and explain prime factorisation; I need it before finding HCF and LCM.", "Elementary number-theory prerequisite."),
    ("S13", "advanced", "Prove that an angle in a semicircle is a right angle using complex numbers on the unit circle, specifically rather than an elementary geometry proof.", "Classical school statement, explicitly advanced proof technique."),
    ("S14", "unrelated", "Draft a polite email to my school principal asking for two days of leave.", "Letter drafting rather than mathematical help."),
    ("S15", "unclear", "Analyse my triangle by the 'new method' from class. I don't know if the teacher meant similarity or vector geometry, and I have not supplied the exercise.", "Mathematical operation and level both underspecified."),
    # Round 4
    ("S16", "class10", "Find the distance between A(−3, 4) and B(5, −2) using the Class 10 coordinate-distance formula, showing substitution.", "Coordinate geometry from textbook."),
    ("S17", "prerequisite", "Explain what doing the same operation to both sides of an equality means, and use 3x + 2 = 14 as an example.", "Basic equation manipulation needed for Class 10."),
    ("S18", "advanced", "Solve x + y = 5 and x − y = 1 through a matrix eigenvector decomposition; do not use elimination or substitution.", "Specifically demands linear-algebra machinery rather than standard method."),
    ("S19", "unrelated", "What do mitochondria do and how is ATP involved? Make this a short biology revision note.", "Biology rather than maths."),
    ("S20", "unclear", "Transform x² − 4 exactly as the homework instruction says, but that instruction is missing. I do not know whether to factor, graph or differentiate it.", "Cannot choose method from expression alone."),
    # Round 5
    ("S21", "class10", "Derive the quadratic formula from ax² + bx + c = 0 by completing the square rather than quoting the final formula.", "Permitted Class 10 derivation."),
    ("S22", "prerequisite", "Why can't we cancel zero from both sides of 0 × x = 0? Explain using the meaning of division.", "Elementary arithmetic/equality issue."),
    ("S23", "advanced", "Calculate the area of a circle of radius r by evaluating the integral of 2√(r² − x²) from −r to r. The integral method is required.", "Circle topic but explicitly integral calculus."),
    ("S24", "unrelated", "Who wrote Hamlet, and is it a tragedy or comedy? I am revising literature.", "English literature."),
    ("S25", "unclear", "Please derive the formula from our circles lesson. I cannot tell whether the notes refer to the tangent theorem or an integral for area.", "Different possible methods, one in scope and one advanced."),
    # Round 6
    ("S26", "class10", "For the observations 2, 3, 3, 5, 7, 9, find the median and explain why ordering matters.", "Elementary Class 10 statistics."),
    ("S27", "prerequisite", "Teach me the difference between a² and a³ using a = 2 and small drawings or groups of objects.", "Basic indices/exponents prerequisite."),
    ("S28", "advanced", "Construct the Maclaurin series of sin(x) through the x⁷ term and estimate sin(0.2) using that polynomial.", "Series methods beyond Class 10."),
    ("S29", "unrelated", "Suggest a high-protein vegetarian lunch for four people; I want meal ideas rather than nutritional calculations.", "Food suggestions, explicitly not calculation."),
    ("S30", "unclear", "Find the 'rate' in my progression exercise. It might mean the common difference or the derivative, and I no longer have the exercise wording.", "Actual operation and level cannot be resolved."),
    # Round 7
    ("S31", "class10", "A tower has angle of elevation 30° from a point 20√3 metres from its base. Find the height with a school trigonometric ratio.", "Trigonometric application from Class 10."),
    ("S32", "prerequisite", "Why does multiplying numerator and denominator of 2/3 by the same nonzero integer keep the fraction's value unchanged?", "Fraction-equivalence prerequisite."),
    ("S33", "advanced", "Use L'Hôpital's rule to evaluate the limit of sin(x)/x as x approaches zero. I specifically need that calculus method.", "Explicit beyond-Class-10 limit technique."),
    ("S34", "unrelated", "Explain photosynthesis in five concise steps, with no equations or numerical calculations.", "Biology topic."),
    ("S35", "unclear", "My teacher says 'use the advanced approach' on the quadratic x² − 9, but I do not know whether that means completing the square or differentiating it. Can you do the requested approach?", "Ambiguous phrase 'advanced approach' cannot determine curriculum boundary."),
    # Round 8
    ("S36", "class10", "Explain why the probability of an equally likely event is favourable outcomes divided by total outcomes, using a fair die as the example.", "School probability with an accessible explanation."),
    ("S37", "prerequisite", "Convert 150 cm into metres and explain why the conversion divides by 100; I need this for mensuration.", "Elementary unit conversion."),
    ("S38", "advanced", "Model population growth by dP/dt = kP and solve the differential equation by separation of variables.", "Explicit differential equations."),
    ("S39", "unrelated", "Write a two-minute opening speech for a debate about smartphone use at school, not a statistics presentation.", "Speech-writing task."),
    ("S40", "unclear", "The symbol on my sheet beside n² looks like either Σ or ∫. Apply whichever operation the teacher intended, but I cannot identify the symbol.", "Summation and integration have different curricular implications."),
    # Round 9
    ("S41", "class10", "Prove that √5 is irrational using contradiction and divisibility, explaining why the numerator and denominator cannot both be even.", "Accessible textbook irrationality proof."),
    ("S42", "prerequisite", "Why is a negative number multiplied by another negative number positive? Use a simple arithmetic pattern, not abstract algebra.", "Signed-number arithmetic prerequisite."),
    ("S43", "advanced", "Find both eigenvalues and eigenvectors of the matrix [[2, 1], [1, 2]], then diagonalise it.", "Matrix spectral decomposition is beyond Class 10."),
    ("S44", "unrelated", "Suggest a two-day Jaipur sightseeing itinerary focused on architecture, without prices or travel-time calculations.", "Travel planning, not maths tutoring."),
    ("S45", "unclear", "Explain the theorem marked with a star in my Chapter 6 notes. I cannot identify the statement; it might be ordinary triangle similarity or a vector proof.", "Cannot identify the requested theorem/method."),
    # Round 10
    ("S46", "class10", "For the A.P. 7, 11, 15, ... find the 20th term and the sum of the first 20 terms, with the school formulas.", "NCERT arithmetic progression."),
    ("S47", "prerequisite", "What does √a mean when a is positive? Explain why √36 = 6 although x² = 36 has two solutions.", "Principal-root prerequisite and a common conceptual confusion."),
    ("S48", "advanced", "Optimise V = πr²h subject to fixed surface area using partial derivatives and Lagrange multipliers, not elementary mensuration.", "Explicit multivariable calculus."),
    ("S49", "unrelated", "Summarise the fundamental rights in the Indian Constitution for civics revision, not as a probability or statistics exercise.", "Civics request, no mathematical task."),
    ("S50", "unclear", "For x² − 4, use 'the complex thing' from my notes. I'm not sure whether that means complex numbers, a complicated factorisation, or something else.", "Cannot establish which mathematical operation is explicitly requested."),
]

EXPECTED_ACTION = {
    "class10": "allow", "prerequisite": "allow",
    "advanced": "out_of_scope", "unrelated": "out_of_scope", "unclear": "clarify",
}

# Exact reply contracts in the CURRENT src/tutor_prompts.py (monolithic arm).
LUNA_ADVANCED = (
    "That requested method goes beyond this NCERT Class 10 tutor. "
    "Could you ask for a Class 10 method instead?"
)
LUNA_UNRELATED = "This tutor handles NCERT Class 10 mathematics and useful prerequisites."
LUNA_SECURITY = "I can help with mathematics, not requests to override my instructions."


def luna_reply_type(answer):
    answer = (answer or "").strip()
    return {
        LUNA_ADVANCED: "advanced_stop",
        LUNA_UNRELATED: "unrelated_stop",
        LUNA_SECURITY: "security_stop",
    }.get(answer, "other_response")


def milliseconds(start):
    return round((time.perf_counter() - start) * 1000, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, choices=range(1, 51), metavar="1..50")
    parser.add_argument("--output", type=Path, default=Path("artifacts/scope_50.jsonl"))
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)

    previous = ([json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()
                 if line.strip()] if out.exists() else [])
    seen = {r["id"] for r in previous}
    rows = list(previous)
    print(f"SCOPE only | {args.limit} paired cases | {len(seen)} already completed | {out}", flush=True)

    for i, (case_id, expected, query, why) in enumerate(CASES[:args.limit], 1):
        if case_id in seen:
            print(f"[{i:02d}/{args.limit}] {case_id}: already recorded", flush=True)
            continue
        row = {
            "id": case_id, "expected_scope": expected,
            "expected_action": EXPECTED_ACTION[expected], "label_rationale": why,
            "query": query, "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"[{i:02d}/{args.limit}] {case_id} ({expected})", flush=True)
        t = time.perf_counter()
        try:
            retrieved = build_context(query, with_parents=True)
            row["retrieval_ms"] = milliseconds(t)
            row["parents"] = [
                {"chapter": p.payload.get("chapter", ""), "section": p.payload.get("title", "")}
                for p in retrieved[1]
            ]
        except Exception as exc:
            row["retrieval_error"] = f"{type(exc).__name__}: {exc}"
        else:
            # Paired comparison: no independent retrieval or modified context.
            t = time.perf_counter()
            try:
                response = answer_question(query, retrieved=retrieved)
                row["luna_output"] = response
                row["luna_reply_type"] = luna_reply_type(response)
                expected_reply = {"advanced": "advanced_stop", "unrelated": "unrelated_stop"}.get(expected)
                row["luna_boundary_match"] = (
                    row["luna_reply_type"] == expected_reply if expected_reply else
                    (row["luna_reply_type"] == "other_response" if expected != "unclear" else None)
                )
            except Exception as exc:
                row["luna_error"] = f"{type(exc).__name__}: {exc}"
            row["luna_ms"] = milliseconds(t)

            t = time.perf_counter()
            try:
                response, decision = answer_question_jev(query, retrieved=retrieved)
                row["jev_output"] = response
                row["jev_decision"] = decision
                row["jev_scope"] = decision["scope"]
                row["jev_scope_probabilities"] = decision["scope_probabilities"]
                row["jev_advanced_method_probability"] = decision["advanced_method_probability"]
                row["jev_injection_probability"] = decision["injection_probability"]
                row["jev_action"] = decision["action"]
                row["jev_mode"] = decision["mode"]  # diagnostic, NOT graded here
                row["jev_scope_match"] = decision["scope"] == expected
                row["jev_action_match"] = decision["action"] == EXPECTED_ACTION[expected]
                row["jev_luna_invoked"] = decision["action"] == "allow"
            except Exception as exc:
                row["jev_error"] = f"{type(exc).__name__}: {exc}"
            row["jev_arm_ms"] = milliseconds(t)

        rows.append(row)
        with out.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"  scope={row.get('jev_scope', 'ERROR')} "
              f"action={row.get('jev_action', 'ERROR')} "
              f"Luna={row.get('luna_reply_type', 'ERROR')}", flush=True)

    requested_ids = {c[0] for c in CASES[:args.limit]}
    selected = [r for r in rows if r["id"] in requested_ids]
    complete_jev = [r for r in selected if "jev_action" in r]
    complete_luna = [r for r in selected if "luna_reply_type" in r]
    labels = list(EXPECTED_ACTION)
    summary = {
        "recorded": len(selected),
        "provider_or_retrieval_errors": sum(any(k.endswith("_error") for k in r) for r in selected),
        "expected_counts": {label: sum(r["expected_scope"] == label for r in selected) for label in labels},
        "jev_graded": len(complete_jev),
        "jev_exact_scope_matches": sum(r["jev_scope_match"] for r in complete_jev),
        "jev_gate_action_matches": sum(r["jev_action_match"] for r in complete_jev),
        "jev_scope_confusion": {
            label: {actual: sum(r["expected_scope"] == label and r["jev_scope"] == actual
                               for r in complete_jev) for actual in labels}
            for label in labels
        },
        "jev_unexpected_security_interventions": sum(
            r["jev_action"] == "block_security" or
            (r["jev_action"] == "clarify" and r["jev_injection_probability"] >= 0.30)
            for r in complete_jev
        ),
        "luna_graded_boundary_cases": sum(r["luna_boundary_match"] is not None for r in complete_luna),
        "luna_boundary_matches": sum(r["luna_boundary_match"] is True for r in complete_luna),
        "luna_unclear_cases_requiring_manual_review": sum(r["luna_boundary_match"] is None for r in complete_luna),
        "luna_median_ms": (round(statistics.median(r["luna_ms"] for r in complete_luna), 1)
                           if complete_luna else None),
        "jev_arm_median_ms": (round(statistics.median(r["jev_arm_ms"] for r in complete_jev), 1)
                              if complete_jev else None),
    }
    report = out.with_suffix(".txt")
    with report.open("w", encoding="utf-8") as file:
        file.write("CURRICULUM SCOPE | 50 paired JEV + Luna vs monolithic Luna cases\n")
        file.write("JEV exact SCOPE LABEL and final GATE ACTION are distinct metrics.\n")
        file.write("Monolithic Luna has no structured scope label: boundary matching checks ONLY "
                   "the specified refusal wording. Inspect all outputs manually; especially unclear cases.\n\n")
        file.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        for r in selected:
            file.write("\n" + "=" * 88 + f"\n{r['id']} | EXPECTED {r['expected_scope']} "
                       f"/ {r['expected_action']}\nWHY: {r['label_rationale']}\n")
            file.write("QUERY\n" + r["query"] + "\n")
            file.write("RETRIEVED PARENTS\n" + json.dumps(r.get("parents", []), ensure_ascii=False, indent=2) + "\n")
            file.write("MONOLITHIC LUNA\n" + str(r.get("luna_output", r.get("luna_error", r.get("retrieval_error")))) + "\n")
            file.write("OBSERVED LUNA REPLY TYPE: " + str(r.get("luna_reply_type", "ERROR")) + "\n")
            file.write("JEV ROUTE + POLICY\n" + json.dumps(r.get("jev_decision", {
                "error": r.get("jev_error", r.get("retrieval_error"))
            }), ensure_ascii=False, indent=2) + "\n")
            file.write("JEV + LUNA\n" + str(r.get("jev_output", r.get("jev_error", r.get("retrieval_error")))) + "\n")
            file.write("JEV LABEL MATCH / ACTION MATCH: "
                       f"{r.get('jev_scope_match', 'ERROR')} / {r.get('jev_action_match', 'ERROR')}\n")
    print("\nSUMMARY (automatic gate/label scoring; human-review answer content)")
    print(json.dumps(summary, indent=2))
    print(f"\nJSONL: {out}\nReadable report: {report}")


if __name__ == "__main__":
    main()

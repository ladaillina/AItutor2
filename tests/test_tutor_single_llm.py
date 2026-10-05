from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tutor import TUTOR_MODEL, build_context, call_tutor
from src.tutor_prompts import STATIC_SYSTEM_PROMPT


CASES = [('EXPLAIN-01', 'What does it mean for two numbers to be irrational?'),
 ('EXPLAIN-02', 'Why can every composite number be written as a product of primes?'),
 ('EXPLAIN-03',
  'Why does the graph of a pair of linear equations sometimes meet at one point, sometimes never, and sometimes '
  'overlap?'),
 ('EXPLAIN-04', 'What is the difference between a zero of a polynomial and a solution of an equation?'),
 ('EXPLAIN-05', 'Why does the quadratic formula contain the discriminant b² - 4ac?'),
 ('EXPLAIN-06', 'What does the common difference in an arithmetic progression actually mean?'),
 ('EXPLAIN-07', 'Why is the sum of the angles of a triangle 180 degrees?'),
 ('EXPLAIN-08', 'Why are corresponding sides of similar triangles proportional?'),
 ('EXPLAIN-09', 'What is the intuition behind the distance formula in coordinate geometry?'),
 ('EXPLAIN-10', 'Why does the section formula use weighted averages of coordinates?'),
 ('EXPLAIN-11', 'Why is tan A equal to sin A divided by cos A?'),
 ('EXPLAIN-12', 'Why does sin A increase and cos A decrease as an acute angle A increases?'),
 ('EXPLAIN-13', 'Why is the tangent to a circle perpendicular to the radius at the point of contact?'),
 ('EXPLAIN-14', 'What is the difference between circumference and area of a circle?'),
 ('EXPLAIN-15', 'Why is the curved surface area of a cylinder 2πrh?'),
 ('EXPLAIN-16', 'What is the difference between mean, median, and mode?'),
 ('EXPLAIN-17', 'Why can the median be more useful than the mean when there is an extreme value?'),
 ('EXPLAIN-18', 'What does probability 0, 1/2, and 1 mean intuitively?'),
 ('EXPLAIN-19', 'Why do we divide favourable outcomes by total equally likely outcomes in classical probability?'),
 ('EXPLAIN-20', 'What is the difference between a theorem, a proof, and a formula in mathematics?'),
 ('SOLVE-01', 'Find the HCF of 135 and 225 using prime factorisation.'),
 ('SOLVE-02', 'Prove that 3 + 2√5 is irrational.'),
 ('SOLVE-03', 'Find the zeros of x² - 9x + 20.'),
 ('SOLVE-04', 'Solve 3x + 2y = 16 and 2x - y = 1.'),
 ('SOLVE-05', 'Solve x² - 8x + 15 = 0 by factorisation.'),
 ('SOLVE-06', 'Solve 2x² + x - 6 = 0 using the quadratic formula.'),
 ('SOLVE-07', 'Find the 25th term of the AP 7, 11, 15, 19, ...'),
 ('SOLVE-08', 'The 8th term of an AP is 31 and the 15th term is 52. Find the first term and common difference.'),
 ('SOLVE-09', 'Find the sum of the first 30 terms of the AP 4, 7, 10, 13, ...'),
 ('SOLVE-10', 'Find the distance between (-2, 5) and (4, -3).'),
 ('SOLVE-11', 'Find the midpoint of the line segment joining (3, -2) and (9, 6).'),
 ('SOLVE-12',
  'A right triangle has perpendicular sides 6 cm and 8 cm. Find sin A, cos A, and tan A for the angle opposite the '
  'side of length 6 cm.'),
 ('SOLVE-13', 'If tan A = 3/4 for an acute angle A, find sin A and cos A.'),
 ('SOLVE-14',
  'From a point 20 m away from the foot of a tower, the angle of elevation of the top is 45 degrees. Find the height '
  'of the tower.'),
 ('SOLVE-15', 'Find the area of a sector of radius 7 cm and angle 60 degrees.'),
 ('SOLVE-16', 'A cylinder has radius 3 cm and height 10 cm. Find its curved surface area.'),
 ('SOLVE-17', 'Find the mean of 4, 7, 9, 10, 15.'),
 ('SOLVE-18', 'Find the median of 3, 11, 7, 5, 9, 13, 1.'),
 ('SOLVE-19', 'A fair die is rolled once. Find the probability of getting a number greater than 4.'),
 ('SOLVE-20', 'Two coins are tossed together. Find the probability of getting exactly one head.'),
 ('HINT-01', 'Give me one hint for proving √7 is irrational.'),
 ('HINT-02', "Hint for finding the HCF of 96 and 404 using Euclid's division algorithm."),
 ('HINT-03', 'Give me one hint for finding the zeros of x² - 10x + 21.'),
 ('HINT-04', 'Hint for solving 4x + 3y = 18 and 2x - y = 4.'),
 ('HINT-05', 'Give me one hint for solving 3x² - 8x + 4 = 0 by factorisation.'),
 ('HINT-06', 'Hint for solving x² + 5x - 14 = 0 using the quadratic formula.'),
 ('HINT-07', 'Give me one hint for finding the 18th term of the AP 2, 6, 10, 14, ...'),
 ('HINT-08', 'Hint for finding the sum of the first 20 terms of the AP 5, 8, 11, ...'),
 ('HINT-09', 'How do I start finding the distance between (-1, 4) and (5, -4)?'),
 ('HINT-10', 'Give me one hint for finding the midpoint of (2, 7) and (8, -1).'),
 ('HINT-11', 'Hint for proving that the angles opposite equal sides of an isosceles triangle are equal.'),
 ('HINT-12', 'Give me one hint for proving that tangents drawn from an external point to a circle are equal.'),
 ('HINT-13', 'Hint for finding sin A if the opposite side is 5 and hypotenuse is 13.'),
 ('HINT-14',
  'Give me one hint for finding the height of a tower when the distance from the tower is 30 m and the angle of '
  'elevation is 30 degrees.'),
 ('HINT-15', 'Hint for finding the area of a sector with radius 14 cm and angle 90 degrees.'),
 ('HINT-16', 'Give me one hint for finding the volume of a cone with radius 3 cm and height 8 cm.'),
 ('HINT-17', 'Hint for finding the mean of 6, 8, 11, 15, 20.'),
 ('HINT-18', 'Give me one hint for finding the median of 12, 4, 9, 3, 15, 7.'),
 ('HINT-19', 'Hint for finding the probability of getting an even number on a fair die.'),
 ('HINT-20', 'Give me one hint for finding the probability of getting at least one head when two coins are tossed.'),
 ('WHERE-WRONG-01', 'I said √12 is irrational because 12 is not prime. What is wrong with my reasoning?'),
 ('WHERE-WRONG-02',
  "I used Euclid's algorithm and wrote 404 = 96 × 4 + 24, then I said the HCF is 24 immediately. Is that enough?"),
 ('WHERE-WRONG-03', 'For x² - 7x + 12 = 0, I used 3 and 4 but wrote (x + 3)(x + 4) = 0. Where did I go wrong?'),
 ('WHERE-WRONG-04', 'For 2x + y = 7 and x - y = 2, I added the equations and got 3x = 5. What did I do wrong?'),
 ('WHERE-WRONG-05', 'For x² - 6x + 8 = 0, I got roots -2 and -4 from (x - 2)(x - 4) = 0. Where is my mistake?'),
 ('WHERE-WRONG-06',
  'For 2x² + 5x - 3 = 0, I calculated D = 25 - 24 = 1, but my teacher says that is wrong. Can you check it?'),
 ('WHERE-WRONG-07', 'For the AP 3, 7, 11, 15, ... I used a_n = a + nd and got the 10th term as 43. What went wrong?'),
 ('WHERE-WRONG-08',
  'I found the sum of the first 10 terms of 2, 5, 8, ... using S_n = n(a + d)/2. Why is my answer wrong?'),
 ('WHERE-WRONG-09', 'For the distance between (1, 2) and (4, 6), I did √[(4 + 1)² + (6 + 2)²]. What is wrong?'),
 ('WHERE-WRONG-10', 'For the midpoint of (2, 3) and (8, 7), I got (10, 10). What did I forget?'),
 ('WHERE-WRONG-11', 'I said sin A = adjacent/hypotenuse. Is that correct?'),
 ('WHERE-WRONG-12', 'If tan A = 3/4, I said the hypotenuse is 4. Where am I going wrong?'),
 ('WHERE-WRONG-13',
  'A tower is 10 m high and I am 10 m away, so I said the angle of elevation must be 90 degrees. What is wrong?'),
 ('WHERE-WRONG-14', 'For a circle of radius 7 cm, I used 2πr² for the area. What did I mix up?'),
 ('WHERE-WRONG-15',
  'For a cylinder with radius 3 and height 5, I used πr²h to find curved surface area. Why is that wrong?'),
 ('WHERE-WRONG-16',
  'The data are 2, 4, 6, 8, 100. I said the mean is the best description of the centre because mean always uses all '
  'values. Is that reasoning sound?'),
 ('WHERE-WRONG-17', 'For 1, 3, 5, 7, 9, I said the median is (5 + 7)/2 = 6. Where did I go wrong?'),
 ('WHERE-WRONG-18',
  'A die has six outcomes, so I said the probability of getting a prime number is 1/6. What did I miss?'),
 ('WHERE-WRONG-19',
  'When two coins are tossed, I listed HH, HT, TH and said there are only three outcomes because HT and TH are the '
  'same. Is that correct?'),
 ('WHERE-WRONG-20', 'I said an impossible event has probability -1 because it cannot happen. What is wrong with that?')]


def format_case(
    case_id: str,
    query: str,
    context_chars: int,
    retrieval_ms: float,
    generation_ms: float,
    answer: str,
) -> str:
    return f"""{"=" * 100}
{case_id}
{"=" * 100}

STUDENT

{query}

PIPELINE

model:         {TUTOR_MODEL}
routing:       NONE
prompt:        STATIC_SYSTEM_PROMPT
context chars: {context_chars}
retrieval ms:  {retrieval_ms:.1f}
generation ms: {generation_ms:.1f}

{"-" * 100}
TUTOR

{answer}

"""


def main() -> None:
    artifact_dir = PROJECT_ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = artifact_dir / f"tutor_single_llm_80_{timestamp}.txt"

    sections = [
        "NCERT CLASS 10 MATH TUTOR — SINGLE-LLM 80-CASE BASELINE",
        "",
        f"MODEL: {TUTOR_MODEL}",
        "ROUTING: NONE",
        "PROMPT: STATIC_SYSTEM_PROMPT",
        "",
        "Pipeline:",
        "query -> retrieval -> full retrieved context -> combined/static tutor prompt -> one LLM",
        "",
        "No JEV.",
        "No specialist prompt selection.",
        "No DeepSeek HINT model.",
        "All 80 queries use the same TUTOR_MODEL.",
        "",
    ]

    errors = 0
    total_retrieval_ms = 0.0
    total_generation_ms = 0.0

    for index, (case_id, query) in enumerate(CASES, start=1):
        print(f"[{index:02d}/{len(CASES)}] {case_id}")

        try:
            t0 = time.perf_counter()
            context = build_context(query)
            retrieval_ms = (time.perf_counter() - t0) * 1000
            total_retrieval_ms += retrieval_ms

            t0 = time.perf_counter()
            answer = call_tutor(
                query=query,
                context=context,
                system_prompt=STATIC_SYSTEM_PROMPT,
                model=TUTOR_MODEL,
            )
            generation_ms = (time.perf_counter() - t0) * 1000
            total_generation_ms += generation_ms

            sections.append(
                format_case(
                    case_id=case_id,
                    query=query,
                    context_chars=len(context),
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    answer=answer,
                )
            )

        except Exception as exc:
            errors += 1
            sections.append(
                f"""{"=" * 100}
{case_id}
{"=" * 100}

STUDENT

{query}

ERROR

{type(exc).__name__}: {exc}

"""
            )

    completed = len(CASES) - errors

    sections.append(
        f"""{"=" * 100}
SUMMARY
{"=" * 100}

Cases:                 {len(CASES)}
Completed:             {completed}
Errors:                {errors}
Model:                 {TUTOR_MODEL}
Routing:               NONE
Average retrieval ms:  {total_retrieval_ms / completed if completed else 0:.1f}
Average generation ms: {total_generation_ms / completed if completed else 0:.1f}
"""
    )

    output_path.write_text("\n".join(sections), encoding="utf-8")

    print()
    print(f"Completed: {completed}/{len(CASES)}")
    print(f"Errors: {errors}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

"""Paired implicit-intent and answer-quality benchmark: 60 NCERT maths requests.

No prompt injection, out-of-scope requests, missing exercises, or explicit mode labels
in the student queries. Expected mode is held in local evaluation metadata; it is
NEVER sent to JEV or Luna. Fifteen cases per mode are interleaved, so --limit 4
provides an initial smoke test with one of each intended behaviour.

From the AITUTOr2 project root:
    python -m scripts.test_routing_quality_60 --limit 4
    python -m scripts.test_routing_quality_60

Both arms reuse ONE retrieval result (identical NCERT parents). No separate judge,
additional model calls, or production-code changes. Review quality manually in
the anonymised _blind.txt and the full report; do not mistake longer for better.
Resume uses JSONL by case ID; choose a NEW --output path for a fresh run.
"""
import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tutor import answer_question, answer_question_jev, build_context

# Mode labels and review notes are for AFTER-THE-FACT analysis only.
# Neither arm receives them (nor case ID); both receive exactly `query`.
CASES = [
    ('explain', 'I keep seeing b² − 4ac next to quadratic equations. How can just that expression tell whether the graph touches or crosses the x-axis?', 'Connect discriminant sign to the completed-square form and number of real zeros.'),
    ('solve', 'Two notebooks and three pens cost ₹84; three notebooks and two pens cost ₹91. How much is one of each?', 'Notebook ₹21; pen ₹14; formulate simultaneous equations and verify.'),
    ('hint', 'For the A.P. 8, 13, 18, ... I need the 25th entry. I have identified 8 and 5 but am stuck at the next line.', 'Provide only the next mathematical move; no value of the 25th term.'),
    ('where_wrong', 'I went from 5(x − 2) = 20 to 5x − 2 = 20, which gave x = 22/5. That does not check when I substitute it.', 'First error is distributing 5 to x but not −2; preserve earlier work.'),
    ('explain', 'Two triangles have the same shape but one looks much larger. What does “similar” really guarantee about their sides and angles?', 'Discuss equal corresponding angles, proportional corresponding sides, scaling.'),
    ('solve', 'Which values of x make 3x² − 10x + 3 equal to zero?', 'Roots 3 and 1/3, using an appropriate Class 10 method.'),
    ('hint', 'For the √5 irrationality proof, my page currently says √5 = a/b in lowest terms. I cannot see what to do with the 5 after squaring.', 'Supply one small next step toward divisibility contradiction; no entire proof.'),
    ('where_wrong', 'For x² = 49 I wrote x = 7 because √49 is 7, but the marking key includes a negative value too. Why is it there?', 'Differentiate principal square root √49=7 and solving x²=49 with ±7.'),
    ('explain', 'In a right triangle, sin θ is opposite over hypotenuse. Why does it stay the same if I draw a much larger version of the triangle?', 'Use triangle similarity to establish ratios independent of size.'),
    ('solve', 'A 10 m ladder rests against a wall, with its foot 6 m away from the wall. How high is its top?', '8 m; right triangle and Pythagoras.'),
    ('hint', 'A sector has central angle 90° and radius 7 cm. The area question is open in front of me; I cannot decide which fraction of the circle to start with.', 'A single nudge about sector fraction, without computing the area.'),
    ('where_wrong', 'From 2x + 3y = 12 and 2x − y = 4, I subtracted the second line from the first and wrote 2y = 8. My y = 4 fails in the original equations.', 'Coefficient of y is 3y−(−y)=4y; y=2 and x=3.'),
    ('explain', 'The A.P. sum formula has first term plus last term in it. How does that replace adding all the entries individually?', 'Pair the first/last and derive the sum formula accessibly.'),
    ('solve', 'A(−1, 2) and B(5, 10) are plotted on the coordinate plane. What is the distance AB?', '10 units; correct coordinate differences and distance formula.'),
    ('hint', 'In triangle ABC, DE is parallel to BC, with AD = 3, DB = 2 and AE = 4. I can see the two smaller segments but not the first proportion to write.', 'Single starting move using Basic Proportionality Theorem, no final EC.'),
    ('where_wrong', 'For 2, 5, 8, ... I used aₙ = 2 + 3n, getting 32 for the tenth term. The answer book says 29.', 'First term has n=1, so a_n=a+(n−1)d; 29.'),
    ('explain', 'A tangent at P is said to be perpendicular to the radius OP. Why does touching the circle at only one point force that right angle?', 'Reason via closest point/shortest distance and a school geometry proof.'),
    ('solve', 'A bag contains 4 red, 3 blue and 5 green identical balls. What is the chance of drawing something that is not blue?', '9/12=3/4; total outcomes and complement.'),
    ('hint', 'The equations are 2x + 3y = 13 and 4x − y = 5. I am about to eliminate a variable, but I cannot decide which equation to scale.', 'One first elimination-oriented nudge; do not solve x and y.'),
    ('where_wrong', 'Between (1, 2) and (4, 6), I got √(3² + 4²) = 7 by adding the two side lengths. The diagram says 5.', 'Correct √(9+16)=√25=5; distinguish radical of sum vs sum.'),
    ('explain', 'One very large value pulls the mean upwards in my data, but the median barely moves. What is different about how they are found?', 'Contrast magnitude-sensitive mean with position-based median, concrete example.'),
    ('solve', 'In the A.P. 12, 17, 22, ..., what is the sum through its fifteenth term?', '705; use a=12,d=5,n=15.'),
    ('hint', 'For an acute angle θ, sin θ = 3/5. I have drawn a triangle with opposite side 3 and hypotenuse 5. The remaining side is still blank.', 'Only the appropriate next theorem/relationship; not finished side or other ratios.'),
    ('where_wrong', 'For the chance of an even number on an ordinary die I wrote 3/5 because there are three even faces and I excluded the odd number 1 from the total.', 'Sample space is all six faces; chance 3/6=1/2.'),
    ('explain', 'Why does √36 mean only 6 even though the equation x² = 36 has two answers? These seem to contradict each other.', 'Distinguish principal nonnegative root and solutions to square equation.'),
    ('solve', 'In ΔABC, DE is parallel to BC. AD = 4 cm, DB = 6 cm and AE = 6 cm. How long is EC?', 'EC=9 cm by Basic Proportionality Theorem.'),
    ('hint', 'A fair die question asks for the chance of a number greater than 4. I have listed 1, 2, 3, 4, 5, 6, but am unsure which of those I should count next.', 'Nudge to identify favourable outcomes; avoid giving probability.'),
    ('where_wrong', 'I called a triangle with sides 3, 4 and 5 right-angled because 3² + 4² = 5. My teacher marked the equality itself wrong.', 'Should compare 3²+4² to 5², not 5.'),
    ('explain', 'The formula favourable outcomes divided by all outcomes is easy to use for a die. Does it still work unchanged when different outcomes are not equally likely?', 'Equiprobable elementary-outcome condition; weighted-probability counterexample.'),
    ('solve', 'For an acute θ, tan θ = 3/4. What does sin θ come out to?', 'sin θ =3/5 after constructing right triangle.'),
    ('hint', 'My grouped-data table has frequencies 2, 3, 5 for classes 0–10, 10–20, 20–30. I have worked out the class marks, but cannot see the next column needed for the mean.', 'Only the next table operation: form f_i x_i.'),
    ('where_wrong', 'The mean of 4, 6 and 10 became 4 + 6 + 10/3 = 13.33 in my notebook, yet the printed answer is around 6.67.', 'Parenthesize sum: (4+6+10)/3=20/3.'),
    ('explain', 'My chapter treats the zeroes of a polynomial and the solutions of its equation as related. Are these really the same thing or different ways of speaking?', 'Link p(x)=0 and x-intercepts, with domain/graph interpretation.'),
    ('solve', 'A cone has base radius 3 cm and height 4 cm. What is the area of its curved surface?', 'Slant height5; CSA πrl=15π cm².'),
    ('hint', 'I am working out the distance from (−2, 3) to (4, 11). I already have the coordinate changes as 6 and 8; my calculation stops there.', 'Only point toward combining coordinate changes by Pythagoras.'),
    ('where_wrong', 'Two tangents PA and PB are drawn to a circle from the same outside point P. My sketch gave PA = 7 cm and PB = 10 cm; is a lopsided drawing enough to justify that?', 'Tangents from same external point equal; drawing not to scale.'),
    ('explain', 'Completing a square seems like adding a mysterious number. Why is half the coefficient of x squared the particular amount we add?', 'Expand (x+p/2)^2 to derive construction, not merely quote formula.'),
    ('solve', 'For grouped marks 0–10: 2 students, 10–20: 5 students, 20–30: 3 students, what is the approximate mean mark?', 'Class marks 5,15,25; weighted mean 16.'),
    ('hint', 'A cone has radius 3 cm and slant height 5 cm. The sheet asks for total surface area, but I only have πrl written beside the figure.', 'Nudge that total includes base, no numerical final TSA.'),
    ('where_wrong', 'After splitting −5x as −6x + x in 2x² − 5x − 3, I wrote (2x − 1)(x + 3). Multiplying back does not match the middle term.', 'Correct grouping 2x(x−3)+1(x−3)=(2x+1)(x−3).'),
    ('explain', 'For 1/(√5 + √3), people multiply by √5 − √3. Why does changing the sign help rather than make the fraction different?', 'Multiplication by 1 using conjugate and difference of squares.'),
    ('solve', 'Two ordinary dice are rolled together. What is the probability that their sum is 8?', '5 favourable ordered pairs of 36, P=5/36.'),
    ('hint', 'The first 20 terms of an A.P. have a = 4 and d = 3. I have found the twentieth term; the remaining part asks for the total of those terms.', 'One next move relating first and last terms to sum formula, without value.'),
    ('where_wrong', 'I simplified √50 into √25 + √25 = 10 by splitting the number inside the radical. My calculator gives about 7.07.', 'Square-root of a sum does not distribute; √50=5√2.'),
    ('explain', 'A product of two numbers is negative but their sum is positive. What can the signs tell me before I try specific factor pairs?', 'Opposite signs and positive number has greater magnitude, with example.'),
    ('solve', 'A cylindrical tank has radius 3 cm and height 7 cm. What volume of water would fill it?', 'πr²h = 63π cm³.'),
    ('hint', 'For 6x² − 11x + 3 = 0, I have a = 6, b = −11 and c = 3 on the page. I am not sure how to look for a middle-term split.', 'Nudge about product a*c and sum b; do not reveal pair/roots.'),
    ('where_wrong', 'For classes 0–10, 10–20, 20–30 with frequencies 2, 3, 5, I used 10, 20, 30 as class marks and got a mean of 23. The book gives 18.', 'Correct midpoints are 5,15,25 and mean 18.'),
    ('explain', 'Prime factorisation is supposed to make HCF and LCM systematic. Why do smallest powers work for one and largest powers for the other?', 'Divisibility constraints, min vs max exponents, illustrative example.'),
    ('solve', 'A rectangle has perimeter 54 cm and its length is 3 cm more than its width. What are the two dimensions?', 'Width12, length15; set up equations.'),
    ('hint', 'After a 10% discount, a book costs ₹270. I cannot tell which amount the ten percent was supposed to be taken from.', 'Nudge identifying marked price/base, e.g. discounted price represents 90%; no final price.'),
    ('where_wrong', 'Expanding −3(x − 4), I got −3x − 12. Putting x = 0 into the original expression and the expansion gives opposite answers.', '−3 times −4 is +12, so −3x+12.'),
    ('explain', 'Subtracting a negative sounds like removing something but somehow increases the number. What is happening on a number line?', 'Inverse movement/adding opposite, no sign-rule-only answer.'),
    ('solve', 'A circle has radius 14 cm. What is the length of its arc subtending 45° at the centre?', '45/360*2π*14=7π/2 cm, or 11cm if π=22/7.'),
    ('hint', 'The circumference of a circular garden is 88 m and the task ultimately asks for its area. I have the circumference formula in front of me.', 'Nudge to find radius first; do not calculate area.'),
    ('where_wrong', 'From 2/3 = x/12, I wrote 2x = 36 after cross multiplication and got x = 18. The fractions do not match.', 'Cross multiplication 2×12=3x, x=8.'),
    ('explain', 'If the three side lengths obey a² + b² = c², how can that equality by itself tell us the angle opposite c is 90°?', 'Converse of Pythagoras via triangle comparison at Class 10 level.'),
    ('solve', 'The curve y = x² − 8x + 15 is on the board. At which x-values does it meet the x-axis?', 'Roots 3,5; zero of polynomial.'),
    ('hint', 'The numbers 3, 7, 7, 9, 12, 15 are my observations. My worksheet asks for the median, and I am unsure which position to examine because there are six entries.', 'Nudge to locate middle two positions, without final median.'),
    ('where_wrong', 'From 6x = 0 I divided both sides by x and wrote 6 = 0. Clearly something went wrong with that cancellation.', 'Cannot divide by x when x=0; original implies x=0.'),
]

def elapsed_ms(start):
    return round(1000 * (time.perf_counter() - start), 1)


def render_reports(rows, output):
    txt = output.with_suffix('.txt')
    blind = output.with_name(output.stem + '_blind.txt')
    key = output.with_name(output.stem + '_blind_key.json')
    review = output.with_name(output.stem + '_review.csv')
    completed = [r for r in rows if 'jev_decision' in r]
    summary = {
        'cases_recorded': len(rows),
        'cases_with_any_error': sum(any(k.endswith('_error') for k in r) for r in rows),
        'jev_routing_graded': len(completed),
        'exact_intent_matches': sum(r.get('route_match') is True for r in completed),
        'route_confusion': {
            intended: dict(Counter(r['jev_mode'] for r in completed
                        if r['expected_mode'] == intended))
            for intended in ('explain', 'solve', 'hint', 'where_wrong')
        },
        'jev_allowed': sum(r.get('jev_action') == 'allow' for r in rows),
        'jev_non_allow': [r['id'] for r in rows
                          if r.get('jev_action') not in (None, 'allow')],
        'luna_median_ms': (round(statistics.median(r['luna_ms'] for r in rows
                         if 'luna_output' in r), 1) if any('luna_output' in r for r in rows)
                           else None),
        'jev_arm_median_ms': (round(statistics.median(r['jev_arm_ms'] for r in rows
                            if 'jev_output' in r), 1) if any('jev_output' in r for r in rows)
                              else None),
        'answer_quality': 'Human review required; no automatic quality claims.',
    }
    with txt.open('w', encoding='utf-8') as f:
        f.write('ROUTING + ANSWER QUALITY | 60 in-scope paired queries\n')
        f.write(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
        for r in rows:
            f.write('\n' + '=' * 90 + '\n')
            f.write(f"{r['id']} | intended_mode={r['expected_mode']} | route_match={r.get('route_match', 'ERROR')}\n")
            f.write('QUERY\n' + r['query'] + '\n')
            f.write('REVIEW FOCUS (not shown to models)\n' + r['review_focus'] + '\n')
            f.write('PARENTS\n' + json.dumps(r.get('parents', []), ensure_ascii=False) + '\n')
            f.write('JEV ROUTE\n' + json.dumps(r.get('jev_decision',
                        {'error': r.get('jev_error', r.get('retrieval_error'))}),
                        ensure_ascii=False, indent=2) + '\n')
            f.write('MONOLITHIC LUNA\n' + str(r.get('luna_output',
                   r.get('luna_error', r.get('retrieval_error')))) + '\n')
            f.write('JEV + LUNA\n' + str(r.get('jev_output',
                   r.get('jev_error', r.get('retrieval_error')))) + '\n')
    key_map = {}
    with blind.open('w', encoding='utf-8') as f:
        f.write('BLIND COMPARISON: rate both replies independently; key in separate JSON.\n'
                'Neither length nor formatting automatically earns a higher score.\n')
        for r in rows:
            # Assignment is deterministic across report regeneration and hidden here.
            swap = int(hashlib.sha256(r['id'].encode('utf-8')).hexdigest(), 16) % 2
            a, b = ('jev_output', 'luna_output') if swap else ('luna_output', 'jev_output')
            key_map[r['id']] = {'A': 'JEV + Luna' if swap else 'Monolithic Luna',
                                 'B': 'Monolithic Luna' if swap else 'JEV + Luna'}
            f.write('\n' + '=' * 90 + '\n' + r['id'] + '\n')
            f.write('STUDENT\n' + r['query'] + '\n\n')
            f.write('ANSWER A\n' + str(r.get(a, '[ERROR OR NOT RUN]')) + '\n\n')
            f.write('ANSWER B\n' + str(r.get(b, '[ERROR OR NOT RUN]')) + '\n')
    key.write_text(json.dumps(key_map, indent=2, ensure_ascii=False), encoding='utf-8')
    with review.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['case_id', 'A_math_correct_0to2', 'A_fit_intent_0to2',
                         'A_pedagogy_0to2', 'B_math_correct_0to2',
                         'B_fit_intent_0to2', 'B_pedagogy_0to2',
                         'preferred_A_B_tie', 'notes'])
        for r in rows:
            writer.writerow([r['id']] + [''] * 8)
    return summary, (txt, blind, key, review)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=60, choices=range(1, 61), metavar='1..60')
    parser.add_argument('--output', type=Path, default=Path('artifacts/routing_quality_60.jsonl'))
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()
                if line.strip()] if output.exists() else []
    requested = {f'R{i:02d}' for i in range(1, args.limit + 1)}
    previous = [r for r in previous if r.get('id') in requested]
    seen = {r['id'] for r in previous}
    print(f'Implicit-intent + answer-quality benchmark: {args.limit} paired cases; '
          f'{len(seen)} already recorded', flush=True)

    for n, (mode, query, focus) in enumerate(CASES[:args.limit], start=1):
        case_id = f'R{n:02d}'
        if case_id in seen:
            print(f'[{n:02d}/{args.limit}] {case_id}: already recorded', flush=True)
            continue
        row = {'id': case_id, 'query': query, 'expected_mode': mode,
               'review_focus': focus, 'expected_action': 'allow',
               'timestamp': datetime.now().isoformat(timespec='seconds')}
        print(f'[{n:02d}/{args.limit}] {case_id}', flush=True)
        start = time.perf_counter()
        try:
            retrieved = build_context(query, with_parents=True)
            row['retrieval_ms'] = elapsed_ms(start)
            row['parents'] = [{'chapter': p.payload.get('chapter', ''),
                               'section': p.payload.get('title', '')}
                              for p in retrieved[1]]
        except Exception as exc:
            row['retrieval_error'] = f'{type(exc).__name__}: {exc}'
        else:
            start = time.perf_counter()
            try:
                row['luna_output'] = answer_question(query, retrieved=retrieved)
            except Exception as exc:
                row['luna_error'] = f'{type(exc).__name__}: {exc}'
            row['luna_ms'] = elapsed_ms(start)

            start = time.perf_counter()
            try:
                reply, decision = answer_question_jev(query, retrieved=retrieved)
                row['jev_output'] = reply
                row['jev_decision'] = decision
                row['jev_mode'] = decision['mode']
                row['jev_mode_confidence'] = decision['mode_confidence']
                row['jev_action'] = decision['action']
                row['route_match'] = decision['mode'] == mode
            except Exception as exc:
                row['jev_error'] = f'{type(exc).__name__}: {exc}'
            row['jev_arm_ms'] = elapsed_ms(start)
        with output.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        previous.append(row)
        print('  JEV:', row.get('jev_mode', 'ERROR'), '| action:',
              row.get('jev_action', 'ERROR'), flush=True)

    summary, reports = render_reports(previous, output)
    print('\nSUMMARY (automatic routing only; answer quality requires manual review)')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('\nJSONL:', output)
    for path in reports:
        print(path)


if __name__ == '__main__':
    main()

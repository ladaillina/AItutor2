"""Disposable, read-only audit of exercise/question coverage in corrected MinerU JSON.

Run from the AITUTOr2 project root:
    python -m scripts.audit_exact_questions

It does NOT alter data, regenerate chunks, call Google, or touch Qdrant.
The question-number scan is a diagnostic heuristic, NOT a question parser.
"""

import json
import re
from pathlib import Path

SOURCE = Path('data/mineru_corrected/structured_content.json')
REPORT = Path('artifacts/exact_question_audit.json')
EXERCISE = re.compile(r'^EXERCISE\s*((?:\d+|A\d+)[.,]\d+)\b', re.I)
SECTION = re.compile(r'^(?:\d+|A\d+)\.\d+(?:\.\d+)*\b')
QUESTION = re.compile(r'(?:^|\n)\s*(\d{1,2})\.\s+(?=\S)')
TEXT_TYPES = {'text', 'equation', 'table', 'paragraph_title'}


def main():
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    exercises = []
    current = None
    chapter = None
    answer_key_page = None

    for page in data['pages']:
        page_idx = page['page_idx']
        if page_idx < 14:  # same front-matter cutoff as src.ingestion
            continue
        for block_idx, block in enumerate(page['blocks']):
            kind = block.get('type', '')
            body = block.get('content', '').strip()
            match = EXERCISE.match(body) if kind in ('doc_title', 'paragraph_title') else None

            if kind == 'doc_title' and body.upper().startswith('ANSWERS/HINTS'):
                answer_key_page = page_idx
                current = None
                break  # ignore answer key, not original exercise questions

            if match:
                current = {
                    'exercise': match.group(1).replace(',', '.'),
                    'chapter': chapter,
                    'heading': body,
                    'heading_type': kind,
                    'page_start': page_idx,
                    'page_end': page_idx,
                    'question_numbers_seen': [],
                    'question_start_locations': [],
                    'image_blocks': [],
                    'text_blocks': 0,
                    'warnings': [],
                }
                if kind == 'doc_title':
                    current['warnings'].append('EXERCISE_IS_DOC_TITLE: current build_parents treats it as a new chapter and skips its questions')
                exercises.append(current)
                continue

            if kind == 'doc_title':
                chapter, current = body, None
                continue
            if kind == 'paragraph_title' and SECTION.match(body):
                current = None
                continue
            if current is None:
                continue

            current['page_end'] = page_idx
            if kind == 'image':
                current['image_blocks'].append({'page': page_idx, 'block': block_idx, 'source': block.get('image_source')})
            elif kind in TEXT_TYPES and body:
                current['text_blocks'] += 1
                for found in QUESTION.finditer(body):
                    number = int(found.group(1))
                    current['question_numbers_seen'].append(number)
                    current['question_start_locations'].append({'number': number, 'page': page_idx, 'block': block_idx})

        if answer_key_page is not None:
            break

    for exercise in exercises:
        numbers = exercise['question_numbers_seen']
        unique = set(numbers)
        if not numbers:
            exercise['warnings'].append('NO_NUMBERED_QUESTION_STARTS_DETECTED')
        else:
            missing = sorted(set(range(1, max(unique) + 1)) - unique)
            if missing:
                exercise['warnings'].append(f'NUMBERING_GAPS: {missing} (inspect source; scan is heuristic)')
            if len(unique) != len(numbers):
                exercise['warnings'].append('REPEATED_QUESTION_NUMBERS: inspect merged/subquestion blocks')
            if numbers[0] != 1:
                exercise['warnings'].append('FIRST_DETECTED_NUMBER_NOT_1')
        if exercise['image_blocks']:
            exercise['warnings'].append('CONTAINS_IMAGE_BLOCKS: text-only question extraction may be incomplete')
        exercise['distinct_question_numbers'] = len(unique)

    report = {
        'source': str(SOURCE),
        'answer_key_starts_at_page_idx': answer_key_page,
        'exercise_headings_before_answers': len(exercises),
        'doc_title_exercises': [x['heading'] for x in exercises if x['heading_type'] == 'doc_title'],
        'exercises_requiring_inspection': [x['exercise'] for x in exercises if x['warnings']],
        'exercise_details': exercises,
        'note': 'Question number detection is heuristic. No question-level index was generated. Inspect flagged exercises, continuation blocks and figures before implementing exact lookup.',
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Exercise headings before answers: {len(exercises)}')
    print(f'Answer key starts at page_idx: {answer_key_page}')
    print(f'Headings labelled doc_title: {report["doc_title_exercises"]}')
    print(f'Flagged exercises: {len(report["exercises_requiring_inspection"])}')
    print(f'Report: {REPORT}')
    for x in exercises:
        print(f'{x["exercise"]:>5} | p{x["page_start"]:>3}-{x["page_end"]:<3} | {x["heading_type"]:<15} | detected Q: {x["distinct_question_numbers"]:<2} | images: {len(x["image_blocks"]):<2} | {"; ".join(x["warnings"])}')


if __name__ == '__main__':
    main()

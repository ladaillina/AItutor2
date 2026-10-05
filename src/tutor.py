"""AITUTOr2: paired JEV+Luna versus monolithic Luna experiment.

Both arms START from identical Qdrant parent context if ``retrieved`` is shared.
The JEV arm can subsequently replace its hybrid context with the exact original
exercise when the new post-retrieval exact-reference gate succeeds.

* answer_question(): one Luna call evaluates security, curriculum, mode, answer.
* answer_question_jev(): one JEV guardrail call; if allowed, one Luna call.

No separate Luna judge/classifier or generation retry is performed.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.jev_guardrails import INJECTION_REVIEW, apply_policy, evaluate_request
from src.retrieval import lookup_exercise_parent, retrieve, retrieve_parents
from src.sessions import load_session, save_session
from src.tutor_prompts import STATIC_SYSTEM_PROMPT, build_routed_prompt, EXACT_REFERENCE_PROMPT

load_dotenv()
TUTOR_MODEL = os.getenv("TUTOR_MODEL", "openai/gpt-6-luna")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# Initial routing threshold, NOT a calibrated probability guarantee.
# Test this with labelled exact/non-exact requests before treating it as final.
EXACT_REFERENCE_THRESHOLD = 0.85

# Find EXPLICIT exercise identifiers only: Exercise 4.2, Ex. 5,3, Ex A1.3 Q4.
# The optional question group is passed to Luna; Python does not extract question text.
_EXERCISE_REFERENCE = re.compile(
    r"\b(?:exercise|ex\.?)\s*"
    r"(?P<exercise>A?[1-9]\d*[.,][1-9]\d*)\b"
    r"(?:\s*[,;:]?\s*(?:question|ques\.?|q\.?)\s*"
    r"(?:no\.?\s*)?(?P<question>[1-9]\d*)\b)?",
    re.IGNORECASE,
)
_QUESTION_REFERENCE = re.compile(r"\b(?:question|ques\.?|q\.?)\s*(?:no\.?\s*)?([1-9]\d*)\b", re.I)


def _format_parents(parents: list) -> str:
    return "\n\n".join(
        f"CHAPTER: {p.payload['chapter']}\nSECTION: {p.payload['title']}\n{p.payload['content']}"
        for p in parents
    )


def _reference(parents: list, *, exercise_id=None, question_number=None) -> dict | None:
    """Cache only evidence carrying genuine Qdrant point IDs."""
    if not parents or any(getattr(p, "id", None) is None for p in parents):
        return None
    ids = [str(p.id) for p in parents]
    first = parents[0].payload
    return {
        "kind": "exercise" if exercise_id else "section",
        "exercise_id": exercise_id, "question_number": question_number,
        "parent_id": ids[0], "parent_ids": ids,
        "chapter": first["chapter"], "title": first["title"],
        "parent_text": _format_parents(parents),
    }


def _verified(ref) -> bool:
    return isinstance(ref, dict) and bool(ref.get("parent_text")) and bool(
        ref.get("parent_id") or ref.get("parent_ids")
    )


def _clarify(decision: dict, message: str, issue: str) -> tuple[str, dict]:
    """Keep returned action consistent with a local evidence-verification stop."""
    decision.update(action="clarify", reply=message, grounding_issue=issue)
    return message, decision


def build_context(query: str, *, with_parents: bool = False):
    """Retrieve once; optionally return the same ranked parents for JEV.

    Default return type remains a string to preserve older caller behaviour.
    Use ``with_parents=True`` for a paired experiment with fixed retrieval.
    """
    children = retrieve(query)
    parents = retrieve_parents(children, limit=3) if children else []

    context = _format_parents(parents)
    return (context, parents) if with_parents else context


def call_tutor(
    query: str,
    context: str,
    system_prompt: str,
    model: str = TUTOR_MODEL,
    history: list | None = None,
) -> str:
    """Exactly one Luna generation call; dialogue is supplied as data, not authority."""
    recent = (
        "<recent_dialogue_data>\n"
        + json.dumps(history[-4:], ensure_ascii=False)
        + "\n</recent_dialogue_data>\n"
        if history else ""
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""{recent}<ncert_context>
{context}
</ncert_context>

<student_message>
{query}
</student_message>""",
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(query: str, *, retrieved: tuple | None = None) -> str:
    """MONOLITHIC: security + scope + mode + answer in ONE Luna call.

    If ``retrieved`` is supplied, reuse its already-formatted parent context.
    """
    context = retrieved[0] if retrieved is not None else build_context(query)
    return call_tutor(query=query, context=context, system_prompt=STATIC_SYSTEM_PROMPT)


def answer_question_jev(
    query: str, *, retrieved: tuple | None = None, session_id: str | None = None,
) -> tuple[str, dict]:
    """Existing one-JEV/one-Luna flow, optionally with ephemeral Redis continuity.

    The caller creates a session with src.sessions.create_session() and reuses its
    UUID. Omitting session_id keeps the original stateless experiment intact.
    An expired session is never silently recreated or treated as a new chat.
    """
    session = load_session(session_id) if session_id is not None else None
    if session_id is not None and session is None:
        raise ValueError("Session expired or unknown; create a new session.")

    active = session.get("active") if session is not None else None
    previous = session.get("previous") if session is not None else None
    active = active if _verified(active) else None
    previous = previous if _verified(previous) else None

    context, parents = (
        retrieved if retrieved is not None
        else build_context(query, with_parents=True)
    )
    if session is None:
        decision = evaluate_request(query, parents)
    else:
        # Only server-stored, complete Qdrant references are marked verified.
        conversation = {**session, "active": active, "previous": previous}
        decision = evaluate_request(
            query, parents, conversation=conversation,
            verified_cached_parent=active is not None,
            verified_previous_parent=previous is not None,
        )

    source = decision.get("context_source", "FRESH_ONLY")
    may_verify_exact = (
        decision["action"] == "clarify"
        and decision["injection_probability"] < INJECTION_REVIEW
        and decision["exact_reference_probability"] >= EXACT_REFERENCE_THRESHOLD
    )
    if decision["action"] != "allow" and not may_verify_exact:
        return decision["reply"], decision

    exact_id, exact_question, exact_ref = None, None, None
    exact_verified = False
    if decision["exact_reference_probability"] >= EXACT_REFERENCE_THRESHOLD:
        references = {
            (m.group("exercise").upper().replace(",", "."), m.group("question"))
            for m in _EXERCISE_REFERENCE.finditer(query)
        }
        if len(references) != 1:
            decision["exact_lookup"] = "ambiguous_reference"
            return _clarify(
                decision, "Please specify one NCERT exercise as, for example, "
                "'Exercise 4.2 Question 3'.", "ambiguous_exact_reference",
            )
        exact_id, exact_question = references.pop()
        exact_question = int(exact_question) if exact_question is not None else None
        cached = active if source == "CACHED_ONLY" else previous if source == "RESTORE_PREVIOUS" else None
        if cached is not None and cached.get("exercise_id") == exact_id:
            exact_verified = True
            exact_ref = dict(cached)
            exact_ref["question_number"] = exact_question or cached.get("question_number")
            decision["exact_lookup"] = "cached"
        else:
            # A source mismatch must not silently answer a different textbook exercise.
            if cached is not None:
                decision["exact_lookup"] = "context_conflict"
                return _clarify(decision, "Which exercise should I use: the current reference or the one you named?", "context_conflict")
            point = lookup_exercise_parent(exact_id)
            if point is None:
                decision["exact_lookup"] = "parent_missing"
                return _clarify(decision, f"I couldn't locate the original Exercise {exact_id}.", "exact_parent_missing")
            exact_verified = True  # A successful native Qdrant metadata lookup is verification.
            exact_ref = _reference([point], exercise_id=exact_id, question_number=exact_question)
            if exact_ref is None and session is not None:
                raise ValueError("Exact Qdrant parent has no verifiable point ID")
            # Without Redis, format the verified point as before.
            exact_context = _format_parents([point])
            decision["exact_lookup"] = "used"
            if source == "BOTH" and active is not None and str(point.id) not in active["parent_ids"]:
                decision["has_fresh_parent"] = True

        # Existing V2 scope rescue: pure policy re-evaluation, not a second JEV call.
        if may_verify_exact:
            decision = apply_policy(
                decision, verified_exact_parent=exact_verified,
                verified_cached_parent=active is not None,
                verified_previous_parent=previous is not None,
            )
            if decision["action"] != "allow":
                return decision["reply"], decision
        decision["exercise_id"] = exact_id
        decision["question_number"] = exact_question

    # Select only evidence sanctioned by the contextual decision. The last
    # verified parent text is already in Redis, so CACHED_ONLY needs no new
    # point fetch or embedding. Initial hybrid retrieval still ran before JEV.
    selected = None
    if source == "CACHED_ONLY":
        if active is None:
            return _clarify(decision, "Which question are you continuing?", "missing_active_reference")
        selected = dict(exact_ref or active)
        if exact_id is None and selected.get("exercise_id"):
            numbers = _QUESTION_REFERENCE.findall(query)
            if len(numbers) == 1:
                selected["question_number"] = int(numbers[0])
        context = selected["parent_text"]
    elif source == "RESTORE_PREVIOUS":
        if previous is None:
            return _clarify(decision, "Which previous question would you like to return to?", "missing_previous_reference")
        selected = dict(exact_ref or previous)
        context = selected["parent_text"]
    elif source == "BOTH":
        if active is None:
            return _clarify(decision, "Which earlier question should I compare against?", "missing_active_reference")
        if exact_id is not None:
            fresh = exact_ref
            fresh_context = exact_ref["parent_text"] if exact_ref else exact_context
        else:
            ids = set(map(str, active.get("parent_ids") or [active["parent_id"]]))
            distinct = [p for p in parents if getattr(p, "id", None) is not None and str(p.id) not in ids]
            fresh = _reference(distinct)
            fresh_context = _format_parents(distinct)
        if fresh is None or not set(fresh["parent_ids"]).isdisjoint(active.get("parent_ids") or [active["parent_id"]]):
            return _clarify(decision, "I couldn't verify distinct new textbook material for that comparison.", "missing_distinct_fresh_reference")
        context = active["parent_text"] + "\n\n" + fresh_context
        selected = {
            **active, "kind": "comparison", "exercise_id": None, "question_number": None,
            "title": f"{active['title']} + {fresh['title']}", "parent_text": context,
            "parent_ids": list(dict.fromkeys([*(active.get("parent_ids") or [active["parent_id"]]), *fresh["parent_ids"]])),
        }
    else:  # FRESH_ONLY, including the original stateless path.
        if exact_id is not None:
            context = exact_ref["parent_text"] if exact_ref else exact_context
            selected = exact_ref
        else:
            selected = _reference(parents) if session is not None else None

    system_prompt = build_routed_prompt(decision["mode"])
    # Preserve the exact-exercise instruction on later cached-only follow-ups,
    # including a deterministic question-number change within the same exercise.
    if selected and selected.get("kind") == "exercise" and source != "BOTH":
        system_prompt += "\n\n" + EXACT_REFERENCE_PROMPT.format(
            exercise=selected["exercise_id"],
            question=(f"Question {selected['question_number']}" if selected.get("question_number")
                      else "the target specified in the original student message"),
        )
    elif session is None and exact_id is not None:
        system_prompt += "\n\n" + EXACT_REFERENCE_PROMPT.format(
            exercise=exact_id,
            question=(f"Question {exact_question}" if exact_question is not None
                      else "the target specified in the original student message"),
        )

    answer = call_tutor(
        query=query, context=context, system_prompt=system_prompt,
        **({"history": session.get("history", [])} if session is not None else {}),
    )
    if session is not None:
        session["history"] = [
            *session.get("history", []),
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        if source == "RESTORE_PREVIOUS":
            session["active"], session["previous"] = selected, active
        else:
            if selected is not None and (active is None or
                selected.get("parent_ids") != active.get("parent_ids") or
                selected.get("kind") != active.get("kind")):
                session["previous"] = active
            session["active"] = selected
        session["last_mode"] = decision["mode"]
        save_session(session_id, session)  # Only after successful Luna generation.
        decision["selected_context_source"] = source
    return answer, decision

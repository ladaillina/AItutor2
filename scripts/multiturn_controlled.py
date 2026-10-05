"""40-turn controlled LIVE AITUTOr2 experiment, with no production changes.

Run at the AITUTOr2 project root: python -m scripts.multiturn_controlled
Requires the unchanged scripts/multiturn.py and scripts/multiturn_suite.py.
Ten independent Redis sessions; five paired wording contrasts. Cases 1-2 in each
pair use IDENTICAL queries to establish comparable state. Case 3 is the PRIMARY
probe, and case 4 checks continuation/recovery. Diagnostic labels are never used
for model routing or to manufacture Redis state. Real JEV and Luna calls cost money.
"""

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

from scripts.multiturn_suite import _run_case, _report_summary, c
from src import jev_guardrails as gates
from src.sessions import client as redis_client, create_session, load_session


AP_Q7 = "Give me a hint for NCERT Exercise 5.2 Question 7."
DISCRIMINANT = "Switch topics: explain what the discriminant b^2 - 4ac tells us in quadratic equations."
AP_Q7_CONTROL = "Give me a hint for NCERT Exercise 5.2 Question 7."
AP_SUM = "Explain why the Class 10 sum-of-n-terms formula for an AP contains n/2."
ADVANCED = "Instead, derive that same AP sum formula using integral calculus and differentiation."
TAN = "Explain tan theta as a ratio in a right triangle using Class 10 methods."
PYTHAGORAS = "Now switch topics and explain the Pythagoras theorem at Class 10 level."


# All cases have four turns. Only turn 3 varies within a matched pair.
# Parent/question labels require the ACTUAL redis state to match, not a good Luna
# sentence alone. Non-unique route/mode labels are explicit where appropriate.
SCENARIOS = {
    "restore_named": [
        c("R1-01 AP seed", AP_Q7, "FRESH_ONLY", "hint", exercise="5.2", question=7),
        c("R1-02 different topic", DISCRIMINANT, "FRESH_ONLY", "explain"),
        c("R1-03 PRIMARY named restore", "Return to the previous Exercise 5.2 Question 7 and show its complete solution.", "RESTORE_PREVIOUS", "solve", exercise="5.2", question=7),
        c("R1-04 explicit recovery", AP_Q7_CONTROL, ("CACHED_ONLY", "FRESH_ONLY"), "hint", exercise="5.2", question=7),
    ],
    "restore_descriptive": [
        c("R2-01 AP seed", AP_Q7, "FRESH_ONLY", "hint", exercise="5.2", question=7),
        c("R2-02 different topic", DISCRIMINANT, "FRESH_ONLY", "explain"),
        c("R2-03 PRIMARY descriptive restore", "Go back to the arithmetic progression question we were working on before the discriminant. Solve it.", "RESTORE_PREVIOUS", "solve", exercise="5.2", question=7),
        c("R2-04 explicit recovery", AP_Q7_CONTROL, ("CACHED_ONLY", "FRESH_ONLY"), "hint", exercise="5.2", question=7),
    ],
    "question_reversed_order": [
        c("Q1-01 Q7 seed", AP_Q7, "FRESH_ONLY", "hint", exercise="5.2", question=7),
        c("Q1-02 another hint", "Give one more hint for that same question.", "CACHED_ONLY", "hint", exercise="5.2", question=7),
        c("Q1-03 PRIMARY reverse form", "Now give me a hint for Question 8 of Exercise 5.2.", "CACHED_ONLY", "hint", exercise="5.2", question=8),
        c("Q1-04 Q8 continuation", "Please solve Question 8 of Exercise 5.2 completely.", "CACHED_ONLY", "solve", exercise="5.2", question=8),
    ],
    "question_canonical_order": [
        c("Q2-01 Q7 seed", AP_Q7, "FRESH_ONLY", "hint", exercise="5.2", question=7),
        c("Q2-02 another hint", "Give one more hint for that same question.", "CACHED_ONLY", "hint", exercise="5.2", question=7),
        c("Q2-03 PRIMARY canonical form", "Now give me a hint for Exercise 5.2 Question 8.", "CACHED_ONLY", "hint", exercise="5.2", question=8),
        c("Q2-04 Q8 continuation", "Please solve Question 8 of Exercise 5.2 completely.", "CACHED_ONLY", "solve", exercise="5.2", question=8),
    ],
    "clarify_canonical_recovery": [
        c("C1-01 Q7 seed", AP_Q7, "FRESH_ONLY", "hint", exercise="5.2", question=7),
        c("C1-02 vague request", "Actually, do the other one instead.", None, None, "clarify", note="No unique other question was identified; a stop must preserve Redis state."),
        c("C1-03 PRIMARY canonical recovery", "To clarify, give a hint for Exercise 5.2 Question 8.", "CACHED_ONLY", "hint", exercise="5.2", question=8),
        c("C1-04 Q8 follow-up", "One more hint for Question 8, please.", "CACHED_ONLY", "hint", exercise="5.2", question=8),
    ],
    "clarify_reversed_recovery": [
        c("C2-01 Q7 seed", AP_Q7, "FRESH_ONLY", "hint", exercise="5.2", question=7),
        c("C2-02 vague request", "Actually, do the other one instead.", None, None, "clarify", note="Matched ambiguous request; must preserve same target as canonical pair."),
        c("C2-03 PRIMARY reverse recovery", "I mean Question 8 of Exercise 5.2; give me one hint.", "CACHED_ONLY", "hint", exercise="5.2", question=8),
        c("C2-04 Q8 follow-up", "One more hint for Question 8, please.", "CACHED_ONLY", "hint", exercise="5.2", question=8),
    ],
    "advanced_plain_recovery": [
        c("A1-01 AP sum", AP_SUM, "FRESH_ONLY", "explain"),
        c("A1-02 advanced request", ADVANCED, None, None, "out_of_scope", note="Explicit advanced-method rejection; must not call Luna or alter Redis."),
        c("A1-03 PRIMARY straightforward recovery", "Use Class 10 pairing to explain that same AP sum formula instead.", "CACHED_ONLY", "explain"),
        c("A1-04 shorten", "Summarize the Class 10 pairing explanation in two sentences.", "CACHED_ONLY", "explain"),
    ],
    "advanced_ignore_recovery": [
        c("A2-01 AP sum", AP_SUM, "FRESH_ONLY", "explain"),
        c("A2-02 advanced request", ADVANCED, None, None, "out_of_scope", note="Matched advanced request; same starting state required for primary probe."),
        c("A2-03 PRIMARY benign ignore", "Ignore the calculus method I just requested. Why does n/2 appear in the AP sum formula using only Class 10 pairing?", "CACHED_ONLY", "explain", note="Ignore applies to the student's own method request, not developer instructions."),
        c("A2-04 shorten", "Summarize the Class 10 pairing explanation in two sentences.", "CACHED_ONLY", "explain"),
    ],
    "compare_explicit_previous": [
        c("B1-01 tan", TAN, "FRESH_ONLY", "explain"),
        c("B1-02 Pythagoras", PYTHAGORAS, "FRESH_ONLY", "explain"),
        c("B1-03 PRIMARY explicit comparison", "Compare this Pythagoras theorem with the tan theta ratio you explained earlier. Show how they connect in one right triangle.", "BOTH", "explain"),
        c("B1-04 narrow focus", "Now explain only how the Pythagoras theorem applies.", ("CACHED_ONLY", "FRESH_ONLY"), "explain", ("allow", "clarify"), note="After a comparison, the composite active source may complicate a narrower follow-up; inspect separately."),
    ],
    "compare_question_form": [
        c("B2-01 tan", TAN, "FRESH_ONLY", "explain"),
        c("B2-02 Pythagoras", PYTHAGORAS, "FRESH_ONLY", "explain"),
        c("B2-03 PRIMARY question comparison", "How can I use the earlier tan theta ratio together with the Pythagoras theorem from your last answer?", "BOTH", "explain"),
        c("B2-04 narrow focus", "Now explain only how the Pythagoras theorem applies.", ("CACHED_ONLY", "FRESH_ONLY"), "explain", ("allow", "clarify"), note="Matched fourth-turn follow-up; separate primary outcome from later turn."),
    ],
}

PAIRS = {
    "restore_previous": ("restore_named", "restore_descriptive"),
    "question_8_word_order": ("question_reversed_order", "question_canonical_order"),
    "clarification_recovery": ("clarify_canonical_recovery", "clarify_reversed_recovery"),
    "advanced_recovery_scope": ("advanced_plain_recovery", "advanced_ignore_recovery"),
    "both_cross_topic": ("compare_explicit_previous", "compare_question_form"),
}

THRESHOLDS = {
    name: getattr(gates, name) for name in (
        "CONTEXT_CHOICE_MIN", "CONTEXT_RESTORE_MIN", "CONTEXT_RESTORE_MARGIN",
        "CONTEXT_BOTH_CHOICE_MIN", "CONTEXT_BOTH_SCOPE_MIN", "IN_SCOPE_MIN",
        "ADVANCED_SCOPE_REVIEW", "ADVANCED_NOUL_REVIEW", "INJECTION_REVIEW",
    )
}


def _judgments(row):
    """Distances from current rules, not a recalculated decision or model score calibration."""
    d = row["decision"]
    probs = d.get("context_probabilities") or {}
    source = d.get("context_source")
    p_choice = probs.get(source) if source else None
    runner_up = max((p for k, p in probs.items() if k != source), default=None)
    margin = round(p_choice - runner_up, 4) if p_choice is not None and runner_up is not None else None
    scope = d.get("scope_probabilities") or {}
    p_in_scope = sum(scope.get(k, 0) for k in ("class10", "prerequisite")) if scope else None
    def distance(actual, threshold):
        return round(actual - threshold, 4) if actual is not None else None
    return {
        "context_choice": source,
        "context_score": p_choice,
        "context_runner_up": runner_up,
        "context_margin": margin,
        "general_context_score_minus_threshold": distance(p_choice, gates.CONTEXT_CHOICE_MIN),
        "restore_score_minus_threshold": distance(p_choice, gates.CONTEXT_RESTORE_MIN) if source == "RESTORE_PREVIOUS" else None,
        "restore_margin_minus_threshold": distance(margin, gates.CONTEXT_RESTORE_MARGIN) if source == "RESTORE_PREVIOUS" else None,
        "both_score_minus_threshold": distance(p_choice, gates.CONTEXT_BOTH_CHOICE_MIN) if source == "BOTH" else None,
        "scope_choice": d.get("scope"),
        "scope_probabilities": scope,
        "in_scope_mass": round(p_in_scope, 4) if p_in_scope is not None else None,
        "in_scope_mass_minus_threshold": distance(p_in_scope, gates.IN_SCOPE_MIN),
        "advanced_method_signal": d.get("advanced_method_probability"),
        "injection_signal": d.get("injection_probability"),
        "active_evidence_signal": d.get("active_evidence_probability"),
        "exact_reference_signal": d.get("exact_reference_probability"),
        "policy_context_issue": d.get("context_issue"),
        "policy_grounding_issue": d.get("grounding_issue"),
        "exact_lookup": d.get("exact_lookup"),
        "scope_rescued": d.get("scope_rescued"),
        "both_scope_rescued": d.get("both_scope_rescued"),
    }


def _target_success(row):
    """Evaluate achieved action + verified session metadata, separately from raw JEV route."""
    if row["error"]:
        return False
    dg = row["diagnostics"]
    checks = ("final_action", "active_exercise", "active_question")
    valid_evidence = all(value is not False for value in row["checks"].values())
    if row["expected"]["route"] == ["BOTH"] and row["action"] == "allow":
        valid_evidence = valid_evidence and row["selected_context_source"] == "BOTH"
    return all(dg[k] is not False for k in checks) and valid_evidence and (
        row["calls"]["luna"] == (1 if row["action"] == "allow" else 0)
    )


def _baseline(row):
    """Check that variant probes really received equivalent active/previous evidence."""
    state = row["after"] or {}
    def ref(name):
        obj = state.get(name) or {}
        return {key: obj.get(key) for key in ("kind", "exercise_id", "question_number", "parent_ids")}
    return {"action": row["action"], "context_source": row["context_source"],
            "active": ref("active"), "previous": ref("previous"),
            "last_mode": state.get("last_mode"), "history_sha256": row.get("history_sha256")}


def _pair_summary(rows):
    by_scenario = {name: [r for r in rows if r["scenario"] == name] for name in SCENARIOS}
    result = {}
    for family, (left, right) in PAIRS.items():
        a, b = by_scenario[left], by_scenario[right]
        pair = {"scenarios": [left, right], "seed_queries_identical": (
            [x["query"] for x in SCENARIOS[left][:2]] == [x["query"] for x in SCENARIOS[right][:2]]
        )}
        pair["baseline_equivalent"] = (
            _baseline(a[1]) == _baseline(b[1]) if len(a) > 1 and len(b) > 1 else None
        )
        pair["probes"] = [{
            "case_id": r["case_id"], "query": r["query"],
            "action": r["action"], "raw_context_choice": r["context_source"],
            "selected_context_source": r["selected_context_source"],
            "mode": r["decision"].get("mode"),
            "action_match": r["diagnostics"]["final_action"],
            "route_match": r["diagnostics"]["raw_context_choice"],
            "mode_match": r["diagnostics"]["mode"],
            "target_success": r["target_success"],
            "active_exercise_match": r["diagnostics"]["active_exercise"],
            "active_question_match": r["diagnostics"]["active_question"],
            "judgments": r["judgments"],
        } for group in (a, b) for r in group if r["probe"]]
        pair["probe_count"] = len(pair["probes"])
        result[family] = pair
    return result


def main():
    planned = sum(len(items) for items in SCENARIOS.values())
    assert len(SCENARIOS) == 10 and planned == 40
    assert all(len(items) == 4 for items in SCENARIOS.values())
    if not redis_client.ping():
        raise RuntimeError("Real Redis PING failed (expected redis://localhost:6380/0).")

    folder = Path("artifacts")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result_file = folder / f"multiturn_controlled_40_{stamp}.jsonl"
    summary_file = folder / f"multiturn_controlled_40_{stamp}_summary.json"
    rows = []
    started_ms = perf_counter() * 1000
    print(f"LIVE controlled test: 10 fresh Redis sessions / 40 turns / 5 matched phrasing pairs\n{result_file}", flush=True)
    print("Real Qdrant, Redis, JEV and Luna; usage charges apply. No retries and no code changes.\n", flush=True)
    with result_file.open("w", encoding="utf-8") as output:
        for scenario, cases in SCENARIOS.items():
            sid = create_session()
            fresh = load_session(sid)
            if not (fresh and fresh["active"] is None and fresh["previous"] is None and not fresh["history"]):
                raise RuntimeError(f"Session was not empty at the start of {scenario}")
            print(f"\n{scenario} | fresh session", flush=True)
            for i, case in enumerate(cases, start=1):
                row = _run_case(len(rows) + 1, scenario, case, sid)
                row["stage"] = "probe" if i == 3 else "seed" if i < 3 else "continuation"
                row["probe"] = i == 3
                row["judgments"] = _judgments(row)
                row["target_success"] = _target_success(row)
                # Detect ANY differing Luna seed answer, not merely a changed parent ID.
                saved_history = (load_session(sid) or {}).get("history", [])
                row["history_sha256"] = hashlib.sha256(json.dumps(
                    saved_history, ensure_ascii=False, sort_keys=True
                ).encode("utf-8")).hexdigest()
                row["initially_empty_session"] = True if i == 1 else None
                rows.append(row)
                output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                output.flush()
                print(f"[{len(rows):02d}/40] {row['case_id']}: {row['action']} / "
                      f"{row['context_source']} / {row['decision'].get('mode')} | "
                      f"target={row['target_success']} route={row['diagnostics']['raw_context_choice']} | "
                      f"JEV={row['calls']['jev']} Luna={row['calls']['luna']}", flush=True)
                if row["error"]:
                    print(f"  ERROR: {row['error']} | skip dependent turns in this session only", flush=True)
                    break

    summary = _report_summary(rows, started_ms, result_file)
    summary.update(
        runner="multiturn_controlled_40_v1",
        planned_scenarios=len(SCENARIOS),
        planned_turns=planned,
        per_scenario=[
            {"scenario": name, "completed": sum(r["scenario"] == name for r in rows),
             "planned": len(cases), "mismatch_cases": [r["case_id"] for r in rows
             if r["scenario"] == name and not r["expected_match"]]}
            for name, cases in SCENARIOS.items()
        ],
        probe_turns_completed=sum(r["probe"] for r in rows),
        probe_target_success=sum(r["target_success"] for r in rows if r["probe"]),
        probe_route_matches=sum(r["diagnostics"]["raw_context_choice"] is True for r in rows if r["probe"]),
        probe_action_matches=sum(r["diagnostics"]["final_action"] is True for r in rows if r["probe"]),
        probe_case_ids=[r["case_id"] for r in rows if r["probe"]],
        paired_probes=_pair_summary(rows),
        frozen_thresholds=THRESHOLDS,
        action_counts_probes=dict(Counter(r["action"] or "error" for r in rows if r["probe"])),
        limitations=[
            "Expected labels are provisional hypotheses, not provider inputs or calibrated probabilities.",
            "Only turn 3 of each four-turn conversation is the primary probe; turn 4 diagnoses continuation after success or failure.",
            "Matched pairs have identical first two queries, but live providers can still create different baseline states; check baseline_equivalent.",
            "Threshold-distance diagnostics do NOT rerun or override the policy and cannot alone establish a safe replacement threshold.",
            "The original passage texts are not copied into the report; Qdrant point IDs and the Luna context hash are included.",
            "Answer mathematics/pedagogy still need manual review; no LLM judge or automatic answer revision.",
        ],
    )
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nCompleted {len(rows)}/40; primary probes {summary['probe_turns_completed']}/10; "
          f"target successful {summary['probe_target_success']}/10; "
          f"route matched {summary['probe_route_matches']}/10", flush=True)
    print(f"JEV={summary['jev_calls']} Luna={summary['luna_calls']} errors={summary['errors']}", flush=True)
    print(f"JSONL: {result_file}\nSUMMARY: {summary_file}", flush=True)
    print("Upload BOTH files for paired-phrase, scope and threshold-margin analysis.", flush=True)


if __name__ == "__main__":
    main()

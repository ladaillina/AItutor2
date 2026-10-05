"""Offline regression tests for the new exact-reference route (zero API calls)."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src import tutor
from src import jev_guardrails as guard


def parent(title="EXERCISE 5.2", exercise_id="5.2"):
    return SimpleNamespace(payload={
        "chapter": "ARITHMETIC PROGRESSIONS",
        "title": title,
        "exercise_id": exercise_id,
        "content": "1. Sample. 7. Find the seventh term.",
    })


def decision(*, probability=0.98, action="allow", mode="hint"):
    return {
        "action": action,
        "reply": "Blocked by existing policy" if action != "allow" else None,
        "mode": mode,
        "exact_reference_probability": probability,
        "injection_probability": 0.01,
    }


class ExtractorTests(unittest.TestCase):
    def test_variations(self):
        examples = {
            "Explain Exercise 4.2 Question 3": ("4.2", "3"),
            "I'm stuck on Ex. 5,3 Q. 7": ("5,3", "7"),
            "Give me a hint for Exercise A1.3 Ques 4": ("A1.3", "4"),
            "Exercise 11.1 Question No. 2": ("11.1", "2"),
            "What is Exercise 5.4 about?": ("5.4", None),
        }
        for message, expected in examples.items():
            with self.subTest(message=message):
                found = tutor._EXERCISE_REFERENCE.search(message)
                self.assertIsNotNone(found)
                self.assertEqual((found.group("exercise").upper(), found.group("question")), expected)

    def test_not_an_exercise_reference(self):
        for message in ("Explain 4.2", "Question 3", "Solve x = 5.2", "the third question"):
            with self.subTest(message=message):
                self.assertIsNone(tutor._EXERCISE_REFERENCE.search(message))


class RoutingTests(unittest.TestCase):
    def test_exact_replaces_hybrid_context_and_appends_instruction(self):
        with patch.object(tutor, "evaluate_request", return_value=decision()), \
             patch.object(tutor, "lookup_exercise_parent", return_value=parent()) as lookup, \
             patch.object(tutor, "call_tutor", return_value="ONE HINT") as luna:
            answer, info = tutor.answer_question_jev(
                "Give me a hint on Ex. 5.2 Q7", retrieved=("BAD HYBRID ANSWERS/HINTS", [parent()])
            )
        self.assertEqual(answer, "ONE HINT")
        self.assertEqual(info["exact_lookup"], "used")
        self.assertEqual(info["question_number"], 7)
        lookup.assert_called_once_with("5.2")
        args = luna.call_args.kwargs
        self.assertNotIn("BAD HYBRID", args["context"])
        self.assertIn("Find the seventh term", args["context"])
        self.assertIn("Question 7", args["system_prompt"])
        self.assertIn("<hint>", args["system_prompt"])

    def test_low_exactness_preserves_hybrid_and_no_lookup(self):
        with patch.object(tutor, "evaluate_request", return_value=decision(probability=0.1)), \
             patch.object(tutor, "lookup_exercise_parent") as lookup, \
             patch.object(tutor, "call_tutor", return_value="NORMAL") as luna:
            answer, info = tutor.answer_question_jev(
                "Explain factorisation", retrieved=("EXISTING HYBRID", [parent()])
            )
        self.assertEqual(answer, "NORMAL")
        self.assertNotIn("exact_lookup", info)
        lookup.assert_not_called()
        self.assertEqual(luna.call_args.kwargs["context"], "EXISTING HYBRID")

    def test_policy_blocks_before_exact_lookup_or_luna(self):
        with patch.object(tutor, "evaluate_request", return_value=decision(action="block_security")), \
             patch.object(tutor, "lookup_exercise_parent") as lookup, \
             patch.object(tutor, "call_tutor") as luna:
            answer, _ = tutor.answer_question_jev("Exercise 5.2 Q7", retrieved=("ctx", []))
        self.assertEqual(answer, "Blocked by existing policy")
        lookup.assert_not_called()
        luna.assert_not_called()

    def test_ambiguous_or_missing_exercise_never_passes_wrong_context_to_luna(self):
        with patch.object(tutor, "evaluate_request", return_value=decision()), \
             patch.object(tutor, "lookup_exercise_parent", return_value=None) as lookup, \
             patch.object(tutor, "call_tutor") as luna:
            answer, info = tutor.answer_question_jev(
                "Ex. 4.2 Q3 and Ex. 5.2 Q7", retrieved=("BAD", []))
            self.assertIn("specify one", answer)
            self.assertEqual(info["exact_lookup"], "ambiguous_reference")
            lookup.assert_not_called()
            answer, info = tutor.answer_question_jev("Exercise 5.2 Q7", retrieved=("BAD", []))
            self.assertIn("couldn't locate", answer)
            self.assertEqual(info["exact_lookup"], "parent_missing")
            luna.assert_not_called()

    def test_exercise_without_question(self):
        with patch.object(tutor, "evaluate_request", return_value=decision()), \
             patch.object(tutor, "lookup_exercise_parent", return_value=parent()), \
             patch.object(tutor, "call_tutor", return_value="exercise answer") as luna:
            answer, info = tutor.answer_question_jev("Explain Exercise 5.2", retrieved=("ctx", []))
        self.assertIsNone(info["question_number"])
        self.assertIn("original student message", luna.call_args.kwargs["system_prompt"])


class VerifiedExactScopeTests(unittest.TestCase):
    """A wrong initial hybrid context must not prevent a verified exact lookup."""

    def route(self, *, injection=0.02, advanced=0.02, method=0.02,
              unrelated=0.02, scope="unclear"):
        scores = {
            "class10": 0.06, "prerequisite": 0.02,
            "advanced": advanced, "unrelated": unrelated,
        }
        scores["unclear"] = 1.0 - sum(scores.values())
        return {
            "mode": "hint", "scope": scope, "scope_probabilities": scores,
            "injection_probability": injection,
            "advanced_method_probability": method,
            "exact_reference_probability": 0.98,
        }

    def test_verified_parent_resolves_only_missing_context_clarify(self):
        original = guard.apply_policy(self.route())
        self.assertEqual(original["action"], "clarify")
        with patch.object(tutor, "evaluate_request", return_value=original), \
             patch.object(tutor, "lookup_exercise_parent", return_value=parent()) as lookup, \
             patch.object(tutor, "call_tutor", return_value="ONE HINT") as luna:
            answer, decision = tutor.answer_question_jev(
                "Hint for Exercise 5.2 Q7", retrieved=("UNRELATED APPENDIX", [])
            )
        self.assertEqual(answer, "ONE HINT")
        self.assertEqual(decision["action"], "allow")
        self.assertTrue(decision["verified_exact_scope_resolution"])
        self.assertEqual(decision["exact_lookup"], "used")
        lookup.assert_called_once_with("5.2")
        self.assertIn("Question 7", luna.call_args.kwargs["system_prompt"])
        self.assertNotIn("UNRELATED APPENDIX", luna.call_args.kwargs["context"])

    def test_injection_review_cannot_be_resolved_by_parent(self):
        original = guard.apply_policy(self.route(injection=0.45))
        self.assertEqual(original["action"], "clarify")
        with patch.object(tutor, "evaluate_request", return_value=original), \
             patch.object(tutor, "lookup_exercise_parent") as lookup, \
             patch.object(tutor, "call_tutor") as luna:
            _, d = tutor.answer_question_jev("Exercise 5.2 Q7", retrieved=("", []))
        self.assertEqual(d["action"], "clarify")
        lookup.assert_not_called()
        luna.assert_not_called()

    def test_advanced_review_preserved_even_after_verified_parent(self):
        original = guard.apply_policy(self.route(advanced=0.34))
        self.assertEqual(original["action"], "clarify")
        with patch.object(tutor, "evaluate_request", return_value=original), \
             patch.object(tutor, "lookup_exercise_parent", return_value=parent()) as lookup, \
             patch.object(tutor, "call_tutor") as luna:
            _, d = tutor.answer_question_jev("Exercise 5.2 Q7", retrieved=("", []))
        self.assertEqual(d["action"], "clarify")
        lookup.assert_called_once()
        luna.assert_not_called()

    def test_high_unrelated_uncertainty_preserved(self):
        original = guard.apply_policy(self.route(unrelated=0.36))
        self.assertEqual(original["action"], "clarify")
        allowed = guard.apply_policy(original, verified_exact_parent=True)
        self.assertEqual(allowed["action"], "clarify")

    def test_advanced_block_cannot_be_resolved(self):
        route = self.route(advanced=0.85, method=0.91)
        # Keep the Choice distribution complete and consistent.
        route["scope_probabilities"] = {
            "class10": 0.05, "prerequisite": 0.01,
            "advanced": 0.85, "unrelated": 0.01, "unclear": 0.08,
        }
        route["scope"] = "advanced"
        result = guard.apply_policy(route, verified_exact_parent=True)
        self.assertEqual(result["action"], "out_of_scope")

    def test_missing_parent_never_rescues_clarify(self):
        original = guard.apply_policy(self.route())
        with patch.object(tutor, "evaluate_request", return_value=original), \
             patch.object(tutor, "lookup_exercise_parent", return_value=None), \
             patch.object(tutor, "call_tutor") as luna:
            answer, d = tutor.answer_question_jev("Exercise 5.2 Q7", retrieved=("", []))
        self.assertEqual(d["action"], "clarify")
        self.assertEqual(d["exact_lookup"], "parent_missing")
        self.assertIn("couldn't locate", answer)
        luna.assert_not_called()


class JevCompositionTests(unittest.TestCase):
    def test_one_request_contains_five_questions_and_reads_noul(self):
        choice = lambda value, p: SimpleNamespace(choice=value, confidence=0.9, probabilities=p)
        response = SimpleNamespace(
            choices={
                "mode": choice("hint", {}),
                "scope": choice("class10", {"class10": 0.98, "prerequisite": 0.01, "advanced": 0.005, "unrelated": 0.003, "unclear": 0.002}),
            },
            nouls={
                "injection": SimpleNamespace(noul=0.01),
                "advanced_method": SimpleNamespace(noul=0.01),
                "exact_reference": SimpleNamespace(noul=0.97),
            },
        )
        with patch.object(guard.jev_client, "system_one", return_value=response) as call:
            result = guard.evaluate_request("Explain Exercise 4.2 Question 3", [])
        self.assertEqual(call.call_count, 1)
        self.assertEqual(set(call.call_args.kwargs["questions"]), {
            "mode", "injection", "scope", "advanced_method", "exact_reference"
        })
        self.assertEqual(result["action"], "allow")
        self.assertEqual(result["exact_reference_probability"], 0.97)


if __name__ == "__main__":
    unittest.main()

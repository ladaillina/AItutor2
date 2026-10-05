"""Complete isolated tests for src/sessions.py; no running Redis required."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.expiries = {}

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.expiries[key] = ex
        return True

    def get(self, key):
        return self.data.get(key)

    def delete(self, key):
        self.expiries.pop(key, None)
        return int(self.data.pop(key, None) is not None)


class ChatSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = FakeRedis()
        redis_stub = ModuleType("redis")
        redis_stub.Redis = SimpleNamespace(from_url=lambda *args, **kwargs: cls.fake)
        spec = importlib.util.spec_from_file_location(
            "tested_sessions", Path(__file__).resolve().parents[1] / "src" / "sessions.py"
        )
        cls.store = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"redis": redis_stub}):
            spec.loader.exec_module(cls.store)

    def setUp(self):
        self.fake.data.clear()
        self.fake.expiries.clear()

    def test_create_and_expire_contract(self):
        session_id = self.store.create_session()
        key = f"aitutor2:chat:{session_id}"
        self.assertEqual(self.fake.expiries[key], 3600)
        self.assertEqual(
            self.store.load_session(session_id),
            {"history": [], "active": None, "previous": None, "last_mode": None},
        )

    def test_one_key_full_parent_round_trip(self):
        session_id = self.store.create_session()
        active = {"kind": "exercise", "exercise_id": "5.2", "question_number": "7",
                  "parent_id": "verified-id", "chapter": "Arithmetic Progressions",
                  "title": "Exercise 5.2", "parent_text": "aₙ = a + (n − 1)d"}
        previous = {"parent_id": "older-verified-id", "parent_text": "b² − 4ac"}
        session = {"history": [{"role": "user", "content": "Hint please"}],
                   "active": active, "previous": previous, "last_mode": "hint"}
        self.store.save_session(session_id, session)
        self.assertEqual(self.store.load_session(session_id), session)
        self.assertEqual(len(self.fake.data), 1)
        self.assertEqual(self.fake.expiries[f"aitutor2:chat:{session_id}"], 3600)

    def test_keep_last_four_exchanges_without_mutating_input(self):
        session_id = self.store.create_session()
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)}
                   for i in range(12)]
        self.store.save_session(session_id, {"history": history})
        self.assertEqual([m["content"] for m in self.store.load_session(session_id)["history"]],
                         [str(i) for i in range(4, 12)])
        self.assertEqual(len(history), 12)

    def test_session_isolation_deletion_and_expiry(self):
        first, second = self.store.create_session(), self.store.create_session()
        self.store.save_session(first, {"history": [{"role": "user", "content": "first"}]})
        self.assertEqual(self.store.load_session(second)["history"], [])
        self.store.delete_session(first)
        self.assertIsNone(self.store.load_session(first))
        self.assertIsNotNone(self.store.load_session(second))
        self.fake.data.pop(f"aitutor2:chat:{second}")  # Simulate Redis TTL eviction.
        self.assertIsNone(self.store.load_session(second))

    def test_invalid_ids_and_storage_errors_fail_closed(self):
        with self.assertRaises(ValueError):
            self.store.load_session("not-a-uuid")
        with self.assertRaises(ValueError):
            self.store.load_session("00000000-0000-1000-8000-000000000000")
        session_id = str(uuid4())
        with patch.object(self.fake, "get", side_effect=RuntimeError("Redis offline")):
            with self.assertRaisesRegex(RuntimeError, "Redis offline"):
                self.store.load_session(session_id)
        with patch.object(self.fake, "set", side_effect=RuntimeError("Redis offline")):
            with self.assertRaisesRegex(RuntimeError, "Redis offline"):
                self.store.save_session(session_id, {"history": []})


if __name__ == "__main__":
    unittest.main()

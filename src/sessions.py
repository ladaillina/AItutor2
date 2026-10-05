"""Ephemeral, one-key-per-chat storage for AITUTOr2. No retrieval or routing here."""

# [PYTHON STANDARD LIBRARY] Built-in JSON, environment and UUID functionality.
import json
import os
from uuid import UUID, uuid4

# [THIRD-PARTY LIBRARIES] Existing documented APIs; no implementation of Redis here.
from dotenv import load_dotenv
from redis import Redis

# [LIBRARY CALL] python-dotenv reads optional local .env settings.
load_dotenv()

# [CUSTOM / PROJECT CONFIGURATION] Our chosen time-to-live and history size.
TTL_SECONDS = 3600
MAX_HISTORY_MESSAGES = 8  # Four student/assistant exchanges.

# [LIBRARY API] Redis.from_url() + decode_responses are provided by redis-py.
# [CUSTOM / PROJECT CONFIGURATION] The REDIS_URL name and default port 6380 are ours.
client = Redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6380/0"), decode_responses=True
)


# [CUSTOM GLUE] Our session-key naming and UUID4 validation policy.
# [STANDARD LIBRARY USED] UUID() parses/validates the supplied identifier.
def _key(session_id: str) -> str:
    session_uuid = UUID(session_id)
    if session_uuid.version != 4:
        raise ValueError("Expected a UUID4 session ID")
    return f"aitutor2:chat:{session_uuid.hex}"


# [CUSTOM GLUE] Our initial chat shape; UUID generation is from Python stdlib.
def create_session() -> str:
    session_id = uuid4().hex
    save_session(session_id, {"history": [], "active": None, "previous": None, "last_mode": None})
    return session_id


# [CUSTOM GLUE] Interpret an absent/expired key as None and JSON-decode the value.
# [LIBRARY API] client.get() is redis-py's wrapper around Redis GET.
# [STANDARD LIBRARY] json.loads() decodes the stored string.
def load_session(session_id: str) -> dict | None:
    raw = client.get(_key(session_id))
    return json.loads(raw) if raw is not None else None  # None means expired/unknown.


# [CUSTOM GLUE] Choose the fields and retain the last eight messages.
# [STANDARD LIBRARY] json.dumps() serializes the record.
# [LIBRARY API] client.set(..., ex=TTL_SECONDS) is redis-py's Redis SET + EX expiry.
# No Python timer or custom eviction process is used; Redis expires the key.
def save_session(session_id: str, session: dict) -> None:
    snapshot = {
        "history": session.get("history", [])[-MAX_HISTORY_MESSAGES:],
        "active": session.get("active"),
        "previous": session.get("previous"),
        "last_mode": session.get("last_mode"),
    }
    client.set(_key(session_id), json.dumps(snapshot, ensure_ascii=False), ex=TTL_SECONDS)


# [CUSTOM GLUE] Expose deletion by session ID.
# [LIBRARY API] client.delete() is redis-py's wrapper around Redis DEL.
def delete_session(session_id: str) -> None:
    client.delete(_key(session_id))

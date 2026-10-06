import hashlib
import json

import pytest

from observatory.store import Store


def test_caller_intent_survives_restart_encrypted_and_never_public(tmp_path):
    path = tmp_path / "state.sqlite3"
    body = {"model": "openai/test", "input": "PRIVATE_PENDING_PROMPT_FIXTURE", "max_output_tokens": 200}
    digest = hashlib.sha256(json.dumps({"protocol": "responses", "body": body}, sort_keys=True).encode()).hexdigest()
    store = Store(path)
    request_id = store.reserve_research_request("agent-1", digest, "responses", body)
    assert store.reserve_research_request("agent-1", digest, "responses", body) == request_id
    assert "PRIVATE_PENDING_PROMPT_FIXTURE" not in json.dumps(store.state())
    store.close()
    assert b"PRIVATE_PENDING_PROMPT_FIXTURE" not in path.read_bytes()
    reopened = Store(path)
    try:
        pending = reopened.pending_research_requests("agent-1")
        assert pending[0]["body"] == body and pending[0]["request_id"] == request_id
        assert reopened.reserve_research_request("agent-1", digest, "responses", body) == request_id
        with pytest.raises(ValueError, match="outstanding"):
            reopened.reserve_research_request("agent-1", "f" * 64, "responses", {**body, "input": "different"})
        other = reopened.reserve_research_request("agent-2", digest, "responses", body)
        assert other != request_id
        reopened.complete_research_request(request_id)
        assert not reopened.pending_research_requests("agent-1")
        assert len(reopened.pending_research_requests()) == 1
    finally:
        reopened.close()

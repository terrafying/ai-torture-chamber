"""Thread-safe SQLite state and encrypted owner configuration."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from cryptography.fernet import Fernet


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def secret_field(key: str) -> bool:
    name = key.lower()
    return name in {"token", "authorization", "password", "credential", "credentials"} or name.endswith("_token") or any(
        part in name for part in ("api_key", "access_token", "hf_token", "secret", "password", "admin_token", "bearer_token", "private_key", "seed_phrase", "mnemonic")
    )


_PRIVATE_FIELDS = {
    "frame_url", "frame_path", "frame_base64", "frame_bytes", "cdp_url", "cdp_ws_url",
    "browser_url", "connect_url", "debugger_url", "session_url", "session_view_url",
    "live_url", "live_view_url", "control_url", "browser_control_url",
    "text", "fulltext", "raw_html", "html", "original_text", "synthetic_sft", "records",
    "payment_signature", "signed_payment", "payment_payload", "signed_transaction",
    "passage", "supporting_passage", "document_key", "expires_monotonic",
    "receipt", "quotes",
}
_PUBLIC_KINDS = ("agents", "sources", "notes", "datasets", "jobs", "checkpoints", "connections", "curation_reviews")


class Store:
    """One SQLite connection per Store; an RLock protects worker/thread access."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._fernet = Fernet(self._load_key())
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA busy_timeout=30000")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL, id TEXT NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY(kind, id)
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_payment_intents (
                request_id TEXT PRIMARY KEY, scope TEXT NOT NULL,
                body_hash TEXT NOT NULL, protocol TEXT NOT NULL,
                payload BLOB NOT NULL, created_at TEXT NOT NULL,
                UNIQUE(scope, body_hash)
            );
        """)
        self._db.commit()

    def _load_key(self) -> bytes:
        configured = os.environ.get("OBSERVATORY_SECRET_KEY")
        if configured:
            return configured.encode("ascii")
        key_path = self.path.with_suffix(self.path.suffix + ".key")
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return key_path.read_bytes().strip()
        key = Fernet.generate_key()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "nt":
                identity = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
                if not identity:
                    identity = getpass.getuser()
                subprocess.run(
                    ["icacls", str(key_path), "/inheritance:r", "/grant:r", f"{identity}:(F)", "SYSTEM:(F)"],
                    capture_output=True, check=True,
                )
            else:
                os.chmod(key_path, 0o600)
        except Exception:
            key_path.unlink(missing_ok=True)
            raise RuntimeError("Could not protect the generated secret key; configure OBSERVATORY_SECRET_KEY") from None
        return key

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def reserve_research_request(self, scope: str, body_hash: str, protocol: str, body: dict) -> str:
        """Retain the caller UUID and encrypted body across pass/process restarts."""
        if (not isinstance(scope, str) or not 1 <= len(scope) <= 200
                or not isinstance(body_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", body_hash)
                or protocol not in {"responses", "messages"} or not isinstance(body, dict)):
            raise ValueError("Invalid private research payment intent")
        with self._lock:
            existing = self._db.execute("SELECT request_id,body_hash FROM research_payment_intents WHERE scope=?", (scope,)).fetchall()
            matching = next((row for row in existing if row[1] == body_hash), None)
            if matching:
                return matching[0]
            if existing:
                raise ValueError("Review the outstanding research payment before using a new request body")
            request_id = str(uuid4())
            encrypted = self._fernet.encrypt(json.dumps(body, sort_keys=True, allow_nan=False).encode())
            self._db.execute("INSERT INTO research_payment_intents(request_id,scope,body_hash,protocol,payload,created_at) VALUES(?,?,?,?,?,?)",
                             (request_id, scope, body_hash, protocol, encrypted, utc_now()))
            self._db.commit()
            return request_id

    def pending_research_requests(self, scope: str | None = None) -> list[dict]:
        """Private owner/runtime data, never included in public state or events."""
        with self._lock:
            query = "SELECT request_id,scope,body_hash,protocol,payload,created_at FROM research_payment_intents"
            rows = self._db.execute(query + (" WHERE scope=?" if scope is not None else "") + " ORDER BY created_at", (scope,) if scope is not None else ()).fetchall()
        return [{"request_id": row[0], "scope": row[1], "body_hash": row[2], "protocol": row[3],
                 "body": json.loads(self._fernet.decrypt(row[4])), "created_at": row[5]} for row in rows]

    def complete_research_request(self, request_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM research_payment_intents WHERE request_id=?", (request_id,))
            self._db.commit()

    def get_settings(self, private: bool = False) -> dict:
        with self._lock:
            row = self._db.execute("SELECT payload FROM meta WHERE key='settings'").fetchone()
        settings = json.loads(self._fernet.decrypt(row[0].encode())) if row else {}
        if private:
            return settings
        result = self.sanitize(settings, strip_content=False)
        for key, value in settings.items():
            if secret_field(key):
                result[key] = "***" if value else ""
                result[key + "_configured"] = bool(value)
        return result

    def save_settings(self, updates: dict) -> dict:
        if not isinstance(updates, dict):
            raise ValueError("settings must be an object")
        from .settings import validate_research_updates
        updates = validate_research_updates(updates)
        with self._lock:
            settings = self.get_settings(private=True)
            for key, value in updates.items():
                if key.endswith("_configured") or (secret_field(key) and value in ("***", "[REDACTED]")):
                    continue
                if key.lower() in {"admin_token", "observatory_admin_token", "authorization", "observatory_secret_key", "observatory_payment_broker_token", "payment_broker_token", "x402_broker_secret_key", "x402_solana_private_key", "wallet_private_key", "spending_private_key", "wallet_seed_phrase", "mnemonic"} or any(part in key.lower() for part in ("private_key", "seed_phrase", "mnemonic")):
                    raise ValueError("Owner and payment signing credentials are environment configuration, not saved settings")
                settings[key] = value
            encrypted = self._fernet.encrypt(json.dumps(settings, sort_keys=True).encode()).decode()
            self._db.execute("INSERT OR REPLACE INTO meta(key,payload) VALUES('settings',?)", (encrypted,))
            self._db.commit()
        return self.get_settings()

    def put(self, kind: str, item: dict) -> dict:
        if not isinstance(item, dict):
            raise ValueError("record must be an object")
        record = dict(item)
        record.setdefault("id", str(uuid4()))
        if not record["id"]:
            raise ValueError("record id must be nonempty")
        with self._lock:
            old = self.get(kind, str(record["id"]))
            if kind in {"datasets", "curation_reviews"} and old and old.get("immutable") and record != old:
                raise ValueError("Dataset snapshots and curation receipts are immutable; create a new record")
            record.setdefault("created_at", old.get("created_at", utc_now()) if old else utc_now())
            record["updated_at"] = utc_now()
            self._db.execute(
                "INSERT INTO records(kind,id,payload,created_at,updated_at) VALUES(?,?,?,?,?) "
                "ON CONFLICT(kind,id) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at",
                (kind, str(record["id"]), json.dumps(record, sort_keys=True), record["created_at"], record["updated_at"]),
            )
            self._db.commit()
        return record

    def get(self, kind: str, record_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM records WHERE kind=? AND id=?", (kind, str(record_id))).fetchone()
        return json.loads(row[0]) if row else None

    def list_records(self, kind: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT payload FROM records WHERE kind=? ORDER BY created_at,id", (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get_mission(self) -> dict:
        with self._lock:
            row = self._db.execute("SELECT payload FROM meta WHERE key='mission'").fetchone()
        return json.loads(row[0]) if row else {"id": None, "status": "stopped", "until_stopped": True}

    def set_mission(self, mission: dict) -> dict:
        payload = dict(mission)
        payload.setdefault("id", str(uuid4()))
        payload["updated_at"] = utc_now()
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta(key,payload) VALUES('mission',?)", (json.dumps(payload, sort_keys=True),))
            self._db.commit()
        return payload

    def sanitize(self, value: Any, strip_content: bool = True) -> Any:
        """Redact both structured credentials and credentials accidentally emitted in prose."""
        private = self.get_settings(private=True)
        secrets: list[str] = []

        def collect(data: Any) -> None:
            if isinstance(data, dict):
                for key, item in data.items():
                    if secret_field(key) and isinstance(item, str) and len(item) >= 4:
                        secrets.append(item)
                    else:
                        collect(item)
        collect(private)
        for name in ("OBSERVATORY_ADMIN_TOKEN", "OBSERVATORY_SECRET_KEY", "OBSERVATORY_PAYMENT_BROKER_TOKEN", "X402_SOLANA_PRIVATE_KEY", "X402_BROKER_SECRET_KEY"):
            env_secret = os.environ.get(name)
            if env_secret:
                secrets.append(env_secret)

        def clean(item: Any) -> Any:
            if isinstance(item, dict):
                return {
                    key: ("[REDACTED]" if secret_field(key) else clean(child))
                    for key, child in item.items()
                    if not (key.lower() in _PRIVATE_FIELDS and (strip_content or key.lower() != "text"))
                }
            if isinstance(item, list):
                return [clean(child) for child in item]
            if isinstance(item, str):
                for secret in secrets:
                    item = item.replace(secret, "[REDACTED]")
                item = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", item)
                item = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,})\b", "[REDACTED]", item)
                item = re.sub(r"(?i)([?&](?:api_key|token|key|secret|authorization)=)[^&\s]+", r"\1[REDACTED]", item)
                item = re.sub(r"(?:wss?://[^\s]+|https?://[^\s]*(?:devtools/browser|/cdp)[^\s]*)", "[PRIVATE BROWSER CONNECTION]", item)
                return item
            return item
        return clean(value)

    def event(self, event_type: str, message: str, agent_id: str | None = None,
              run_id: str | None = None, data: Any = None) -> dict:
        payload = self.sanitize({
            "type": event_type, "message": message, "agent_id": agent_id,
            "run_id": run_id, "data": data, "created_at": utc_now(),
        })
        with self._lock:
            cursor = self._db.execute("INSERT INTO events(payload) VALUES(?)", (json.dumps(payload, sort_keys=True),))
            seq = int(cursor.lastrowid)
            self._db.commit()
        return {**payload, "seq": seq, "id": seq}

    def events_after(self, after: int = 0, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT seq,payload FROM events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit)).fetchall()
        return [self.sanitize({**json.loads(payload), "seq": seq, "id": seq}) for seq, payload in rows]

    def state(self) -> dict:
        with self._lock:
            last = self._db.execute("SELECT COALESCE(MAX(seq),0) FROM events").fetchone()[0]
        result = {kind: self.sanitize(self.list_records(kind)) for kind in _PUBLIC_KINDS}
        result["jobs"] += self.sanitize(self.list_records("training_runs"))
        # Notes retain their user-facing text, without exposing raw source content.
        result["notes"] = self.sanitize(self.list_records("notes"), strip_content=False)
        # Publish only receipt summaries. Detailed reviewer output and supporting
        # quotes stay in the private source-bound audit record.
        result["curation_reviews"] = [self.sanitize({key: value for key, value in review.items()
                                                   if key != "reviews"})
                                     for review in self.list_records("curation_reviews")]
        work = self.get("curation_work", "automatic-curation") or {}
        result["curation_runtime"] = {key: work.get(key) for key in
                                     ("status", "call_state", "active_stage", "review_id", "source_id")}
        result.update(mission=self.sanitize(self.get_mission()), settings=self.get_settings(),
                      events=self.events_after(max(0, last - 100)), cursor=last)
        return result

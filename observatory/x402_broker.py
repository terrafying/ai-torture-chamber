"""Isolated, opt-in Solana x402 buyer. Importing this module never makes a payment.

Only this process holds X402_SOLANA_PRIVATE_KEY. The researcher supplies native
vendor requests, never URLs, payment requirements or arbitrary signing commands.
Unknown settlement is a durable stop condition, not permission to pay again.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
from time import monotonic
from typing import Any
from uuid import UUID

from cryptography.fernet import Fernet
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
ASSET = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
GATEWAY = "https://sol.blockrun.ai/api/v1"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
MEMO_PROGRAM = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
MAX_BYTES = 4 * 1024 * 1024
ADDRESS = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}\Z")
TRANSACTION = re.compile(r"[1-9A-HJ-NP-Za-km-z]{64,88}\Z")
MODEL_ID = re.compile(r"[A-Za-z0-9_.:/-]{1,160}\Z")
CAPABILITY_FLAGS = ("vision", "structured_actions", "structured_outputs", "tool_calling")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class BrokerError(Exception):
    def __init__(self, code: str, message: str, status: int = 503, *, retryable: bool = False):
        self.code, self.message, self.status, self.retryable = code, message, status, retryable
        super().__init__(message)

    def public(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, "retryable": self.retryable}}


def usdc_atomic(value: str | None, *, zero: bool = False) -> int | None:
    if value is None or value == "":
        return None
    try:
        amount = Decimal(value) * 1_000_000
        if not amount.is_finite() or amount != amount.to_integral_value() or amount < (0 if zero else 1) or amount > 9_000_000_000_000_000:
            raise ValueError
        return int(amount)
    except (ValueError, InvalidOperation):
        raise ValueError("USDC limits must be finite decimal amounts with at most six decimal places") from None


def capability_registry(value: str | None) -> dict:
    """Owner declarations are explicit assertions, never inferred compatibility."""
    if not value:
        return {}
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict) or len(raw) > 512:
            raise ValueError
        result = {}
        for identifier, entry in raw.items():
            if not isinstance(identifier, str) or not MODEL_ID.fullmatch(identifier) or not isinstance(entry, dict):
                raise ValueError
            if set(entry) - {"protocols", *CAPABILITY_FLAGS}:
                raise ValueError
            protocols = entry.get("protocols")
            if not isinstance(protocols, list) or not protocols or any(item not in {"responses", "messages"} for item in protocols):
                raise ValueError
            if any(key in entry and not isinstance(entry[key], bool) for key in CAPABILITY_FLAGS):
                raise ValueError
            result[identifier] = {"protocols": list(dict.fromkeys(protocols)),
                                  **{key: entry.get(key, False) for key in CAPABILITY_FLAGS}}
        return result
    except (ValueError, TypeError):
        raise ValueError("X402_RESEARCH_MODEL_CAPABILITIES must declare exact model IDs, native protocols and boolean capability flags") from None


@dataclass(frozen=True)
class BrokerConfig:
    db_path: str = "observatory-data/payments.sqlite3"
    token: str = field(default="", repr=False)
    private_key: str = field(default="", repr=False)
    encryption_key: str = field(default="", repr=False)
    enabled: bool = False
    max_request_atomic: int | None = None
    daily_limit_atomic: int | None = None
    min_reserve_atomic: int | None = None
    allowed_pay_to: frozenset[str] = frozenset()
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    timeout_seconds: float = 150
    request_deadline_seconds: float = 420
    model_capabilities: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "BrokerConfig":
        env = os.environ if env is None else env
        return cls(db_path=env.get("X402_BROKER_DB", "observatory-data/payments.sqlite3"),
            token=env.get("OBSERVATORY_PAYMENT_BROKER_TOKEN", ""),
            private_key=env.get("X402_SOLANA_PRIVATE_KEY", ""),
            encryption_key=env.get("X402_BROKER_SECRET_KEY", ""),
            enabled=env.get("X402_ENABLED", "").lower() == "true",
            max_request_atomic=usdc_atomic(env.get("X402_MAX_REQUEST_USDC")),
            daily_limit_atomic=usdc_atomic(env.get("X402_DAILY_LIMIT_USDC")),
            min_reserve_atomic=usdc_atomic(env.get("X402_MIN_RESERVE_USDC"), zero=True),
            allowed_pay_to=frozenset(item.strip() for item in env.get("X402_ALLOWED_PAY_TO", "").split(",") if item.strip()),
            rpc_url=env.get("X402_SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            model_capabilities=capability_registry(env.get("X402_RESEARCH_MODEL_CAPABILITIES")))

    def spending_errors(self) -> list[str]:
        errors = []
        if not self.enabled:
            errors.append("payments_disabled")
        if not self.token:
            errors.append("internal_token_missing")
        if not self.private_key:
            errors.append("signer_not_configured")
        if self.max_request_atomic is None or self.daily_limit_atomic is None or self.min_reserve_atomic is None:
            errors.append("explicit_spending_limits_missing")
        if not self.allowed_pay_to or any(not ADDRESS.fullmatch(item) for item in self.allowed_pay_to):
            errors.append("approved_recipient_allowlist_missing_or_invalid")
        if not self.rpc_url.startswith("https://"):
            errors.append("rpc_requires_https")
        return errors


class PaymentLedger:
    def __init__(self, path: str, encryption_key: str = ""):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._writer = self.path.with_suffix(self.path.suffix + ".writer.lock").open("a+b")
        self._writer.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                if self._writer.read(1) == b"":
                    self._writer.write(b"0")
                    self._writer.flush()
                self._writer.seek(0)
                msvcrt.locking(self._writer.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                self._writer.close()
                raise RuntimeError("Only one payment broker writer may use this ledger") from None
        else:
            import fcntl
            try:
                fcntl.flock(self._writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._writer.close()
                raise RuntimeError("Only one payment broker writer may use this ledger") from None
        self.cipher = Fernet(encryption_key.encode() if encryption_key else self._protected_key())
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS intents (
                request_id TEXT PRIMARY KEY, body_hash TEXT NOT NULL, protocol TEXT NOT NULL,
                model TEXT NOT NULL, status TEXT NOT NULL, amount INTEGER NOT NULL DEFAULT 0,
                pay_to TEXT, memo TEXT, tx TEXT UNIQUE, response BLOB, error_code TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
        """)
        # No gateway request can precede the durable authorized transition.
        self.db.execute("UPDATE intents SET status='intent', amount=0 WHERE status IN ('reserved','signing')")
        self.db.execute("UPDATE intents SET status='unknown' WHERE status='authorized'")
        self.db.commit()

    def _protected_key(self) -> bytes:
        path = self.path.with_suffix(self.path.suffix + ".key")
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return path.read_bytes().strip()
        key = Fernet.generate_key()
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name == "nt":
                identity = subprocess.run(["whoami"], check=True, capture_output=True, text=True).stdout.strip()
                subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", f"{identity}:(F)", "SYSTEM:(F)"], check=True, capture_output=True)
            else:
                os.chmod(path, 0o600)
        except Exception:
            path.unlink(missing_ok=True)
            raise RuntimeError("Protect the payment encryption key or configure X402_BROKER_SECRET_KEY") from None
        return key

    def close(self):
        self.db.close()
        self._writer.close()

    def intent(self, request_id: str, body_hash: str, protocol: str, model: str) -> dict:
        with self.lock, self.db:
            old = self.db.execute("SELECT * FROM intents WHERE request_id=?", (request_id,)).fetchone()
            if old:
                if old["body_hash"] != body_hash:
                    raise BrokerError("REQUEST_CONFLICT", "A request ID cannot be reused for different content", 409)
                return dict(old)
            stamp = utc_now()
            self.db.execute("INSERT INTO intents(request_id,body_hash,protocol,model,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                            (request_id, body_hash, protocol, model, "intent", stamp, stamp))
        return self.get(request_id)

    def get(self, request_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute("SELECT * FROM intents WHERE request_id=?", (request_id,)).fetchone()
            return dict(row) if row else None

    def update(self, request_id: str, **fields):
        if "response" in fields:
            fields["response"] = self.cipher.encrypt(canonical(fields["response"]))
        fields["updated_at"] = utc_now()
        with self.lock, self.db:
            self.db.execute("UPDATE intents SET " + ",".join(key + "=?" for key in fields) + " WHERE request_id=?",
                            (*fields.values(), request_id))

    def decoded(self, row: dict) -> dict | None:
        return json.loads(self.cipher.decrypt(row["response"])) if row.get("response") else None

    def totals(self) -> dict:
        with self.lock:
            reserved = self.db.execute("SELECT COALESCE(SUM(amount),0) FROM intents WHERE status IN ('reserved','signing','authorized','unknown')").fetchone()[0]
            settled = self.db.execute("SELECT COALESCE(SUM(amount),0) FROM intents WHERE status IN ('completed','settled_error','reconciled')").fetchone()[0]
            # UTC calendar-day limit includes every unresolved reservation, even from a prior day.
            daily = self.db.execute("SELECT COALESCE(SUM(amount),0) FROM intents WHERE status IN ('reserved','signing','authorized','unknown') OR (status IN ('completed','settled_error','reconciled') AND substr(updated_at,1,10)=?)",
                                    (utc_now()[:10],)).fetchone()[0]
            # Authorized is an ordinary in-flight reservation until completion or
            # timeout. Startup recovery converts leftover authorizations to unknown.
            unknown = self.db.execute("SELECT COUNT(*) FROM intents WHERE status='unknown'").fetchone()[0]
        return {"reserved": reserved, "settled": settled, "daily": daily, "unknown": unknown}

    def reserve(self, request_id: str, quote: dict, balance: int, config: BrokerConfig):
        with self.lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                totals = self.totals()
                if totals["unknown"]:
                    raise BrokerError("SETTLEMENT_UNKNOWN", "Reconcile an uncertain payment before authorizing another", 409)
                amount = int(quote["amount"])
                if amount > config.max_request_atomic:
                    raise BrokerError("CAP_EXCEEDED", "The quote exceeds the configured per-request cap", 422)
                if totals["daily"] + amount > config.daily_limit_atomic:
                    raise BrokerError("DAILY_LIMIT", "The UTC daily spending limit has insufficient headroom", 429, retryable=True)
                if balance - totals["reserved"] - amount < config.min_reserve_atomic:
                    raise BrokerError("UNFUNDED", "Fund the spending wallet above its reserved amount and minimum reserve", 402, retryable=True)
                self.db.execute("UPDATE intents SET status='reserved',amount=?,pay_to=?,updated_at=? WHERE request_id=?",
                                (amount, quote["payTo"], utc_now(), request_id))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise

    @staticmethod
    def receipt(row: dict) -> dict:
        return {"request_id": row["request_id"], "status": "settled" if row["status"] in {"completed", "settled_error", "reconciled"} else row["status"],
                "amount_atomic": str(row["amount"]), "network": NETWORK, "asset": ASSET,
                "transaction": row.get("tx"), "model": row["model"], "protocol": row["protocol"],
                "created_at": row["created_at"], "updated_at": row["updated_at"]}

    def receipts(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT * FROM intents WHERE amount>0 ORDER BY created_at DESC LIMIT 50").fetchall()
        return [self.receipt(dict(row)) for row in rows]


@dataclass
class SignedPayment:
    header: str = field(repr=False)
    memo: str | None = None


class OfficialSvmSigner:
    """x402 2.25.0 creates the TransferChecked transaction; no custom signing code."""
    def __init__(self, config: BrokerConfig):
        from importlib.metadata import version
        from x402.mechanisms.svm.signers import KeypairSigner
        if version("x402") != "2.25.0":
            raise RuntimeError("Install the pinned official payment SDK from requirements-payments.txt")
        self.signer = KeypairSigner.from_base58(config.private_key)
        self.config = config
        self.address = self.signer.address

    async def sign(self, required: dict, quote: dict) -> SignedPayment:
        return await asyncio.wait_for(asyncio.to_thread(self._sign, required, quote), timeout=45)

    def _sign(self, required: dict, quote: dict) -> SignedPayment:
        from x402 import x402ClientSync
        from x402.mechanisms.svm.exact.client import ExactSvmScheme
        from x402.http.utils import encode_payment_signature_header
        from x402.schemas import PaymentRequired
        from solders.transaction import VersionedTransaction
        # A seller-controlled fixed memo can match an unrelated older transfer.
        # Keep the SDK's fresh client nonce; never rewrite a merchant's quote.
        if quote.get("extra", {}).get("memo") not in (None, ""):
            raise ValueError("Seller-defined payment memos are unsupported; a fresh client nonce is required")
        client = x402ClientSync()
        client.register(NETWORK, ExactSvmScheme(self.signer, rpc_url=self.config.rpc_url))
        client.set_spend_controls({"max_amount_per_payment": "$" + str(Decimal(self.config.max_request_atomic) / 1_000_000)})
        # The broker already selected one exact, approved offer; extensions cannot broaden it.
        challenge = PaymentRequired.model_validate({"x402Version": 2, "resource": required["resource"], "accepts": [quote]})
        payload = client.create_payment_payload(challenge)
        tx = VersionedTransaction.from_bytes(base64.b64decode(payload.payload["transaction"]))
        memo = None
        for instruction in tx.message.instructions:
            if str(tx.message.account_keys[instruction.program_id_index]) == MEMO_PROGRAM:
                memo = bytes(instruction.data).decode("utf-8")
        return SignedPayment(encode_payment_signature_header(payload), memo)


class GatewayTransport:
    def __init__(self, config: BrokerConfig):
        self.config = config
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_seconds, connect=15), follow_redirects=False, trust_env=False)

    async def close(self):
        await self.http.aclose()

    async def catalog(self) -> dict:
        response = await self.http.get(GATEWAY + "/models")
        response.raise_for_status()
        return response.json()

    async def post(self, protocol: str, body: dict, payment_header: str | None = None, request_id: str = "") -> httpx.Response:
        headers = {"Content-Type": "application/json", "Idempotency-Key": request_id}
        if protocol == "messages":
            headers["anthropic-version"] = "2023-06-01"
        if payment_header:
            headers["PAYMENT-SIGNATURE"] = payment_header
        return await self.http.post(GATEWAY + "/" + protocol, json=body, headers=headers)

    async def rpc(self, method: str, params: list) -> Any:
        response = await self.http.post(self.config.rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=20)
        response.raise_for_status()
        body = response.json()
        if body.get("error"):
            raise RuntimeError("Solana read-only RPC did not return a valid result")
        return body["result"]

    async def balance(self, address: str) -> int:
        from x402.mechanisms.svm.utils import derive_ata
        ata = derive_ata(address, ASSET)
        accounts = await self.rpc("getTokenAccountsByOwner", [address, {"mint": ASSET}, {"encoding": "jsonParsed", "commitment": "confirmed"}])
        for account in accounts.get("value", []):
            if account.get("pubkey") == ata:
                info = account["account"]["data"]["parsed"]["info"]
                if info["mint"] != ASSET or info["owner"] != address or info["tokenAmount"]["decimals"] != 6:
                    raise RuntimeError("Unexpected USDC associated account")
                return int(info["tokenAmount"]["amount"])
        return 0

    async def verify_receipt(self, transaction: str, address: str, pay_to: str, amount: int, memo: str | None) -> bool:
        from x402.mechanisms.svm.utils import derive_ata
        if not TRANSACTION.fullmatch(transaction) or not memo:
            return False
        result = await self.rpc("getTransaction", [transaction, {"encoding": "jsonParsed", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}])
        if not result or result["meta"].get("err") is not None:
            return False
        message = result["transaction"]["message"]
        instructions = list(message["instructions"])
        for inner in result["meta"].get("innerInstructions", []):
            instructions.extend(inner["instructions"])
        transfer = any(instruction.get("programId") == TOKEN_PROGRAM and instruction.get("parsed", {}).get("type") == "transferChecked"
            and instruction["parsed"]["info"].get("authority") == address
            and instruction["parsed"]["info"].get("mint") == ASSET
            and instruction["parsed"]["info"].get("source") == derive_ata(address, ASSET)
            and instruction["parsed"]["info"].get("destination") == derive_ata(pay_to, ASSET)
            and instruction["parsed"]["info"].get("tokenAmount", {}).get("amount") == str(amount)
            and instruction["parsed"]["info"].get("tokenAmount", {}).get("decimals") == 6 for instruction in instructions)
        matching_memo = any(instruction.get("programId") == MEMO_PROGRAM and instruction.get("parsed") == memo for instruction in instructions)
        return transfer and matching_memo


class PaymentBroker:
    def __init__(self, config: BrokerConfig, *, transport: Any = None, signer: Any = None, ledger: PaymentLedger | None = None):
        self.config = config
        self.ledger = ledger or PaymentLedger(config.db_path, config.encryption_key)
        self.transport = transport or GatewayTransport(config)
        self.signer = signer
        self.signer_error = False
        if self.signer is None and config.private_key:
            try:
                self.signer = OfficialSvmSigner(config)
            except Exception:
                self.signer_error = True
        self.lock = asyncio.Lock()
        self._active_requests: dict[str, int] = {}
        self._catalog: dict | None = None
        self._catalog_at = 0.0

    async def close(self):
        await self.transport.close()
        self.ledger.close()

    def health(self) -> dict:
        return {"status": "ok", "service": "observatory-payments", "gateway": "blockrun-solana",
                "configured": bool(self.signer), "enabled": self.config.enabled, "network": NETWORK, "asset": ASSET}

    async def catalog(self) -> dict:
        if self._catalog is not None and monotonic() - self._catalog_at < 300:
            return self._catalog
        try:
            raw = await self.transport.catalog()
            entries = raw.get("data", raw.get("models", []))
            if not isinstance(entries, list):
                raise ValueError
            models = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                identifier = entry.get("id", "")
                if not isinstance(identifier, str) or not MODEL_ID.fullmatch(identifier):
                    continue
                raw_categories = entry.get("categories", [])
                categories = [item for item in raw_categories if isinstance(item, str) and item in {"chat", "vision", "reasoning", "coding"}] if isinstance(raw_categories, list) else []
                declared = self.config.model_capabilities.get(identifier)
                source = "owner_declared" if declared is not None else "unverified"
                if declared is None:
                    advertised_protocols = entry.get("supportedprotocols", entry.get("protocols"))
                    advertised_flags = entry.get("capabilities")
                    if isinstance(advertised_protocols, list) and isinstance(advertised_flags, dict):
                        declared = {"protocols": [item for item in advertised_protocols if isinstance(item, str) and item in {"responses", "messages"}],
                                    **{key: advertised_flags.get(key) is True for key in CAPABILITY_FLAGS}}
                        source = "gateway_advertised"
                declared = declared or {}
                protocols = declared.get("protocols", [])
                capabilities = {key: declared.get(key) is True for key in CAPABILITY_FLAGS}
                eligible = capabilities["vision"] and capabilities["structured_actions"] and any(
                    capabilities["structured_outputs"] if protocol == "responses" else capabilities["tool_calling"] for protocol in protocols)
                pricing = entry.get("pricing", {})
                safe_pricing = {key: value for key, value in pricing.items() if key in {"input", "output", "unit", "currency", "input_per_million", "output_per_million"} and isinstance(value, (str, int, float))} if isinstance(pricing, dict) else {}
                models.append({"id": identifier, "name": identifier, "categories": categories,
                    "supportedprotocols": protocols, "capabilities": capabilities, "capability_source": source, "eligible": eligible,
                    "pricing": safe_pricing, "availability": "advertised"})
            self._catalog = {"gateway": "blockrun-solana", "network": NETWORK, "asset": ASSET, "models": models,
                             "fetched_at": utc_now(), "cache_ttl_seconds": 300}
            self._catalog_at = monotonic()
            return self._catalog
        except Exception:
            raise BrokerError("CATALOG_UNAVAILABLE", "The gateway model catalog is unavailable; no payment was attempted", retryable=True) from None

    async def funding(self) -> dict:
        totals = self.ledger.totals()
        result = {"status": "unknown", "configured": bool(self.signer), "enabled": self.config.enabled,
            "wallet_address": self.signer.address if self.signer else None, "network": NETWORK, "asset": ASSET,
            "balance_atomic": None, "reserved_atomic": str(totals["reserved"]), "settled_atomic": str(totals["settled"]),
            "required_request_reserve_atomic": str(self.config.max_request_atomic) if self.config.max_request_atomic is not None else None,
            "daily_spent_and_reserved_atomic": str(totals["daily"]), "daily_remaining_atomic": str(max(0, self.config.daily_limit_atomic - totals["daily"])) if self.config.daily_limit_atomic is not None else None,
            "receipts": self.ledger.receipts()}
        if totals["unknown"]:
            return {**result, "reason": "SETTLEMENT_UNKNOWN"}
        if not self.signer:
            return {**result, "reason": "SIGNER_INVALID" if self.signer_error else "SIGNER_NOT_CONFIGURED"}
        if self.config.spending_errors():
            return {**result, "status": "configured", "reason": "NOT_CONFIGURED"}
        if totals["daily"] + self.config.max_request_atomic > self.config.daily_limit_atomic:
            return {**result, "status": "configured", "reason": "DAILY_LIMIT"}
        try:
            balance = await self.transport.balance(self.signer.address)
        except Exception:
            return {**result, "reason": "FUNDING_UNKNOWN"}
        funded = balance - totals["reserved"] >= self.config.min_reserve_atomic + self.config.max_request_atomic
        return {**result, "balance_atomic": str(balance), "status": "ready" if funded else "unfunded", "reason": None if funded else "UNFUNDED"}

    @staticmethod
    def validate_request(payload: dict) -> tuple[str, str, dict]:
        if not isinstance(payload, dict) or set(payload) != {"request_id", "protocol", "body"}:
            raise BrokerError("INVALID_REQUEST", "Supply request_id, protocol and body only", 422)
        try:
            request_id = str(UUID(payload["request_id"]))
            if request_id != payload["request_id"]:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise BrokerError("INVALID_REQUEST", "request_id must be a canonical UUID retained across retries", 422) from None
        protocol, body = payload["protocol"], payload["body"]
        if not isinstance(protocol, str) or protocol not in {"responses", "messages"} or not isinstance(body, dict):
            raise BrokerError("INVALID_REQUEST", "Choose a supported native protocol with a JSON body", 422)
        content = body.get("input" if protocol == "responses" else "messages")
        if protocol == "responses":
            valid_content = isinstance(content, (str, list)) and bool(content)
        else:
            valid_content = isinstance(content, list) and bool(content) and all(isinstance(item, dict) for item in content)
        if not valid_content:
            raise BrokerError("INVALID_REQUEST", "Supply nonempty native vendor input or messages", 422)
        limit = body.get("max_output_tokens" if protocol == "responses" else "max_tokens")
        if (body.get("stream") is not None and body.get("stream") is not False) or not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 128000:
            raise BrokerError("INVALID_REQUEST", "Non-streaming requests require an explicit bounded output-token limit", 422)
        if not isinstance(body.get("model"), str) or not body["model"]:
            raise BrokerError("INVALID_REQUEST", "Select an exact gateway catalog model ID", 422)
        if any(key in body for key in {"api_key", "authorization", "private_key", "base_url", "endpoint", "payment"}):
            raise BrokerError("INVALID_REQUEST", "Provider credentials and payment instructions are broker configuration", 422)
        try:
            if len(canonical(payload)) > MAX_BYTES:
                raise ValueError
        except (ValueError, TypeError):
            raise BrokerError("INVALID_REQUEST", "Request content must be bounded, finite JSON", 422) from None
        return request_id, protocol, body

    def select_quote(self, response: httpx.Response, protocol: str) -> tuple[dict, dict]:
        try:
            header = response.headers.get("PAYMENT-REQUIRED") or response.headers.get("X-PAYMENT-REQUIRED")
            if not header or len(header) > 65536:
                raise ValueError
            required = json.loads(base64.b64decode(header, validate=True))
            if required.get("x402Version") != 2 or required.get("resource", {}).get("url") != GATEWAY + "/" + protocol:
                raise ValueError
            for quote in required.get("accepts", []):
                extra = quote.get("extra", {})
                amount = quote.get("amount")
                if quote.get("scheme") == "exact" and quote.get("network") == NETWORK and quote.get("asset") == ASSET and quote.get("payTo") in self.config.allowed_pay_to:
                    if not isinstance(amount, str) or not re.fullmatch(r"[0-9]{1,16}", amount) or int(amount) < 1:
                        continue
                    if not isinstance(extra, dict) or not ADDRESS.fullmatch(str(extra.get("feePayer", ""))) or extra.get("feePayer") == self.signer.address:
                        continue
                    if extra.get("memo") not in (None, ""):
                        # SDK 2.25.0 otherwise uses this memo verbatim, permitting
                        # an old unrecorded same-amount receipt to match. Routes
                        # requiring seller memos need a separately bound adapter.
                        continue
                    if extra.get("paymentFlow") not in {None, "authorization"}:
                        continue
                    timeout = quote.get("maxTimeoutSeconds")
                    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 300:
                        continue
                    if int(amount) > self.config.max_request_atomic:
                        raise BrokerError("CAP_EXCEEDED", "The quote exceeds the configured per-request cap", 422)
                    return required, quote
            raise ValueError
        except BrokerError:
            raise
        except Exception:
            raise BrokerError("UNAPPROVED_QUOTE", "No exact approved Solana USDC merchant quote was offered", 422) from None

    async def request(self, payload: dict) -> dict:
        request_id, _, _ = self.validate_request(payload)
        self._active_requests[request_id] = self._active_requests.get(request_id, 0) + 1
        try:
            # Includes waiting for the serial payment lock, not only vendor time.
            async with asyncio.timeout(self.config.request_deadline_seconds):
                return await self._request(payload)
        except TimeoutError:
            row = self.ledger.get(request_id)
            if row and row["status"] in {"unknown", "authorized"}:
                raise BrokerError("SETTLEMENT_UNKNOWN", "The broker deadline expired after payment authorization; reconcile before new work", 409) from None
            raise BrokerError("UPSTREAM_UNAVAILABLE", "The broker deadline expired before payment authorization; retain this request ID when retrying", retryable=True) from None
        finally:
            remaining = self._active_requests[request_id] - 1
            if remaining:
                self._active_requests[request_id] = remaining
            else:
                self._active_requests.pop(request_id, None)

    async def _request(self, payload: dict) -> dict:
        request_id, protocol, body = self.validate_request(payload)
        async with self.lock:
            row = self.ledger.intent(request_id, hashlib.sha256(canonical(payload)).hexdigest(), protocol, body["model"])
            if row["status"] == "cancelled":
                raise BrokerError("REQUEST_ABANDONED", "This unpaid request was abandoned; its ID cannot authorize work", 409)
            cached = self.ledger.decoded(row)
            if cached is not None:
                return {"response": cached, "payment": self.ledger.receipt(row)}
            if row["status"] in {"unknown", "authorized"} or self.ledger.totals()["unknown"]:
                raise BrokerError("SETTLEMENT_UNKNOWN", "Reconcile uncertain settlement before requesting another payment", 409)
            if row["status"] in {"completed", "settled_error", "reconciled"}:
                raise BrokerError("PAID_RESPONSE_UNAVAILABLE", "This request was settled; inspect its receipt instead of paying again", 409)
            if self.config.spending_errors() or not self.signer:
                raise BrokerError("NOT_CONFIGURED", "Configure the broker signer, enablement, explicit limits and approved recipient list")
            catalog = await self.catalog()
            chosen = next((model for model in catalog["models"] if model["id"] == body["model"]), None)
            if not chosen or protocol not in chosen["supportedprotocols"] or not chosen["capabilities"]["vision"] or not chosen["capabilities"]["structured_actions"] or not chosen["capabilities"]["structured_outputs" if protocol == "responses" else "tool_calling"]:
                raise BrokerError("UNSUPPORTED_MODEL", "The selected catalog model does not support this native protocol", 422)
            try:
                probe = await self.transport.post(protocol, body, request_id=request_id)
            except Exception:
                raise BrokerError("UPSTREAM_UNAVAILABLE", "The unpaid gateway probe failed; no payment was authorized", retryable=True) from None
            if probe.status_code != 402:
                if probe.status_code in {400, 422}:
                    raise BrokerError("INVALID_REQUEST", "The native vendor request was rejected before payment; check its schema", 422)
                if probe.status_code in {401, 403, 404, 410}:
                    raise BrokerError("UPSTREAM_ACCESS_DENIED", "The selected native model route is unavailable or access was denied; no payment was authorized", 403)
                if probe.status_code == 429 or probe.status_code >= 500:
                    raise BrokerError("UPSTREAM_UNAVAILABLE", "The unpaid gateway route is temporarily unavailable; no payment was authorized", retryable=True)
                raise BrokerError("UNEXPECTED_GATEWAY_RESPONSE", "The configured paid gateway did not supply an approved payment challenge", 502)
            required, quote = self.select_quote(probe, protocol)
            try:
                balance = await self.transport.balance(self.signer.address)
            except Exception:
                raise BrokerError("FUNDING_UNKNOWN", "The Solana USDC balance could not be verified", retryable=True) from None
            self.ledger.reserve(request_id, quote, balance, self.config)
            try:
                self.ledger.update(request_id, status="signing")
                signed = await self.signer.sign(required, quote)
            except BaseException as exc:
                self.ledger.update(request_id, status="intent", amount=0)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise BrokerError("SIGNING_FAILED", "The official payment signer could not authorize this quote; no paid request was sent", retryable=True) from None
            self.ledger.update(request_id, status="authorized", memo=signed.memo)
            try:
                response = await self.transport.post(protocol, body, payment_header=signed.header, request_id=request_id)
                # Preserve a useful delivered response even if settlement confirmation
                # is late. It remains encrypted and carries an unknown payment state.
                vendor = None
                if response.status_code == 200 and response.headers.get("X-Fallback-Used", "").lower() != "true" and response.headers.get("X-Context-Truncated", "").lower() != "true":
                    try:
                        candidate = response.json()
                        if isinstance(candidate, dict) and len(canonical(candidate)) <= MAX_BYTES:
                            vendor = candidate
                            self.ledger.update(request_id, response=vendor)
                    except (ValueError, TypeError):
                        pass
                transaction = None
                receipt_header = response.headers.get("PAYMENT-RESPONSE")
                receipt = json.loads(base64.b64decode(receipt_header, validate=True)) if receipt_header else {}
                if receipt.get("network") == NETWORK and TRANSACTION.fullmatch(str(receipt.get("transaction", ""))):
                    transaction = receipt["transaction"]
                    self.ledger.update(request_id, tx=transaction)
                if receipt.get("success") is not True or not transaction or response.headers.get("X-Payment-Settled", "").lower() == "false":
                    raise ValueError("Settlement must be reconciled")
                verified = await self.transport.verify_receipt(transaction, self.signer.address, quote["payTo"], int(quote["amount"]), signed.memo)
                if not verified:
                    raise ValueError("Settlement must be reconciled")
                self.ledger.update(request_id, status="settled_error", error_code="UPSTREAM_REJECTED")
                if vendor is None:
                    raise BrokerError("PAID_UPSTREAM_REJECTED", "The paid response failed or changed the requested model/context; inspect the recorded receipt", 502)
                self.ledger.update(request_id, status="completed", response=vendor, error_code=None)
                return {"response": vendor, "payment": self.ledger.receipt(self.ledger.get(request_id))}
            except BrokerError:
                raise
            except BaseException as exc:
                self.ledger.update(request_id, status="unknown", error_code="SETTLEMENT_UNKNOWN")
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise BrokerError("SETTLEMENT_UNKNOWN", "A payment may have been sent; reconcile its receipt before retrying or authorizing new work", 409) from None

    async def reconcile(self, request_id: str, transaction: str | None = None) -> dict:
        """Read-only on-chain verification; never resend or sign a payment."""
        async with self.lock:
            row = self.ledger.get(request_id)
            if not row or row["status"] != "unknown":
                raise BrokerError("INVALID_RECONCILIATION", "Select an existing uncertain payment", 422)
            transaction = transaction or row.get("tx")
            if not isinstance(transaction, str) or not TRANSACTION.fullmatch(transaction) or not self.signer:
                raise BrokerError("SETTLEMENT_UNKNOWN", "An on-chain transaction receipt is required; no reservation was released", 409)
            try:
                verified = await self.transport.verify_receipt(transaction, self.signer.address, row["pay_to"], row["amount"], row.get("memo"))
            except Exception:
                verified = False
            if not verified:
                raise BrokerError("SETTLEMENT_UNKNOWN", "The exact memo-bound USDC transfer is not confirmed; reservation remains held", 409)
            try:
                self.ledger.update(request_id, status="completed" if row.get("response") else "reconciled", tx=transaction, error_code=None)
            except sqlite3.IntegrityError:
                raise BrokerError("INVALID_RECONCILIATION", "The transaction is already attributed to another payment", 409) from None
            return {"payment": self.ledger.receipt(self.ledger.get(request_id)), "response_recovered": bool(row.get("response"))}

    def inspect_request(self, request_id: str) -> dict:
        """Inspect the local receipt and decrypted cache without authorizing work."""
        try:
            if str(UUID(request_id)) != request_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise BrokerError("INVALID_REQUEST", "request_id must be a canonical UUID", 422) from None
        row = self.ledger.get(request_id)
        in_flight = self._active_requests.get(request_id, 0) > 0
        if row is None:
            return {"found": False, "request_id": request_id, "status": "not_found", "state": "not_found", "response": None, "payment": None, "in_flight": in_flight}
        payment = self.ledger.receipt(row)
        return {"found": True, "request_id": request_id, "status": payment["status"], "state": row["status"],
                "response": self.ledger.decoded(row), "payment": payment, "in_flight": in_flight}

    async def abandon_request(self, request_id: str, payload: dict) -> dict:
        """Tombstone exact unpaid content so a delayed POST cannot authorize it."""
        if not isinstance(payload, dict) or set(payload) != {"protocol", "body"}:
            raise BrokerError("INVALID_REQUEST", "Supply the original protocol and body only", 422)
        original = {"request_id": request_id, **payload}
        request_id, protocol, body = self.validate_request(original)
        if self._active_requests.get(request_id, 0):
            raise BrokerError("REQUEST_IN_FLIGHT", "Wait for this active request to finish before reviewing abandonment", 409)
        try:
            async with asyncio.timeout(5):
                async with self.lock:
                    if self._active_requests.get(request_id, 0):
                        raise BrokerError("REQUEST_IN_FLIGHT", "Wait for this active request to finish before reviewing abandonment", 409)
                    row = self.ledger.intent(request_id, hashlib.sha256(canonical(original)).hexdigest(), protocol, body["model"])
                    if row["status"] not in {"intent", "cancelled"}:
                        raise BrokerError("REQUEST_NOT_ABANDONABLE", "An authorized, reserved, uncertain or settled payment cannot be abandoned", 409)
                    if row["status"] != "cancelled":
                        self.ledger.update(request_id, status="cancelled", amount=0, error_code="REQUEST_ABANDONED")
                    return {"abandoned": True, "request_id": request_id, "state": "cancelled", "in_flight": False}
        except TimeoutError:
            raise BrokerError("BROKER_BUSY", "The broker is processing another request; inspect again before abandonment", 409, retryable=True) from None


def create_app(broker: PaymentBroker | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        owned = broker is None
        application.state.broker = broker or PaymentBroker(BrokerConfig.from_env())
        try:
            yield
        finally:
            if owned:
                await application.state.broker.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    if broker:
        app.state.broker = broker

    @app.exception_handler(BrokerError)
    async def broker_error(_request: Request, exc: BrokerError):
        return JSONResponse(exc.public(), status_code=exc.status, headers={"Cache-Control": "no-store"})

    def authenticate(request: Request):
        token = request.app.state.broker.config.token
        if not token:
            raise BrokerError("NOT_CONFIGURED", "Configure the internal broker Bearer token")
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            raise BrokerError("AUTH_FAILED", "Valid internal Bearer authentication is required", 401)

    async def read_payload(request: Request) -> dict:
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_BYTES:
                raise BrokerError("INVALID_REQUEST", "Supply a bounded JSON object", 422)
        try:
            if len(raw) > MAX_BYTES:
                raise ValueError
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError
            return payload
        except (ValueError, TypeError):
            raise BrokerError("INVALID_REQUEST", "Supply a bounded JSON object", 422) from None

    @app.get("/health")
    async def health(request: Request):
        return request.app.state.broker.health()

    @app.get("/catalog")
    async def catalog(request: Request):
        return await request.app.state.broker.catalog()

    @app.get("/funding")
    async def funding(request: Request):
        return JSONResponse(await request.app.state.broker.funding(), headers={"Cache-Control": "no-store"})

    @app.post("/v1/request")
    async def paid_request(request: Request):
        authenticate(request)
        result = await request.app.state.broker.request(await read_payload(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.get("/v1/requests/{request_id}")
    async def inspect_request(request: Request, request_id: str):
        authenticate(request)
        return JSONResponse(request.app.state.broker.inspect_request(request_id), headers={"Cache-Control": "no-store"})

    @app.post("/v1/requests/{request_id}/abandon")
    async def abandon_request(request: Request, request_id: str):
        authenticate(request)
        result = await request.app.state.broker.abandon_request(request_id, await read_payload(request))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    @app.post("/v1/reconcile")
    async def reconcile(request: Request):
        authenticate(request)
        payload = await read_payload(request)
        if set(payload) - {"request_id", "transaction"} or not isinstance(payload.get("request_id"), str):
            raise BrokerError("INVALID_RECONCILIATION", "Supply request_id and an optional transaction receipt", 422)
        result = await request.app.state.broker.reconcile(payload["request_id"], payload.get("transaction"))
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    return app


app = create_app()

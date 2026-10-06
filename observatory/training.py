"""Durable, opt-in training jobs. Importing this module never provisions a GPU."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

SELECTED_MODEL = json.loads(Path(__file__).with_name("selected-model.json").read_text(encoding="utf-8"))
SELECTED_SETTINGS = SELECTED_MODEL["settings"]

DEFAULTS = {
    "training_enabled": False, "training_provider": "hf_jobs",
    "training_hardware": "a100-large", "training_interval_hours": 4,
    "training_timeout_seconds": 14400, "training_mode": "qlora",
    "publish_policy": "private", "activation_policy": "manual",
    "training_min_documents": 20, "training_min_tokens": 50000,
    "training_max_steps": 100, "training_sequence_length": 2048,
    "training_batch_size": 1, "training_gradient_accumulation": 8,
    "training_learning_rate": 0.0001, "training_lora_rank": 16,
    "training_lora_alpha": 32, "training_seed": 792351,
    "training_max_loss_ratio": 1.0, "training_calibration_episodes": 40,
    "training_calibration_min_rate": 0.65, "training_budget_usd": None,
    "training_continue_from_previous": True,
    "training_eval_suite_path": "", "training_eval_expected_sha256": "",
    "training_eval_max_accuracy_drop": 0.0, "training_eval_max_nll_ratio": 1.0,
    "training_eval_min_domain_accuracy": 0.5, "training_eval_min_general_accuracy": 0.5,
    "training_eval_max_length": 1024,
    **SELECTED_SETTINGS,
}
ACTIVE = {"preparing", "submitting", "submission_unknown", "submitted", "running"}
HARDWARE = {"a100-large", "h200"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def stage_content_hash(snapshot: dict, stage: str) -> str:
    if stage == "cpt":
        content = sorted((row.get("content_hash") or digest(row["text"]), row["family_id"], row["split"])
                         for row in snapshot["original_text"])
    else:
        content = sorted((digest(row["messages"]), tuple(sorted(row["family_ids"])), row["split"])
                         for row in snapshot["synthetic_sft"])
    return digest(content)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _field(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _redact(message: Any, token: str) -> str:
    text = str(message).replace(token, "[REDACTED]") if token else str(message)
    return re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "[REDACTED]", text)[:8000]


class HFJobsProvider:
    """Small HF Hub boundary; all calls run in bounded coordinator tasks."""

    def __init__(self, settings: dict, token: str):
        from huggingface_hub import HfApi
        self.settings, self.token = settings, token
        self.api = HfApi(token=token)
        self.namespace = settings["hf_namespace"]

    def prepare(self, manifest: dict, payload: dict) -> dict:
        from .train_worker import validate_payload
        validate_payload(payload, manifest["stage"], require_quality=True)
        # Resolve a moving owner-selected ref once; the worker only receives commits.
        base_info = self.api.model_info(manifest["base_model"], revision=manifest.get("base_revision"))
        parameters = int(_field(_field(base_info, "safetensors", {}), "total", 0) or 0)
        if manifest["training"]["mode"] == "lora" and (parameters >= 60_000_000_000 or
                re.search(r"(?:^|[-/])(?:70|72)b(?:[-/]|$)", manifest["base_model"], re.I)):
            raise ValueError("The supported single-GPU 70B profiles require QLoRA; full-precision LoRA needs a separately benchmarked distributed recipe")
        from .evaluation import load_eval_suite, suite_hash, validate_dataset_exclusion
        suite = load_eval_suite(self.settings)
        if suite_hash(suite) != manifest.get("evaluation", {}).get("sha256"):
            raise ValueError("evaluation_suite_changed_after_job_preparation")
        validate_dataset_exclusion(payload, suite)
        revision = base_info.sha
        manifest = {**manifest, "base_revision": revision, "tokenizer_revision": revision}
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tokenizer_path = hf_hub_download(repo_id=manifest["base_model"], filename="tokenizer.json",
                                          revision=revision, token=self.token)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        if manifest["stage"] == "cpt":
            actual_counts = {split + "_tokens": sum(len(tokenizer.encode(row["text"], add_special_tokens=False).ids) + 1
                for row in payload["original_text"] if row["split"] == split) for split in ("train", "validation")}
        else:
            from .train_worker import render_sft_text
            config_path = hf_hub_download(repo_id=manifest["base_model"], filename="tokenizer_config.json",
                                           revision=revision, token=self.token)
            with open(config_path, encoding="utf-8") as handle:
                eos = json.load(handle).get("eos_token")
            if isinstance(eos, dict):
                eos = eos.get("content")
            if not isinstance(eos, str) or not eos:
                raise ValueError("The supported SFT recipe requires an explicit tokenizer EOS token")
            actual_counts = {split + "_tokens": sum(len(tokenizer.encode(render_sft_text(row["messages"], eos), add_special_tokens=False).ids)
                for row in payload["synthetic_sft"] if row["split"] == split) for split in ("train", "validation")}
        if actual_counts["train_tokens"] < manifest["training"]["min_tokens"]:
            raise ValueError("Insufficient actual tokenizer-counted training tokens; no GPU job was provisioned")
        manifest["preflight_counts"] = actual_counts
        repo = manifest["dataset_repo"]
        # Dataset exports remain private even when a model adapter is public.
        self.api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        if not self.api.dataset_info(repo).private:
            raise ValueError("Corpus exports require a private dataset repository")
        path = f"observatory/snapshots/{manifest['snapshot_id']}.json"
        upload = self.api.upload_file(path_or_fileobj=canonical_bytes(payload), path_in_repo=path,
                                      repo_id=repo, repo_type="dataset", commit_message=f"Seal {manifest['snapshot_id']}")
        manifest.update(dataset_path=path, dataset_revision=str(upload.oid), dataset_sha256=digest(payload))
        eval_path = f"observatory/evaluations/{suite_hash(suite)}.json"
        eval_upload = self.api.upload_file(path_or_fileobj=canonical_bytes(suite), path_in_repo=eval_path,
                                         repo_id=repo, repo_type="dataset", commit_message=f"Seal evaluation {suite['id']} {suite['version']}")
        manifest["evaluation"] = {**manifest["evaluation"], "repo_id": repo, "path": eval_path,
                                  "revision": str(eval_upload.oid)}
        sealed = dict(manifest)
        manifest_path = f"observatory/jobs/{manifest['run_id']}/manifest.json"
        commit = self.api.upload_file(path_or_fileobj=canonical_bytes(sealed), path_in_repo=manifest_path,
                                     repo_id=repo, repo_type="dataset", commit_message=f"Prepare {manifest['run_id']}")
        return {"manifest": sealed, "manifest_repo": repo, "manifest_path": manifest_path,
                "manifest_revision": str(commit.oid), "manifest_sha256": digest(sealed)}

    def run_job(self, bundle: dict) -> dict:
        manifest = bundle["manifest"]
        job = self.api.run_job(
            namespace=self.namespace,
            image=self.settings["training_image"],
            command=["python", "-m", "observatory.train_worker"],
            flavor=self.settings["training_hardware"], timeout=int(self.settings["training_timeout_seconds"]),
            env={
                "OBSERVATORY_MANIFEST_REPO": bundle["manifest_repo"],
                "OBSERVATORY_MANIFEST_PATH": bundle["manifest_path"],
                "OBSERVATORY_MANIFEST_REVISION": bundle["manifest_revision"],
                "OBSERVATORY_MANIFEST_SHA256": bundle["manifest_sha256"],
                "HF_HUB_DOWNLOAD_TIMEOUT": "60", "HF_HUB_ETAG_TIMEOUT": "30",
            }, secrets={"HF_TOKEN": self.token},
            labels={"observatory_run_id": manifest["run_id"], "stage": manifest["stage"]},
        )
        return {"id": str(job.id), "url": _field(job, "url")}

    def inspect_job(self, job_id: str) -> dict:
        job = self.api.inspect_job(job_id=job_id, namespace=self.namespace)
        status = _field(job, "status", {})
        stage = _field(status, "stage", status)
        return {"status": str(getattr(stage, "value", stage)).upper(),
                "status_message": _field(status, "message"), "url": _field(job, "url")}

    def find_job(self, run_id: str) -> dict | None:
        matches = [job for job in self.api.list_jobs(namespace=self.namespace)
                   if (_field(job, "labels", {}) or {}).get("observatory_run_id") == run_id]
        if len(matches) > 1:
            raise ValueError("Multiple HF jobs share this run label; owner inspection is required")
        return {"id": str(_field(matches[0], "id")), "url": _field(matches[0], "url")} if matches else None

    def logs(self, job_id: str) -> list[str]:
        # follow=False is crucial: polling must never wait for the job to finish.
        return list(self.api.fetch_job_logs(job_id=job_id, namespace=self.namespace, follow=False, tail=100))

    def cancel(self, job_id: str) -> None:
        self.api.cancel_job(job_id=job_id, namespace=self.namespace)

    def result(self, run: dict) -> dict | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            info = self.api.model_info(run["manifest"]["output_repo"])
            path = hf_hub_download(repo_id=run["manifest"]["output_repo"],
                                   filename=f"runs/{run['id']}/result.json", revision=info.sha, token=self.token)
        except EntryNotFoundError:
            return None
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
        result["artifact_revision"] = info.sha
        return result


class TrainingCoordinator:
    def __init__(self, store: Any, *, provider_factory=HFJobsProvider, poll_seconds: float = 30):
        self.store, self.provider_factory, self.poll_seconds = store, provider_factory, poll_seconds
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._preparations: set[asyncio.Task] = set()

    def _settings(self) -> tuple[dict, str]:
        saved = self.store.get_settings(private=True)
        settings = {**DEFAULTS, **saved}
        # The handoff pin belongs to its selected repository. An owner-selected
        # model without an explicit ref must resolve its own revision in prepare.
        if settings["hf_base_model"] != SELECTED_SETTINGS["hf_base_model"] and "hf_base_revision" not in saved:
            settings["hf_base_revision"] = None
        token = os.environ.get("HF_TOKEN") or settings.get("hf_token", "")
        # An empty saved field must not override an explicitly configured environment.
        for name in ("hf_namespace", "hf_dataset_repo", "hf_model_repo", "training_image"):
            if not settings.get(name):
                settings[name] = os.environ.get("OBSERVATORY_" + name.upper(), "")
        return settings, token

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        # HTTP process restarts cannot silently resume a partially submitted API
        # request. Submitted jobs retain their IDs and continue to be polled.
        for run in self.store.list_records("training_runs"):
            if run.get("status") in {"preparing", "submitting"}:
                run["status"] = "failed" if run["status"] == "preparing" else "submission_unknown"
                run["error"] = "Coordinator restarted during job preparation/submission; HF jobs are recovered by their observatory_run_id label"
                self.store.put("training_runs", run)
        self._task = asyncio.create_task(self._loop(), name="observatory-training")

    async def close(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        # Let a bounded submission resolve rather than losing the identity of a paid job.
        if self._preparations:
            await asyncio.gather(*self._preparations, return_exceptions=True)

    def _configuration_errors(self, settings: dict, token: str) -> list[str]:
        reasons = []
        if settings.get("training_enabled") is not True:
            reasons.append("training_disabled")
        if not token:
            reasons.append("hf_token_missing")
        if settings.get("training_provider") != "hf_jobs":
            reasons.append("unsupported_training_provider")
        for key in ("hf_namespace", "hf_dataset_repo", "hf_model_repo", "training_image"):
            if not settings.get(key):
                reasons.append(key + "_missing")
        image = settings.get("training_image")
        if image and (not isinstance(image, str) or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image)):
            reasons.append("training_image_requires_immutable_sha256_digest")
        for key in ("hf_dataset_repo", "hf_model_repo"):
            value = settings.get(key, "")
            if value and (value.count("/") != 1 or value.split("/")[0] != settings.get("hf_namespace")):
                reasons.append(key + "_must_belong_to_owner_namespace")
        if settings.get("training_hardware") not in HARDWARE:
            reasons.append("unsupported_hardware_profile")
        if settings.get("training_mode") not in {"qlora", "lora"}:
            reasons.append("unsupported_training_mode")
        if settings.get("publish_policy") not in {"private", "public", "hold"}:
            reasons.append("unsupported_publish_policy")
        try:
            if not 60 <= int(settings["training_timeout_seconds"]) <= 604800:
                reasons.append("training_timeout_out_of_range")
            if float(settings["training_interval_hours"]) <= 0:
                reasons.append("training_interval_must_be_positive")
        except (TypeError, ValueError):
            reasons.append("invalid_schedule_or_timeout")
        return reasons

    async def submit(self, snapshot_id: str | None = None, *, retry: bool = False) -> dict:
        return await self._submit("cpt", snapshot_id, retry=retry)

    async def submit_sft(self, parent_run_id: str, snapshot_id: str | None = None, *, retry: bool = False) -> dict:
        settings, _token = self._settings()
        if settings.get("synthetic_training_approved") is not True or not settings.get("provider_policy_reference"):
            return {"status": "not_ready", "reasons": ["synthetic_sft_requires_owner_approval_and_provider_policy_reference"]}
        parent = self.store.get("training_runs", parent_run_id)
        if (not parent or parent.get("stage") != "cpt" or not parent.get("checks", {}).get("passed")
                or not parent.get("checks", {}).get("independent_evaluation_passed") or not parent.get("artifact_revision")):
            return {"status": "not_ready", "reasons": ["sft_requires_a_published_cpt_checkpoint_with_passed_checks"]}
        return await self._submit("sft", snapshot_id, parent=parent, retry=retry)

    async def _submit(self, stage: str, snapshot_id: str | None, parent: dict | None = None, *, retry: bool = False) -> dict:
        async with self._lock:
            settings, token = self._settings()
            reasons = self._configuration_errors(settings, token)
            if reasons:
                return {"status": "not_ready", "reasons": reasons}
            if snapshot_id:
                snapshot = self.store.get("datasets", snapshot_id)
                if not snapshot:
                    return {"status": "not_ready", "reasons": ["snapshot_not_found"]}
            else:
                from .curation import build_snapshot
                snapshot = build_snapshot(self.store)
            from .train_worker import validate_payload
            from .evaluation import load_eval_suite, suite_hash, evaluation_policy, validate_dataset_exclusion
            try:
                suite = load_eval_suite(settings)
                eval_policy = evaluation_policy(settings)
                validate_dataset_exclusion(snapshot, suite)
                splits = validate_payload(snapshot, stage, require_quality=True)
                if stage == "cpt" and len(splits["train"]) < int(settings["training_min_documents"]):
                    reasons.append("insufficient_training_documents")
            except (ValueError, TypeError, KeyError, OSError) as exc:
                reasons.append(str(exc))
            if not snapshot.get("immutable"):
                reasons.append("snapshot_must_be_immutable")
            if reasons:
                return {"status": "not_ready", "snapshot_id": snapshot["id"], "reasons": reasons}
            runs = self.store.list_records("training_runs")
            if stage == "cpt" and settings.get("training_continue_from_previous") is True:
                candidates = [run for run in runs if run.get("stage") == "cpt" and run.get("status") == "passed"
                    and run.get("checks", {}).get("passed") and run.get("checks", {}).get("independent_evaluation_passed")
                    and run.get("published") and run.get("artifact_revision")
                    and run.get("manifest", {}).get("base_model") == settings["hf_base_model"]
                    and (not settings.get("hf_base_revision") or run["manifest"].get("base_revision") == settings["hf_base_revision"])]
                parent = candidates[-1] if candidates else None
            effective_base = parent["manifest"]["base_model"] if parent else settings["hf_base_model"]
            effective_revision = parent["manifest"]["base_revision"] if parent else settings.get("hf_base_revision")
            fingerprint = digest({"corpus": stage_content_hash(snapshot, stage), "stage": stage,
                                  "evaluation_sha256": suite_hash(suite),
                                  "base": effective_base if stage == "sft" else settings["hf_base_model"],
                                  "revision": effective_revision if stage == "sft" else settings.get("hf_base_revision"),
                                  "parent": parent["id"] if parent and stage == "sft" else None,
                                  "recipe": {key: value for key, value in settings.items() if key.startswith("training_") and key not in {
                                      "training_enabled", "training_image", "training_budget_usd", "training_interval_hours",
                                      "training_timeout_seconds", "training_hardware", "training_provider"}}})
            previous = next((run for run in reversed(runs) if run.get("fingerprint") == fingerprint), None)
            if previous and (retry is not True or previous.get("status") not in {"failed", "cancelled", "completed_unverified"}):
                return previous
            active = next((run for run in runs if run.get("status") in ACTIVE), None)
            if active:
                return {"status": "not_ready", "reasons": ["training_job_in_progress"], "active_run_id": active["id"]}
            run_id = str(uuid4())
            config = {
                "mode": settings["training_mode"], "seed": int(settings["training_seed"]),
                "max_steps": int(settings["training_max_steps"]), "sequence_length": int(settings["training_sequence_length"]),
                "batch_size": int(settings["training_batch_size"]), "gradient_accumulation": int(settings["training_gradient_accumulation"]),
                "learning_rate": float(settings["training_learning_rate"]), "lora_rank": int(settings["training_lora_rank"]),
                "lora_alpha": int(settings["training_lora_alpha"]), "min_tokens": int(settings["training_min_tokens"]),
                "max_loss_ratio": float(settings["training_max_loss_ratio"]),
                "calibration_episodes": int(settings["training_calibration_episodes"]),
                "calibration_min_rate": float(settings["training_calibration_min_rate"]),
                "resume_from_checkpoint": settings.get("training_resume_checkpoint"),
            }
            if any(config[key] < 1 for key in ("max_steps", "sequence_length", "batch_size", "gradient_accumulation", "lora_rank")):
                return {"status": "not_ready", "reasons": ["training_dimensions_must_be_positive"]}
            manifest = {
                "schema_version": 1, "run_id": run_id, "stage": stage,
                "snapshot_id": snapshot["id"], "snapshot_hash": snapshot["manifest_hash"],
                "stage_content_hash": stage_content_hash(snapshot, stage),
                "base_model": settings["hf_base_model"], "base_revision": settings.get("hf_base_revision"),
                "dataset_repo": settings["hf_dataset_repo"], "output_repo": settings["hf_model_repo"],
                "output_subfolder": f"runs/{run_id}/adapter", "training": config,
                "publish_policy": settings["publish_policy"], "created_at": now(),
                "evaluation": {"id": suite["id"], "version": suite["version"], "sha256": suite_hash(suite),
                               "kind": suite.get("kind", "operator_evaluation"), "policy": eval_policy},
                "parent_adapter": {"repo_id": parent["manifest"]["output_repo"], "revision": parent["artifact_revision"],
                                   "subfolder": parent["manifest"]["output_subfolder"], "run_id": parent["id"]} if parent else None,
            }
            # SFT must continue the pinned base underlying its CPT adapter, even after owner settings change.
            if parent:
                manifest.update(base_model=parent["manifest"]["base_model"], base_revision=parent["manifest"]["base_revision"])
            manifest["heldout_family_ids"] = sorted(set(snapshot.get("heldout_family_ids", [])) |
                set(parent.get("manifest", {}).get("heldout_family_ids", []) if parent else []))
            record = self.store.put("training_runs", {"id": run_id, "stage": stage, "status": "preparing",
                "snapshot_id": snapshot["id"], "fingerprint": fingerprint, "manifest": manifest,
                "hardware": settings["training_hardware"], "parent_run_id": parent["id"] if parent else None,
                "retry_of": previous["id"] if previous else None,
                "checks": {"passed": False, "status": "not_run"}, "calibration": {"passed": False, "status": "not_run"},
                "budget_usd": settings.get("training_budget_usd"), "cost_usd": None})
            self.store.event("training.preparing", f"Preparing separately tracked {stage.upper()} job", run_id=run_id)
            task = asyncio.create_task(self._prepare(record, snapshot, settings, token))
            self._preparations.add(task)
            task.add_done_callback(self._preparations.discard)
            return record

    async def _call(self, method, *args):
        return await asyncio.wait_for(asyncio.to_thread(method, *args), timeout=55)

    async def _prepare(self, run: dict, snapshot: dict, settings: dict, token: str) -> None:
        submitting = False
        try:
            provider = self.provider_factory(settings, token)
            bundle = await self._call(provider.prepare, run["manifest"], snapshot)
            current = self.store.get("training_runs", run["id"])
            if current.get("status") == "cancelled":
                return
            run.update(bundle, status="submitting")
            self.store.put("training_runs", run)
            submitting = True
            job = await self._call(provider.run_job, bundle)
            run.update(provider_job_id=job["id"], provider_job_url=job.get("url"), status="submitted")
            # A cancellation that arrived during the API call must cancel the actual created job.
            current = self.store.get("training_runs", run["id"])
            if current.get("status") == "cancelled":
                await self._call(provider.cancel, job["id"])
                run["status"] = "cancelled"
            self.store.put("training_runs", run)
            self.store.event("training.submitted", "HF GPU job submitted", run_id=run["id"], data={"job_id": job["id"]})
        except Exception as exc:
            current = self.store.get("training_runs", run["id"])
            if current and current.get("status") == "cancelled":
                return
            # A timeout can occur after HF accepted the paid job. Never automatically resubmit.
            response_code = getattr(getattr(exc, "response", None), "status_code", None)
            rejected = response_code is not None and 400 <= response_code < 500 and response_code != 408
            run.update(status="submission_unknown" if submitting and not rejected else "failed",
                       error=_redact(f"{type(exc).__name__}: {exc}", token))
            self.store.put("training_runs", run)
            self.store.event("training.error", run["error"], run_id=run["id"])

    async def _poll(self, run: dict) -> None:
        settings, token = self._settings()
        if not token or not settings.get("hf_namespace"):
            return
        provider = self.provider_factory(settings, token)
        try:
            inspection = await self._call(provider.inspect_job, run["provider_job_id"])
            lines = await self._call(provider.logs, run["provider_job_id"])
            seen = set(run.get("log_hashes", []))
            for line in lines:
                line = _redact(line, token)
                key = hashlib.sha256(line.encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                self.store.event("training.log", line, run_id=run["id"])
                marker = line.find("OBS_EVENT ")
                if marker >= 0:
                    try:
                        event = json.loads(line[marker + len("OBS_EVENT "):])
                        if event.get("type") == "metrics":
                            run["metrics"] = event.get("metrics", {})
                        elif event.get("type") == "result" and event.get("result", {}).get("run_id") == run["id"]:
                            run["worker_result"] = event["result"]
                    except (ValueError, TypeError):
                        pass
            run["log_hashes"] = sorted(seen)[-1000:]
            run["provider_status"] = inspection["status"]
            stage = inspection["status"]
            if stage in {"RUNNING"}:
                run["status"] = "running"
            elif stage in {"COMPLETED", "SUCCEEDED"}:
                result = run.get("worker_result")
                if not result or (result.get("published") and not result.get("artifact_revision")):
                    try:
                        result = await self._call(provider.result, run) or result
                    except Exception as exc:
                        self.store.event("training.result_unavailable", _redact(f"{type(exc).__name__}: {exc}", token), run_id=run["id"])
                if result and result.get("run_id") == run["id"] and result.get("snapshot_hash") == run["manifest"]["snapshot_hash"]:
                    evaluation = result.get("evaluation", {})
                    expected_eval = run["manifest"].get("evaluation", {})
                    eval_verified = (evaluation.get("passed") is True and evaluation.get("status") == "measured"
                        and evaluation.get("suite_sha256") == expected_eval.get("sha256")
                        and result.get("checks", {}).get("independent_evaluation_passed") is True)
                    if not eval_verified:
                        result = {**result, "checks": {**result.get("checks", {}), "passed": False,
                                  "independent_evaluation_passed": False}, "published": False}
                    run.update(checks=result.get("checks", {"passed": False}), calibration=result.get("calibration", {"passed": False}),
                               metrics=result.get("metrics", {}), evaluation=evaluation, artifact_revision=result.get("artifact_revision"),
                               published=bool(result.get("published")))
                    run["status"] = "passed" if run["checks"].get("passed") else "failed"
                    if run.get("artifact_revision") and eval_verified:
                        self.store.put("checkpoints", {"id": run["id"], "run_id": run["id"], "stage": run["stage"],
                            "status": "candidate", "checks": run["checks"], "calibration": run["calibration"],
                            "repo_id": run["manifest"]["output_repo"], "revision": run["artifact_revision"],
                            "subfolder": run["manifest"]["output_subfolder"], "base_model": run["manifest"]["base_model"],
                            "base_revision": run["manifest"]["base_revision"], "metrics": run["metrics"]})
                else:
                    run["status"] = "completed_unverified"
            elif stage in {"ERROR", "FAILED", "CANCELED", "CANCELLED", "EXPIRED", "DELETED"}:
                run["status"] = "cancelled" if stage in {"CANCELED", "CANCELLED"} else "failed"
                run["error"] = _redact(inspection.get("status_message") or stage, token)
            if run["status"] not in ACTIVE:
                run["finished_at"] = now()
            self.store.put("training_runs", run)
        except Exception as exc:
            self.store.event("training.poll_error", _redact(f"{type(exc).__name__}: {exc}", token), run_id=run["id"])

    async def tick(self, current_time: datetime | None = None) -> None:
        """One scheduler tick, public for deterministic startup/schedule verification."""
        for run in self.store.list_records("training_runs"):
            if run.get("status") == "submission_unknown":
                settings, token = self._settings()
                if token and settings.get("hf_namespace"):
                    try:
                        provider = self.provider_factory(settings, token)
                        found = await self._call(provider.find_job, run["id"])
                        if found:
                            run.update(provider_job_id=found["id"], provider_job_url=found.get("url"), status="submitted")
                            if run.get("cancel_requested"):
                                await self._call(provider.cancel, found["id"])
                                run.update(status="cancelled", finished_at=now())
                            self.store.put("training_runs", run)
                            self.store.event("training.recovered", "Recovered HF job identity from its run label", run_id=run["id"])
                    except Exception as exc:
                        self.store.event("training.recovery_error", _redact(f"{type(exc).__name__}: {exc}", token), run_id=run["id"])
            if run.get("status") in {"submitted", "running"} and run.get("provider_job_id"):
                await self._poll(run)
        settings, token = self._settings()
        if self._configuration_errors(settings, token):
            return
        current = current_time or datetime.now(timezone.utc)
        state = self.store.get("runtime", "training_scheduler") or {"id": "training_scheduler"}
        previous = state.get("last_tick")
        if previous and current < datetime.fromisoformat(previous.replace("Z", "+00:00")) + timedelta(hours=float(settings["training_interval_hours"])):
            return
        state["last_tick"] = current.isoformat()
        self.store.put("runtime", state)
        result = await self.submit()
        if result.get("status") == "not_ready":
            self.store.event("training.not_ready", "Scheduled training is waiting for readiness", data={"reasons": result["reasons"]})

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.store.event("training.scheduler_error", type(exc).__name__)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_seconds)
            except asyncio.TimeoutError:
                pass

    async def cancel(self, run_id: str) -> dict:
        async with self._lock:
            run = self.store.get("training_runs", run_id)
            if not run:
                raise ValueError("Training run not found")
            if run.get("status") not in ACTIVE:
                return run
            if run.get("status") == "submission_unknown":
                run["cancel_requested"] = True
                self.store.put("training_runs", run)
                self.store.event("training.cancel_pending", "Cancellation will apply when HF job identity is recovered", run_id=run_id)
                return run
            settings, token = self._settings()
            if run.get("provider_job_id"):
                if not token:
                    raise ValueError("HF token is required to cancel the remote job")
                await self._call(self.provider_factory(settings, token).cancel, run["provider_job_id"])
            run.update(status="cancelled", finished_at=now())
            self.store.put("training_runs", run)
            self.store.event("training.cancelled", "Training job cancellation requested", run_id=run_id)
            return run

    async def activate(self, candidate_id: str) -> dict:
        async with self._lock:
            run = self.store.get("training_runs", candidate_id)
            if not run or run.get("status") != "passed" or not run.get("checks", {}).get("passed"):
                raise ValueError("Activation requires a candidate with passed training checks")
            if (not run.get("checks", {}).get("independent_evaluation_passed") or
                    run.get("evaluation", {}).get("passed") is not True or
                    run.get("evaluation", {}).get("suite_sha256") != run.get("manifest", {}).get("evaluation", {}).get("sha256")):
                raise ValueError("Activation requires the sealed independent domain/general evaluations")
            if not run.get("calibration", {}).get("passed"):
                raise ValueError("Activation requires measured, passed chamber task calibration")
            if not run.get("artifact_revision") or not run.get("published"):
                raise ValueError("Activation requires a published, revision-pinned adapter")
            if run["manifest"]["base_model"] == SELECTED_SETTINGS["hf_base_model"] and run.get("stage") != "sft":
                raise ValueError("The selected Llama Base has no chat template; live Chamber selection requires a separately validated SFT checkpoint and its tokenizer. CPT remains available for completion-based research.")
            active = self.store.get("runtime", "active_checkpoint")
            if not self.store.get("runtime", "baseline_checkpoint"):
                settings, _token = self._settings()
                self.store.put("runtime", {"id": "baseline_checkpoint", "selection": active or {
                    "base_model": settings["hf_base_model"], "base_revision": settings.get("hf_base_revision"), "adapter_id": None}})
            selection = {"id": "active_checkpoint", "run_id": run["id"], "base_model": run["manifest"]["base_model"],
                         "base_revision": run["manifest"]["base_revision"], "adapter_id": run["manifest"]["output_repo"],
                         "adapter_revision": run["artifact_revision"], "adapter_subfolder": run["manifest"]["output_subfolder"],
                         "tokenizer_id": run["manifest"]["output_repo"], "tokenizer_revision": run["artifact_revision"],
                         "tokenizer_subfolder": run["manifest"]["output_subfolder"],
                         "status": "selected", "remote_endpoint_active": False,
                         "adapter_dose_calibration": {"status": "required", "method": "adapted_model_dose_sweep"}}
            deployment_env = {
                "CHAMBER_MODEL": selection["base_model"], "CHAMBER_MODEL_REVISION": selection["base_revision"],
                "MODEL_ADAPTER_ID": selection["adapter_id"], "MODEL_ADAPTER_REVISION": selection["adapter_revision"],
                "MODEL_ADAPTER_SUBFOLDER": selection["adapter_subfolder"],
                "MODEL_TOKENIZER_ID": selection["tokenizer_id"], "MODEL_TOKENIZER_REVISION": selection["tokenizer_revision"],
                "MODEL_TOKENIZER_SUBFOLDER": selection["tokenizer_subfolder"],
                "CHAMBER_QUANTIZE_4BIT": "true" if run["manifest"]["training"]["mode"] == "qlora" else "false",
                "CHAMBER_DOSE_CAP": "0", "CHAMBER_COHERENT_CAP": "0",
            }
            calibrated_layer = run["calibration"].get("layer")
            if calibrated_layer is None:
                calibrated_layer = (run["calibration"].get("fresh_representation") or {}).get("layer")
            if isinstance(calibrated_layer, int) and not isinstance(calibrated_layer, bool) and calibrated_layer >= 0:
                deployment_env["CHAMBER_LAYER"] = str(calibrated_layer)
            selection.update(deployment_env=deployment_env, selected_at=now())
            self.store.put("runtime", selection)
            for previous in self.store.list_records("checkpoints"):
                if previous.get("status") == "selected" and previous["id"] != run["id"]:
                    previous.update(status="candidate", remote_endpoint_active=False, deselected_at=now())
                    self.store.put("checkpoints", previous)
            checkpoint = self.store.get("checkpoints", run["id"]) or {
                "id": run["id"], "run_id": run["id"], "stage": run["stage"], "checks": run["checks"],
                "calibration": run["calibration"], "repo_id": run["manifest"]["output_repo"],
                "revision": run["artifact_revision"], "subfolder": run["manifest"]["output_subfolder"],
                "base_model": run["manifest"]["base_model"], "base_revision": run["manifest"]["base_revision"],
            }
            checkpoint.update(status="selected", selected_at=selection["selected_at"],
                              remote_endpoint_active=False, deployment_env=deployment_env,
                              adapter_dose_calibration=selection["adapter_dose_calibration"])
            self.store.put("checkpoints", checkpoint)
            self.store.event("checkpoint.selected", "Validated adapter selected; remote chamber reload requires the deployment bridge", run_id=run["id"])
            return selection

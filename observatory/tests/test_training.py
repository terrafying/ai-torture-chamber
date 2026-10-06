from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from observatory.training import HFJobsProvider, TrainingCoordinator, digest
from observatory.train_worker import validate_payload


class MemoryStore:
    def __init__(self):
        self.records, self.events = {}, []
        self.settings = {"training_enabled": True, "hf_token": "hf_private_owner_token", "hf_namespace": "owner",
                         "hf_dataset_repo": "owner/corpus", "hf_model_repo": "owner/adapters",
                         "training_image": "owner/training@sha256:" + "a" * 64, "training_min_documents": 1,
                         "training_min_tokens": 1, "synthetic_training_approved": True,
                         "provider_policy_reference": "owner-reviewed-policy"}

    def get_settings(self, private=False):
        return dict(self.settings)

    def put(self, kind, item):
        self.records[(kind, item["id"])] = copy.deepcopy(item)
        return copy.deepcopy(item)

    def get(self, kind, record_id):
        return copy.deepcopy(self.records.get((kind, record_id)))

    def list_records(self, kind):
        return [copy.deepcopy(value) for (record_kind, _id), value in self.records.items() if record_kind == kind]

    def event(self, event_type, message, agent_id=None, run_id=None, data=None):
        self.events.append({"type": event_type, "message": message, "run_id": run_id, "data": data})


def snapshot(record_id="snapshot-one", change=""):
    original = [
        {"id": "a", "text": "Research on consciousness compares competing theories. " * 12 + change,
         "source_ids": ["source-a"], "family_id": "family-a", "split": "train", "synthetic": False},
        {"id": "b", "text": "Held out research distinguishes sentience from task performance. " * 8,
         "source_ids": ["source-b"], "family_id": "family-b", "split": "validation", "synthetic": False},
    ]
    synthetic = [{"id": "qa-" + row["id"], "messages": [{"role": "user", "content": "What does the source establish?"},
                  {"role": "assistant", "content": "It compares theories; it does not establish sentience."}],
                  "source_ids": row["source_ids"], "family_ids": [row["family_id"]],
                  "evidence_ids": ["evidence-" + row["id"]], "split": row["split"], "synthetic": True} for row in original]
    manifest = {"policy_version": "consciousness-corpus-v2", "quality_gate": {"ready": True, "reasons": []},
                "original_text": original, "synthetic_sft": synthetic}
    return {"id": record_id, "immutable": True, "manifest_hash": digest(manifest), "manifest": manifest,
            "corpus_hash": digest(manifest), "source_ids": ["source-a", "source-b"],
            "heldout_family_ids": ["family-b"], "original_text": original, "synthetic_sft": synthetic}


class FakeProvider:
    calls = []
    stage = "RUNNING"
    final_result = None

    def __init__(self, settings, token):
        self.settings, self.token = settings, token

    def prepare(self, manifest, payload):
        self.calls.append("prepare")
        revision = manifest.get("base_revision") or "base-commit"
        manifest = {**manifest, "base_revision": revision, "tokenizer_revision": revision}
        return {"manifest": manifest, "manifest_sha256": digest(manifest)}

    def run_job(self, bundle):
        self.calls.append("run_job")
        return {"id": "paid-job-id", "url": "https://huggingface.co/jobs/owner/paid-job-id"}

    def inspect_job(self, job_id):
        self.calls.append("inspect")
        return {"status": self.stage}

    def logs(self, job_id):
        return ["real provider log hf_private_owner_token"]

    def result(self, run):
        return self.final_result

    def cancel(self, job_id):
        self.calls.append("cancel")

    def find_job(self, run_id):
        return {"id": "recovered-job"}


class TrainingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.store = MemoryStore()
        self.store.put("datasets", snapshot())
        FakeProvider.calls, FakeProvider.stage, FakeProvider.final_result = [], "RUNNING", None
        self.coordinator = TrainingCoordinator(self.store, provider_factory=FakeProvider)

    async def asyncTearDown(self):
        await self.coordinator.close()

    async def settle(self):
        if self.coordinator._preparations:
            await asyncio.gather(*self.coordinator._preparations)

    async def test_selected_handoff_pin_and_owner_override(self):
        settings, _ = self.coordinator._settings()
        self.assertEqual(settings["hf_base_model"], "meta-llama/Llama-3.1-70B")
        self.assertEqual(settings["hf_base_revision"], "349b2ddb53ce8f2849a6c168a81980ab25258dac")
        self.store.settings["hf_base_model"] = "owner/another-model"
        settings, _ = self.coordinator._settings()
        self.assertIsNone(settings["hf_base_revision"])
        self.store.settings["hf_base_revision"] = "owner-reviewed-revision"
        settings, _ = self.coordinator._settings()
        self.assertEqual(settings["hf_base_revision"], "owner-reviewed-revision")

    async def test_explicit_owner_ref_applies_to_first_cpt_manifest(self):
        self.store.settings["hf_base_revision"] = "owner-reviewed-revision"
        run = await self.coordinator.submit("snapshot-one")
        self.assertEqual(run["manifest"]["base_revision"], "owner-reviewed-revision")
        await self.settle()

    async def test_disabled_never_constructs_provider_or_uploads(self):
        self.store.settings["training_enabled"] = False
        result = await self.coordinator.submit("snapshot-one")
        self.assertIn("training_disabled", result["reasons"])
        self.assertEqual(FakeProvider.calls, [])

    async def test_preparing_response_idempotence_and_no_overlap(self):
        first = await self.coordinator.submit("snapshot-one")
        self.assertEqual(first["status"], "preparing")
        await self.settle()
        same = await self.coordinator.submit("snapshot-one")
        self.assertEqual(same["id"], first["id"])
        self.store.put("datasets", snapshot("snapshot-two", " New distinct research."))
        other = await self.coordinator.submit("snapshot-two")
        self.assertEqual(other["reasons"], ["training_job_in_progress"])
        self.assertEqual(FakeProvider.calls.count("run_job"), 1)
        self.assertNotIn("hf_private_owner_token", json.dumps(self.store.list_records("training_runs")))

    async def test_provenance_only_snapshot_change_does_not_retrain(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        revised = snapshot("snapshot-revised")
        revised["manifest"]["provenance_revision"] = "owner-reviewed-update"
        revised["manifest_hash"] = digest(revised["manifest"])
        self.store.put("datasets", revised)
        second = await self.coordinator.submit("snapshot-revised")
        self.assertEqual(second["id"], first["id"])

    async def test_synthetic_notes_and_schedule_edits_do_not_retrain_cpt(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        revised = snapshot("snapshot-new-qa")
        revised["synthetic_sft"][0]["messages"][1]["content"] += " Additional supported context."
        revised["manifest_hash"] = digest(revised["manifest"])
        self.store.put("datasets", revised)
        self.store.settings["training_interval_hours"] = 8
        result = await self.coordinator.submit(revised["id"])
        self.assertEqual(result["id"], first["id"])
        self.assertEqual(FakeProvider.calls.count("run_job"), 1)

    async def test_explicit_retry_creates_fresh_lineage_default_does_not_retry(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        failed = self.store.get("training_runs", first["id"])
        failed["status"] = "failed"
        self.store.put("training_runs", failed)
        same = await self.coordinator.submit("snapshot-one")
        self.assertEqual(same["id"], first["id"])
        retry = await self.coordinator.submit("snapshot-one", retry=True)
        await self.settle()
        self.assertNotEqual(retry["id"], first["id"])
        self.assertEqual(retry["retry_of"], first["id"])
        again = await self.coordinator.submit("snapshot-one")
        self.assertEqual(again["id"], retry["id"])
        self.assertEqual(FakeProvider.calls.count("run_job"), 2)

    async def test_poll_records_actual_logs_and_unverified_completion(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        run = self.store.get("training_runs", first["id"])
        await self.coordinator._poll(run)
        self.assertEqual(self.store.get("training_runs", first["id"])["status"], "running")
        self.assertNotIn("hf_private_owner_token", json.dumps(self.store.events))
        FakeProvider.stage = "COMPLETED"
        await self.coordinator._poll(self.store.get("training_runs", first["id"]))
        final = self.store.get("training_runs", first["id"])
        self.assertEqual(final["status"], "completed_unverified")
        self.assertFalse(final["checks"]["passed"])

    async def test_failed_job_never_becomes_candidate(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        FakeProvider.stage = "ERROR"
        await self.coordinator._poll(self.store.get("training_runs", first["id"]))
        self.assertEqual(self.store.get("training_runs", first["id"])["status"], "failed")
        self.assertFalse(self.store.list_records("checkpoints"))

    async def test_durable_four_hour_tick(self):
        current = datetime(2026, 10, 5, tzinfo=timezone.utc)
        with patch("observatory.curation.build_snapshot", return_value=snapshot()):
            await self.coordinator.tick(current)
            await self.settle()
            first_tick = self.store.get("runtime", "training_scheduler")["last_tick"]
            second_coordinator = TrainingCoordinator(self.store, provider_factory=FakeProvider)
            await second_coordinator.tick(current + timedelta(hours=3, minutes=59))
            self.assertEqual(self.store.get("runtime", "training_scheduler")["last_tick"], first_tick)
            await second_coordinator.tick(current + timedelta(hours=4))
            self.assertNotEqual(self.store.get("runtime", "training_scheduler")["last_tick"], first_tick)
            self.assertEqual(FakeProvider.calls.count("run_job"), 1)
            await second_coordinator.close()

    async def passed_run(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        run = self.store.get("training_runs", first["id"])
        run.update(status="passed", checks={"passed": True, "independent_evaluation_passed": True}, calibration={"passed": False},
                   evaluation={"passed": True, "status": "measured", "suite_sha256": run["manifest"]["evaluation"]["sha256"]},
                   artifact_revision="adapter-commit", published=True)
        self.store.put("training_runs", run)
        return run

    async def test_activation_requires_calibration_and_preserves_selection(self):
        run = await self.passed_run()
        with self.assertRaisesRegex(ValueError, "calibration"):
            await self.coordinator.activate(run["id"])
        baseline = {"id": "active_checkpoint", "base_model": "existing/chamber", "adapter_id": None}
        self.store.put("runtime", baseline)
        self.store.put("checkpoints", {"id": "old-checkpoint", "status": "selected", "revision": "preserved-commit"})
        run["calibration"] = {"passed": True, "layer": 40}
        run["stage"] = run["manifest"]["stage"] = "sft"
        self.store.put("training_runs", run)
        selection = await self.coordinator.activate(run["id"])
        self.assertFalse(selection["remote_endpoint_active"])
        self.assertEqual(selection["adapter_revision"], "adapter-commit")
        self.assertEqual(self.store.get("runtime", "baseline_checkpoint")["selection"], baseline)
        public = self.store.get("checkpoints", run["id"])
        self.assertEqual(public["status"], "selected")
        self.assertFalse(public["remote_endpoint_active"])
        self.assertEqual(public["deployment_env"]["MODEL_ADAPTER_REVISION"], "adapter-commit")
        self.assertEqual(public["deployment_env"]["CHAMBER_QUANTIZE_4BIT"], "true")
        self.assertEqual(public["deployment_env"]["CHAMBER_LAYER"], "40")
        self.assertEqual(public["deployment_env"]["CHAMBER_DOSE_CAP"], "0")
        self.assertEqual(public["deployment_env"]["CHAMBER_COHERENT_CAP"], "0")
        self.assertEqual(public["adapter_dose_calibration"]["status"], "required")
        old = self.store.get("checkpoints", "old-checkpoint")
        self.assertEqual(old["status"], "candidate")
        self.assertEqual(old["revision"], "preserved-commit")
        self.assertNotIn("hf_private_owner_token", json.dumps(public))

    async def test_older_calibration_layer_is_used_only_when_recorded(self):
        run = await self.passed_run()
        run["stage"] = run["manifest"]["stage"] = "sft"
        run["calibration"] = {"passed": True, "fresh_representation": {"layer": 12}}
        self.store.put("training_runs", run)
        selection = await self.coordinator.activate(run["id"])
        self.assertEqual(selection["deployment_env"]["CHAMBER_LAYER"], "12")
        run["calibration"] = {"passed": True}
        self.store.put("training_runs", run)
        selection = await self.coordinator.activate(run["id"])
        self.assertNotIn("CHAMBER_LAYER", selection["deployment_env"])

    async def test_selected_base_cpt_cannot_be_exported_as_live_chat(self):
        run = await self.passed_run()
        run["calibration"] = {"passed": True, "layer": 40}
        self.store.put("training_runs", run)
        with self.assertRaisesRegex(ValueError, "SFT checkpoint"):
            await self.coordinator.activate(run["id"])
        self.assertIsNone(self.store.get("runtime", "active_checkpoint"))

    async def test_sft_is_manual_separate_and_reuses_pinned_cpt_base(self):
        run = await self.passed_run()
        self.store.settings["hf_base_model"] = "changed/owner-selection"
        sft = await self.coordinator.submit_sft(run["id"], snapshot_id="snapshot-one")
        await self.settle()
        self.assertEqual(sft["stage"], "sft")
        self.assertNotEqual(sft["id"], run["id"])
        self.assertEqual(sft["parent_run_id"], run["id"])
        self.assertEqual(sft["manifest"]["base_model"], run["manifest"]["base_model"])
        self.assertEqual(sft["manifest"]["parent_adapter"]["revision"], "adapter-commit")

    async def test_sft_requires_explicit_synthetic_policy(self):
        run = await self.passed_run()
        self.store.settings["synthetic_training_approved"] = False
        result = await self.coordinator.submit_sft(run["id"])
        self.assertEqual(result["status"], "not_ready")

    async def test_cpt_continues_previous_adapter_only_for_new_corpus(self):
        parent = await self.passed_run()
        same = await self.coordinator.submit("snapshot-one")
        self.assertEqual(same["id"], parent["id"])
        self.store.put("datasets", snapshot("snapshot-new", " Fresh primary evidence."))
        run = await self.coordinator.submit("snapshot-new")
        await self.settle()
        self.assertEqual(run["parent_run_id"], parent["id"])
        self.assertEqual(run["manifest"]["parent_adapter"]["revision"], parent["artifact_revision"])
        self.assertEqual(run["manifest"]["base_revision"], parent["manifest"]["base_revision"])

    async def test_sft_can_use_later_approved_snapshot(self):
        parent = await self.passed_run()
        latest = snapshot("snapshot-with-approved-qa", " Additional approved source.")
        with patch("observatory.curation.build_snapshot", return_value=latest):
            run = await self.coordinator.submit_sft(parent["id"])
        await self.settle()
        self.assertEqual(run["snapshot_id"], latest["id"])
        self.assertEqual(run["parent_run_id"], parent["id"])

    async def test_remote_cancellation_uses_provider(self):
        first = await self.coordinator.submit("snapshot-one")
        await self.settle()
        cancelled = await self.coordinator.cancel(first["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertIn("cancel", FakeProvider.calls)

    async def test_unknown_submission_is_recovered_without_resubmitting(self):
        run = await self.passed_run()
        run.update(status="submission_unknown", cancel_requested=True)
        run.pop("provider_job_id", None)
        self.store.put("training_runs", run)
        self.store.settings["training_enabled"] = False
        await self.coordinator.tick()
        recovered = self.store.get("training_runs", run["id"])
        self.assertEqual(recovered["status"], "cancelled")
        self.assertEqual(recovered["provider_job_id"], "recovered-job")
        self.assertEqual(FakeProvider.calls.count("run_job"), 1)


class DataContractTests(unittest.TestCase):
    def test_original_text_and_sft_never_mix(self):
        data = snapshot()
        cpt, sft = validate_payload(data, "cpt"), validate_payload(data, "sft")
        self.assertTrue(all("text" in item and "messages" not in item for item in cpt["train"]))
        self.assertTrue(all("messages" in item and "text" not in item for item in sft["train"]))

    def test_family_leakage_and_missing_sources_are_rejected(self):
        data = snapshot()
        data.pop("manifest")
        data["original_text"][1]["family_id"] = "family-a"
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_payload(data, "cpt")
        data = snapshot()
        data.pop("manifest")
        data["original_text"][0]["source_ids"] = ["not-approved"]
        with self.assertRaisesRegex(ValueError, "unapproved"):
            validate_payload(data, "cpt")

    def test_modified_manifest_is_rejected(self):
        data = snapshot()
        data["manifest"]["original_text"][0]["text"] += "tamper"
        with self.assertRaisesRegex(ValueError, "hash_mismatch"):
            validate_payload(data, "cpt")

    def test_hf_status_enum_is_normalized_and_log_poll_does_not_follow(self):
        from enum import Enum
        class JobStage(str, Enum):
            COMPLETED = "COMPLETED"
        provider = object.__new__(HFJobsProvider)
        provider.namespace = "owner"
        from unittest.mock import Mock
        provider.api = Mock()
        provider.api.inspect_job.return_value = {"status": {"stage": JobStage.COMPLETED}}
        provider.api.fetch_job_logs.return_value = iter(["a"])
        self.assertEqual(provider.inspect_job("id")["status"], "COMPLETED")
        self.assertEqual(provider.logs("id"), ["a"])
        provider.api.fetch_job_logs.assert_called_once_with(job_id="id", namespace="owner", follow=False, tail=100)

    def test_submission_only_uses_hf_hub_1_16_supported_arguments(self):
        from types import SimpleNamespace
        captured = {}
        class Api:
            def run_job(self, *, image, command, env, secrets, flavor, timeout, labels, namespace):
                captured.update(locals())
                return SimpleNamespace(id="job", url="https://huggingface.co/jobs/owner/job")
        provider = object.__new__(HFJobsProvider)
        provider.settings = {"training_image": "pinned-image", "training_hardware": "a100-large", "training_timeout_seconds": 14400}
        provider.token, provider.namespace, provider.api = "private-token", "owner", Api()
        bundle = {"manifest": {"run_id": "run", "stage": "cpt"}, "manifest_repo": "owner/data",
                  "manifest_path": "manifest.json", "manifest_revision": "commit", "manifest_sha256": "hash"}
        self.assertEqual(provider.run_job(bundle)["id"], "job")
        self.assertEqual(captured["secrets"], {"HF_TOKEN": "private-token"})
        self.assertNotIn("HF_TOKEN", captured["env"])
        self.assertEqual(captured["labels"]["observatory_run_id"], "run")

    def test_unsupported_full_precision_70b_lora_rejected_before_upload_or_gpu(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        provider = object.__new__(HFJobsProvider)
        provider.api = Mock()
        provider.api.model_info.return_value = SimpleNamespace(sha="base", safetensors=SimpleNamespace(total=70_553_706_496))
        manifest = {"base_model": "meta-llama/Llama-3.1-70B", "stage": "cpt", "training": {"mode": "lora"}}
        with self.assertRaisesRegex(ValueError, "require QLoRA"):
            provider.prepare(manifest, snapshot())
        provider.api.create_repo.assert_not_called()
        provider.api.run_job.assert_not_called()


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "trl", "datasets")), "optional model dependencies unavailable")
class TinyRealTrainingTests(unittest.TestCase):
    def test_real_cpt_and_sft_adapter_save_reload_and_hook(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from observatory.train_worker import train_adapter, calibrate_chamber
        from observatory.train_worker import SFT_TEMPLATE, render_sft_text
        from painlab.models.hf_model import HFModel
        from painlab.models.hooks import SteeringHook
        import numpy as np

        torch.set_num_threads(2)
        vocabulary = {token: index for index, token in enumerate(["<unk>", "<pad>", "<eos>", "A", "B", "Research", "on", "consciousness", "compares", "competing", "theories", ".", "Held", "out", "research", "distinguishes", "sentience", "from", "task", "performance", "What", "does", "the", "source", "establish", "?", "It", "it", "not", "user", "assistant", ":", ";"])}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
        model = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
            pad_token_id=1, eos_token_id=2, attention_dropout=0))
        data = snapshot()
        tokenizer.chat_template = SFT_TEMPLATE
        messages = data["synthetic_sft"][0]["messages"]
        self.assertEqual(tokenizer.apply_chat_template(messages, tokenize=False), render_sft_text(messages, tokenizer.eos_token))
        cfg = {"mode": "lora", "seed": 42, "max_steps": 2, "sequence_length": 64, "batch_size": 1,
               "gradient_accumulation": 1, "learning_rate": 0.001, "lora_rank": 2, "lora_alpha": 4,
               "min_tokens": 1, "max_loss_ratio": 2, "calibration_episodes": 0,
               # Random tiny models test real scorer/trainer/save plumbing, not
               # scientific capability. Production thresholds remain strict.
               "evaluation_policy": {"max_accuracy_drop": 1.0, "max_nll_ratio": 2.0,
                                     "min_domain_accuracy": 0.0, "min_general_accuracy": 0.0,
                                     "max_length": 128}}
        manifest = {"run_id": "tiny-cpt", "stage": "cpt", "snapshot_hash": data["manifest_hash"],
                    "snapshot_id": data["id"], "base_model": "tiny/local", "base_revision": "fixture",
                    "tokenizer_revision": "fixture", "training": cfg, "parent_adapter": None}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base"
            model.save_pretrained(base_path)
            tokenizer.save_pretrained(base_path)
            result = train_adapter(manifest, data, root / "cpt", model=model, tokenizer=tokenizer, run_calibration=False)
            self.assertTrue(result["checks"]["passed"])
            self.assertGreater(result["metrics"]["adapter_parameter_squared_change"], 0)
            self.assertEqual(result["metrics"]["trained_steps"], 2)
            self.assertFalse(result["calibration"]["passed"])
            self.assertTrue(result["checks"]["independent_evaluation_passed"])
            self.assertEqual(set(result["evaluation"]["controls"]), {"unadapted_base", "incoming_parent", "trained_candidate"})
            self.assertTrue((root / "cpt/evaluation/summary.json").is_file())
            loaded = HFModel.from_pretrained(str(base_path), adapter_id=str(root / "cpt/adapter"), device="cpu", dtype="float32")
            before = loaded.logits("Research on consciousness")
            layer = loaded.layer_module(1)
            with SteeringHook(layer, np.ones(32, dtype=np.float32), scale=0.5):
                during = loaded.logits("Research on consciousness")
            after = loaded.logits("Research on consciousness")
            self.assertFalse(np.allclose(before, during))
            np.testing.assert_allclose(before, after, atol=1e-7)
            self.assertTrue(np.isfinite(loaded.perplexity("Research on consciousness compares theories .")))
            self.assertEqual(loaded.final_token_hidden("Research on consciousness", 1).shape, (32,))
            calibration_manifest = {**manifest, "training": {**cfg, "calibration_episodes": 20,
                                                           "calibration_min_rate": 0.65}}
            calibration = calibrate_chamber(loaded.model, loaded.tokenizer, calibration_manifest, root / "cpt")
            self.assertEqual(calibration["status"], "measured")
            self.assertEqual(calibration["layer"], 1)
            self.assertEqual(calibration["fresh_representation"]["layer"], 1)
            saved_calibration = json.loads((root / "cpt/calibration/summary.json").read_text())
            self.assertEqual(saved_calibration["layer"], 1)
            # The same real trainer exercises assistant-only SFT as a distinct artifact.
            sft_manifest = {**manifest, "run_id": "tiny-sft", "stage": "sft", "parent_adapter": {
                "repo_id": str(root / "cpt/adapter"), "revision": None, "subfolder": "", "run_id": "tiny-cpt"}}
            sft_base = LlamaForCausalLM.from_pretrained(base_path)
            sft_result = train_adapter(sft_manifest, data, root / "sft", model=sft_base, tokenizer=loaded.tokenizer, run_calibration=False)
            self.assertTrue(sft_result["checks"]["passed"])
            lineage = json.loads((root / "sft/adapter/observatory_lineage.json").read_text())
            self.assertEqual(lineage["objective"], "supported_messages_assistant_only")

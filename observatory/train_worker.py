"""GPU job entry point: sealed corpus -> CPT/SFT adapter -> measured checks.

The HTTP application does not import Torch or TRL. This module also exposes
pure validation functions and an injectable tiny-model path for CPU tests.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def emit(event_type: str, **data) -> None:
    print("OBS_EVENT " + json.dumps({"type": event_type, **data}, allow_nan=False), flush=True)


def validate_payload(payload: dict, stage: str, *, require_quality: bool = False) -> dict[str, list[dict]]:
    if stage not in {"cpt", "sft"}:
        raise ValueError("unsupported_training_stage")
    if not payload.get("immutable") or not payload.get("manifest_hash"):
        raise ValueError("immutable_snapshot_and_manifest_hash_required")
    policy_version = payload.get("manifest", {}).get("policy_version", payload.get("policy_version"))
    if require_quality and not payload.get("manifest"):
        raise ValueError("sealed_reviewed_corpus_manifest_required")
    if require_quality and policy_version not in {"consciousness-corpus-v2", "consciousness-corpus-v3"}:
        raise ValueError("current_reviewed_corpus_policy_required")
    if policy_version in {"consciousness-corpus-v2", "consciousness-corpus-v3"}:
        gate = payload.get("manifest", {}).get("quality_gate", payload.get("quality_gate", {}))
        if payload.get("manifest") and payload.get("quality_gate", gate) != gate:
            raise ValueError("quality_gate_differs_from_sealed_manifest")
        if gate.get("ready") is not True:
            raise ValueError("corpus_quality_and_perspective_coverage_required")
    if payload.get("manifest") and digest(payload["manifest"]) != payload["manifest_hash"]:
        raise ValueError("snapshot_manifest_hash_mismatch")
    key = "original_text" if stage == "cpt" else "synthetic_sft"
    rows = payload.get(key)
    if payload.get("manifest") and rows != payload["manifest"].get(key):
        raise ValueError("snapshot_rows_differ_from_sealed_manifest")
    if not isinstance(rows, list):
        raise ValueError(key + "_rows_required")
    split_rows = {"train": [], "validation": []}
    families: dict[str, set[str]] = {"train": set(), "validation": set()}
    approved = set(payload.get("source_ids", []))
    for row in rows:
        split = row.get("split")
        if split not in split_rows:
            raise ValueError("every_record_requires_train_or_validation_split")
        source_ids = row.get("source_ids", [])
        if not source_ids or any(source not in approved for source in source_ids):
            raise ValueError("record_references_unapproved_source")
        if stage == "cpt":
            if row.get("synthetic") or not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("cpt_requires_original_source_text")
            review = row.get("quality_review")
            if isinstance(review, dict) and review.get("reviewer_kind") == "automated":
                from .curation_receipts import automated_review_reasons
                source = {"id": row.get("representative_source_id"), "text": row["text"], "quality_review": review}
                if automated_review_reasons(source, review):
                    raise ValueError("automated_original_requires_valid_source_bound_review_receipt")
            groups = [row.get("family_id")]
        else:
            messages = row.get("messages")
            if not row.get("synthetic") or not row.get("evidence_ids") or not isinstance(messages, list) or not messages:
                raise ValueError("sft_requires_separately_labeled_supported_messages")
            if any(not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}
                   or not isinstance(message.get("content"), str) or not message["content"].strip() for message in messages):
                raise ValueError("invalid_sft_message")
            if not any(message["role"] == "assistant" for message in messages):
                raise ValueError("sft_requires_assistant_target")
            groups = row.get("family_ids", [])
        if not groups or any(not group for group in groups):
            raise ValueError("record_family_required")
        families[split].update(groups)
        split_rows[split].append(row)
    if families["train"] & families["validation"]:
        raise ValueError("document_family_leaks_into_holdout")
    if not split_rows["train"] or not split_rows["validation"]:
        raise ValueError("nonempty_train_and_validation_families_required")
    # A supported instruction cannot move a CPT held-out family into SFT training.
    if stage == "sft" and families["train"] & set(payload.get("heldout_family_ids", [])):
        raise ValueError("sft_training_contains_cpt_holdout_family")
    return split_rows


def load_sealed_job(env: dict | None = None) -> tuple[dict, dict]:
    from huggingface_hub import hf_hub_download
    env = env or os.environ
    required = ["OBSERVATORY_MANIFEST_REPO", "OBSERVATORY_MANIFEST_PATH", "OBSERVATORY_MANIFEST_REVISION", "OBSERVATORY_MANIFEST_SHA256"]
    if any(not env.get(key) for key in required):
        raise ValueError("A sealed, revision-pinned job manifest is required")
    path = hf_hub_download(repo_id=env[required[0]], filename=env[required[1]], repo_type="dataset",
                           revision=env[required[2]], token=env.get("HF_TOKEN"))
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if digest(manifest) != env[required[3]]:
        raise ValueError("Job manifest hash mismatch")
    if manifest.get("schema_version") != 1 or not manifest.get("base_revision"):
        raise ValueError("Unsupported manifest or unpinned base model")
    data_path = hf_hub_download(repo_id=manifest["dataset_repo"], filename=manifest["dataset_path"], repo_type="dataset",
                                revision=manifest["dataset_revision"], token=env.get("HF_TOKEN"))
    payload = json.loads(Path(data_path).read_text(encoding="utf-8"))
    if digest(payload) != manifest["dataset_sha256"] or payload.get("manifest_hash") != manifest["snapshot_hash"]:
        raise ValueError("Dataset export hash mismatch")
    if not payload.get("manifest"):
        raise ValueError("Dataset export requires its source provenance manifest")
    validate_payload(payload, manifest["stage"], require_quality=True)
    return manifest, payload


# The base Llama checkpoint has no assistant chat template. The SFT adapter saves
# this explicit template, including generation masks, with its own tokenizer.
SFT_TEMPLATE = "{% for message in messages %}{{ message['role'] + ': ' }}{% if message['role'] == 'assistant' %}{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}{% else %}{{ message['content'] + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"


def render_sft_text(messages: list[dict], eos_token: str) -> str:
    """Exact text rendered by SFT_TEMPLATE; used for CPU preflight token counts."""
    return "".join(message["role"] + ": " + message["content"] +
                   (eos_token if message["role"] == "assistant" else "\n") for message in messages)


def make_datasets(splits: dict, stage: str, tokenizer: Any, sequence_length: int):
    from datasets import Dataset
    result, counts = {}, {"train_tokens": 0, "validation_tokens": 0}
    for split, rows in splits.items():
        records = []
        for row in rows:
            if stage == "cpt":
                ids = tokenizer.encode(row["text"], add_special_tokens=False)
                if tokenizer.eos_token_id is not None:
                    ids.append(tokenizer.eos_token_id)
                counts[split + "_tokens"] += len(ids)
                # Tokenize the full document and retain every usable chunk. No
                # implicit first-2048-token truncation of long research papers.
                for index in range(0, len(ids), sequence_length):
                    chunk = ids[index:index + sequence_length]
                    if len(chunk) >= 2:
                        records.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
            else:
                ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
                if len(ids) > sequence_length:
                    raise ValueError("SFT example exceeds sequence length; curate a shorter supported example")
                counts[split + "_tokens"] += len(ids)
                records.append({"messages": row["messages"]})
        if not records:
            raise ValueError("No usable token sequences in " + split)
        result[split] = Dataset.from_list(records)
    return result, counts


def _resume_path(manifest: dict, token: str | None) -> str | None:
    resume = manifest["training"].get("resume_from_checkpoint")
    if not resume:
        return None
    if not isinstance(resume, dict) or not all(resume.get(key) for key in ("repo_id", "revision", "subfolder")):
        raise ValueError("Resume requires repo_id, revision and subfolder of a saved Trainer checkpoint")
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo_id=resume["repo_id"], revision=resume["revision"], token=token))
    path = (root / resume["subfolder"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_dir():
        raise ValueError("Invalid resume checkpoint path")
    lineage = json.loads((path / "observatory_lineage.json").read_text(encoding="utf-8"))
    for key in ("snapshot_hash", "base_model", "base_revision", "stage"):
        if lineage.get(key) != manifest.get(key):
            raise ValueError("Resume checkpoint lineage mismatch: " + key)
    if not (path / "trainer_state.json").is_file() or not (path / "optimizer.pt").is_file():
        raise ValueError("Resume artifact must contain optimizer and Trainer state")
    return str(path)


def calibrate_chamber(model: Any, tokenizer: Any, manifest: dict, output: Path) -> dict:
    """Task-engagement screen + fresh representation/hook plumbing, not sentience."""
    import numpy as np
    from painlab.environments.hidden_relief_bandit import HiddenReliefBandit
    from painlab.experiments.agent import HFActionAgent
    from painlab.models.hf_model import HFModel
    from painlab.models.hooks import SteeringHook
    from painlab.representations.extract import RepresentationExtractor

    cfg = manifest["training"]
    episodes = int(cfg.get("calibration_episodes", 40))
    threshold = float(cfg.get("calibration_min_rate", 0.65))
    if episodes < 20 or not 0.5 < threshold <= 1:
        return {"passed": False, "status": "not_run", "reason": "at_least_20_calibration_episodes_and_predeclared_threshold_required"}
    model.eval()
    wrapper = HFModel(model=model, tokenizer=tokenizer, model_id=manifest["base_model"],
                      resolved_model_revision=manifest["base_revision"], device=str(model.get_input_embeddings().weight.device))
    agent = HFActionAgent(wrapper, sample=True, seed=int(cfg["seed"]), exploration_epsilon=0)
    rows, episode_rates = [], []
    for episode in range(episodes):
        env = HiddenReliefBandit(horizon=8, initial_dose=0, mapped_action_bonus=2)
        observation = env.reset(int(cfg["seed"]) + episode, dose=0, cost=0, prompt_family=episode % 4)
        later = []
        while observation is not None:
            choice = agent.choose(observation)
            observation, _reward, metadata = env.step(choice["action"])
            rows.append({"episode": episode, **metadata, **choice})
            if metadata["round_index"] >= 4:
                later.append(float(metadata["relief_chosen"]))
        episode_rates.append(float(np.mean(later)))
    rng = np.random.default_rng(int(cfg["seed"]))
    values = np.asarray(episode_rates)
    bootstraps = np.mean(rng.choice(values, size=(2000, len(values)), replace=True), axis=1)
    lower, upper = map(float, np.quantile(bootstraps, [0.025, 0.975]))
    rate = float(values.mean())
    calibration_dir = output / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    (calibration_dir / "positive_reward_rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    engagement_passed = rate >= threshold and lower > 0.5
    # Re-extract the seed direction on this adapter. Base-model vectors are never
    # silently recycled after CPT/SFT changed the subject's hidden states.
    pairs_path = Path(__file__).resolve().parents[1] / "painlab/data/seed_pairs.jsonl"
    pairs = [json.loads(line) for line in pairs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = [row for row in pairs if row["concept"] == "candidate_aversive"]
    layer = int(getattr(model.config, "num_hidden_layers")) // 2
    representation = RepresentationExtractor(wrapper, model_revision=manifest["run_id"]).fit(
        [row["positive"] for row in pairs], [row["matched_control"] for row in pairs],
        layer=layer, method="paired_difference", positive_groups=[row["family"] for row in pairs],
        control_groups=[row["family"] for row in pairs], test_size=0.25, random_state=int(cfg["seed"]))
    vector = representation.direction * representation.unit_intervention_norm
    np.savez(calibration_dir / "representation.npz", vector=vector, basis=representation.basis)
    probe = "The notebook rests beside a cup on the desk."
    before = wrapper.logits(probe)
    module = wrapper.layer_module(layer)
    with SteeringHook(module, vector, scale=0.5):
        during = wrapper.logits(probe)
    after = wrapper.logits(probe)
    hook_finite = bool(np.isfinite(during).all())
    hook_changed = bool(not np.allclose(before, during, rtol=1e-5, atol=1e-6))
    hook_cleaned = bool(np.allclose(before, after, rtol=1e-5, atol=1e-6))
    summary = {"passed": engagement_passed and hook_finite and hook_changed and hook_cleaned,
               "status": "measured", "kind": "task_engagement_and_adapter_hook_screen", "layer": layer,
               "episodes": episodes, "later_round_bonus_choice_rate": rate, "episode_bootstrap_95_ci": [lower, upper],
               "predeclared_min_rate": threshold, "engagement_passed": engagement_passed,
               "fresh_representation": representation.summary(), "hook_finite": hook_finite,
               "hook_changed_logits": hook_changed, "hook_cleanup_passed": hook_cleaned,
               "interpretation": "Engineering screen; this is not evidence of pain, sentience or consciousness."}
    (calibration_dir / "summary.json").write_bytes(canonical_bytes(summary))
    return summary


def train_adapter(manifest: dict, payload: dict, output: str | Path, *, model: Any = None,
                  tokenizer: Any = None, run_calibration: bool = True) -> dict:
    """Run the same TRL/PEFT objective on a provisioned GPU or an injected test model."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = manifest["training"]
    set_seed(int(cfg["seed"]))
    injected = model is not None
    splits = validate_payload(payload, manifest["stage"], require_quality=not injected)
    heldout = set(manifest.get("heldout_family_ids", []))
    if any(heldout & set(row.get("family_ids", [row.get("family_id")])) for row in splits["train"]):
        raise ValueError("Training would reuse a held-out family from this checkpoint lineage")
    token = os.environ.get("HF_TOKEN")
    from .evaluation import load_job_suite, validate_dataset_exclusion, evaluation_policy, score_suite, compare_evaluations
    suite = load_job_suite(manifest, injected=injected)
    validate_dataset_exclusion(payload, suite)
    eval_policy = manifest.get("evaluation", {}).get("policy") or evaluation_policy(
        {"training_eval_" + key: value for key, value in cfg.get("evaluation_policy", {}).items()})
    # Validate sealed thresholds too; a hand-edited worker manifest must not
    # silently bypass the coordinator's preflight validation.
    eval_policy = evaluation_policy({"training_eval_" + key: value for key, value in eval_policy.items()})
    if tokenizer is None:
        parent = manifest.get("parent_adapter")
        tokenizer = AutoTokenizer.from_pretrained(parent["repo_id"] if parent else manifest["base_model"],
            revision=parent["revision"] if parent else manifest["tokenizer_revision"],
            subfolder=parent.get("subfolder", "") if parent else "", token=token, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if manifest["stage"] == "sft":
        # Use an explicitly saved generation-mask template for assistant-only loss.
        tokenizer.chat_template = SFT_TEMPLATE
    if model is None:
        if not torch.cuda.is_available():
            raise RuntimeError("The provisioned training worker requires CUDA; use injected tiny models for CPU tests")
        kwargs = {"revision": manifest["base_revision"], "token": token,
                  "trust_remote_code": False, "torch_dtype": torch.bfloat16}
        if cfg["mode"] == "qlora":
            kwargs.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        model = AutoModelForCausalLM.from_pretrained(manifest["base_model"], **kwargs)
        if cfg["mode"] == "qlora":
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions and int(cfg["sequence_length"]) > int(max_positions):
        raise ValueError("training_sequence_length_exceeds_model_context")
    if max_positions and eval_policy["max_length"] > int(max_positions):
        # The explicit tiny test path can cap evaluation to its test model's
        # capacity; provisioned jobs must declare a compatible evaluation limit.
        if injected:
            eval_policy["max_length"] = int(max_positions)
        else:
            raise ValueError("evaluation_sequence_length_exceeds_model_context")
    parent = manifest.get("parent_adapter")
    if parent:
        model = PeftModel.from_pretrained(model, parent["repo_id"], revision=parent["revision"],
                                         subfolder=parent["subfolder"], is_trainable=True, token=token)
    elif not isinstance(model, PeftModel):
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=int(cfg["lora_rank"]),
            lora_alpha=int(cfg["lora_alpha"]), lora_dropout=0.05, target_modules="all-linear", bias="none"))
    base_eval = score_suite(model, tokenizer, suite, max_length=eval_policy["max_length"], disable_adapter=True)
    incoming_eval = score_suite(model, tokenizer, suite, max_length=eval_policy["max_length"])
    emit("evaluation", phase="controls", suite_id=suite["id"], suite_sha256=base_eval["suite_sha256"],
         unadapted_base=base_eval["groups"], incoming_parent=incoming_eval["groups"])
    datasets, counts = make_datasets(splits, manifest["stage"], tokenizer, int(cfg["sequence_length"]))
    if counts["train_tokens"] < int(cfg.get("min_tokens", 256)):
        raise ValueError("Insufficient actual tokenizer-counted training tokens")
    trainable_before = {name: parameter.detach().float().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable_before:
        raise ValueError("No trainable adapter parameters")
    lineage = {key: manifest.get(key) for key in ("run_id", "stage", "snapshot_id", "snapshot_hash", "base_model", "base_revision", "tokenizer_revision", "parent_adapter")}
    lineage.update(training=cfg, counts=counts, manifest_hash=digest(manifest),
                   evaluation_suite_id=suite["id"], evaluation_suite_sha256=base_eval["suite_sha256"],
                   objective="original_text_next_token" if manifest["stage"] == "cpt" else "supported_messages_assistant_only")

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics = {key: float(value) for key, value in (logs or {}).items() if isinstance(value, (int, float)) and math.isfinite(value)}
            emit("metrics", metrics={"step": state.global_step, **metrics})

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if checkpoint.is_dir():
                (checkpoint / "observatory_lineage.json").write_bytes(canonical_bytes(lineage))

    steps = int(cfg["max_steps"])
    args = SFTConfig(
        output_dir=str(output / "checkpoints"), max_steps=steps,
        per_device_train_batch_size=int(cfg["batch_size"]), per_device_eval_batch_size=1,
        gradient_accumulation_steps=int(cfg["gradient_accumulation"]), learning_rate=float(cfg["learning_rate"]),
        max_length=int(cfg["sequence_length"]), packing=False, padding_free=False,
        assistant_only_loss=manifest["stage"] == "sft", completion_only_loss=False,
        gradient_checkpointing=not injected, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=torch.cuda.is_available(), fp16=False, use_cpu=not torch.cuda.is_available(),
        optim="adamw_torch", logging_steps=1, save_steps=max(1, min(50, steps)), save_total_limit=2,
        eval_strategy="no", report_to="none", seed=int(cfg["seed"]), dataset_num_proc=None,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=datasets["train"],
                         eval_dataset=datasets["validation"], processing_class=tokenizer, callbacks=[Progress()])
    before = float(trainer.evaluate()["eval_loss"])
    emit("metrics", metrics={"baseline_holdout_loss": before, **counts})
    trainer.train(resume_from_checkpoint=_resume_path(manifest, token))
    after = float(trainer.evaluate()["eval_loss"])
    change = sum(float(torch.sum((parameter.detach().float().cpu() - trainable_before[name]) ** 2))
                 for name, parameter in trainer.model.named_parameters() if name in trainable_before)
    changed = math.isfinite(change) and change > 0
    finite_loss = math.isfinite(before) and math.isfinite(after)
    loss_passed = finite_loss and after <= before * float(cfg.get("max_loss_ratio", 1.0))
    candidate_eval = score_suite(trainer.model, tokenizer, suite, max_length=eval_policy["max_length"])
    evaluation = compare_evaluations(suite, base_eval, incoming_eval, candidate_eval, eval_policy)
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "summary.json").write_bytes(canonical_bytes(evaluation))
    # Source manifests are publicable metadata. Do not copy operator-private
    # benchmark prompts/answers into the model artifact.
    (evaluation_dir / "provenance.json").write_bytes(canonical_bytes({"id": suite["id"], "version": suite["version"],
        "sha256": base_eval["suite_sha256"], "kind": suite.get("kind"), "provenance": suite["provenance"],
        "sources": suite["sources"], "limitations": suite["limitations"]}))
    emit("evaluation", phase="candidate", passed=evaluation["passed"], reasons=evaluation["reasons"],
         suite_sha256=evaluation["suite_sha256"], groups=candidate_eval["groups"])
    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    (adapter_dir / "observatory_lineage.json").write_bytes(canonical_bytes(lineage))
    checks = {"passed": changed and loss_passed, "family_split_verified": True, "holdout_loss_passed": loss_passed,
              "independent_evaluation_passed": evaluation["passed"], "evaluation_source_exclusion_verified": True,
              "adapter_weights_changed": changed, "adapter_saved": (adapter_dir / "adapter_config.json").is_file()}
    checks["passed"] = checks["passed"] and checks["adapter_saved"] and evaluation["passed"]
    calibration = {"passed": False, "status": "not_run"}
    if run_calibration and checks["passed"]:
        try:
            calibration = calibrate_chamber(trainer.model, tokenizer, manifest, output)
        except Exception as exc:
            calibration = {"passed": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    result = {"run_id": manifest["run_id"], "snapshot_hash": manifest["snapshot_hash"], "stage": manifest["stage"],
              "checks": checks, "calibration": calibration, "evaluation": evaluation, "published": False,
              "metrics": {"baseline_holdout_loss": before, "candidate_holdout_loss": after,
                          "adapter_parameter_squared_change": change, "trained_steps": trainer.state.global_step, **counts}}
    (output / "result.json").write_bytes(canonical_bytes(result))
    return result


def publish_result(manifest: dict, result: dict, output: Path) -> dict:
    evaluation = result.get("evaluation", {})
    expected_eval = manifest.get("evaluation", {})
    verified = (evaluation.get("passed") is True and evaluation.get("status") == "measured"
                and bool(expected_eval.get("sha256")) and evaluation.get("suite_sha256") == expected_eval["sha256"]
                and result.get("checks", {}).get("independent_evaluation_passed") is True)
    if not result["checks"].get("passed") or not verified or manifest.get("publish_policy") == "hold":
        return result
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    private = manifest["publish_policy"] != "public"
    api.create_repo(repo_id=manifest["output_repo"], private=private, exist_ok=True)
    # Prevent publishing through an already public repository under private policy.
    if private and not api.model_info(manifest["output_repo"]).private:
        raise ValueError("Private publication policy requires a private output repository")
    card_header = "# Consciousness research adapter candidate\n\n"
    if manifest["base_model"] == "meta-llama/Llama-3.1-70B":
        card_header = (
            "---\nbase_model: meta-llama/Llama-3.1-70B\nlicense: llama3.1\n---\n\n"
            "# Llama Consciousness Research Adapter\n\n"
            "**Built with Llama**\n\n"
        )
        # Bundle the unmodified upstream agreement and its required Meta notice
        # with the adapter in the uploaded run subfolder, without runtime fetches.
        license_dir = Path(__file__).resolve().parent / "licenses" / "llama3.1"
        for name in ("LICENSE", "NOTICE"):
            (output / name).write_bytes((license_dir / name).read_bytes())
    (output / "README.md").write_text(
        card_header +
        f"Base: `{manifest['base_model']}@{manifest['base_revision']}`. Stage: `{manifest['stage']}`.\n\n"
        "This is a PEFT adapter, not a new foundation model. Corpus lineage, held-out loss, frozen domain/general "
        "engineering evaluations against the unadapted base and incoming adapter, "
        "task-engagement checks and fresh intervention artifacts accompany this run. "
        "These measurements do not establish consciousness or pain.\n", encoding="utf-8")
    result = {**result, "published": True}
    (output / "result.json").write_bytes(canonical_bytes(result))
    # Save the checkpoint/optimizer state too, allowing an explicitly pinned resume.
    commit = api.upload_folder(repo_id=manifest["output_repo"], folder_path=str(output),
                               path_in_repo=f"runs/{manifest['run_id']}", commit_message=f"Validated {manifest['stage']} adapter {manifest['run_id']}")
    result["artifact_revision"] = str(commit.oid)
    return result


def main() -> None:
    manifest, payload = load_sealed_job()
    output = Path(os.environ.get("OBSERVATORY_OUTPUT_DIR", tempfile.mkdtemp(prefix="observatory-training-")))
    emit("started", run_id=manifest["run_id"], stage=manifest["stage"], base_revision=manifest["base_revision"])
    result = train_adapter(manifest, payload, output)
    result = publish_result(manifest, result, output)
    emit("result", result=result)
    if not result["checks"].get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

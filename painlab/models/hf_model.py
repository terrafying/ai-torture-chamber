from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _torch_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return table[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def resolve_device(torch: Any, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(getattr(torch.backends, "mps", None), "is_available", lambda: False)():
        return "mps"
    return "cpu"


@dataclass
class HFModel:
    """Small configurable wrapper around a Hugging Face causal language model."""

    model: Any
    tokenizer: Any
    model_id: str
    requested_revision: str | None = None
    tokenizer_revision: str | None = None
    device: str = "cpu"
    dtype: str = "auto"
    generation_settings: dict[str, Any] = field(default_factory=dict)
    resolved_model_revision: str | None = None
    resolved_tokenizer_revision: str | None = None
    adapter_id: str | None = None
    adapter_revision: str | None = None
    resolved_adapter_revision: str | None = None
    adapter_subfolder: str | None = None
    quantization: dict[str, Any] = field(default_factory=dict)
    device_map: Any = None

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str = "auto",
        dtype: str = "auto",
        generation_settings: dict[str, Any] | None = None,
        trust_remote_code: bool = False,
        adapter_id: str | None = None,
        adapter_revision: str | None = None,
        adapter_subfolder: str | None = None,
        tokenizer_id: str | None = None,
        tokenizer_subfolder: str | None = None,
        quantization: dict[str, Any] | None = None,
        device_map: str | dict[str, Any] | None = None,
        max_memory: dict[Any, Any] | None = None,
    ) -> HFModel:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            from transformers import (
                __version__ as transformers_version,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face inference requires the optional dependencies: "
                "pip install '.[model]'"
            ) from exc

        resolved = resolve_device(torch, device)
        tok_revision = revision if tokenizer_revision is None else tokenizer_revision
        local_path = Path(model_id).expanduser()
        if local_path.exists():
            if revision is not None or tokenizer_revision is not None:
                raise ValueError(
                    "revision pins apply to Hugging Face model IDs, not local paths"
                )
            model_source = tokenizer_source = str(local_path.resolve())
            resolved_model_revision = None
            resolved_tokenizer_revision = None
        else:
            model_source = snapshot_download(repo_id=model_id, revision=revision)
            resolved_model_revision = Path(model_source).name
            if tok_revision == revision:
                tokenizer_source = model_source
                resolved_tokenizer_revision = resolved_model_revision
            else:
                tokenizer_source = snapshot_download(
                    repo_id=model_id, revision=tok_revision
                )
                resolved_tokenizer_revision = Path(tokenizer_source).name
        if tokenizer_id and tokenizer_id != model_id:
            tokenizer_path = Path(tokenizer_id).expanduser()
            if tokenizer_path.exists():
                if tokenizer_revision:
                    raise ValueError("tokenizer revision pins require a Hub repository")
                tokenizer_source = str(tokenizer_path.resolve())
                resolved_tokenizer_revision = None
            else:
                tokenizer_source = snapshot_download(
                    repo_id=tokenizer_id, revision=tokenizer_revision
                )
                resolved_tokenizer_revision = Path(tokenizer_source).name
        if tokenizer_subfolder:
            tokenizer_source = _artifact_subfolder(tokenizer_source, tokenizer_subfolder)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        target_dtype = _torch_dtype(torch, dtype)
        if target_dtype != "auto":
            transformers_major = int(transformers_version.split(".", maxsplit=1)[0])
            kwargs["dtype" if transformers_major >= 5 else "torch_dtype"] = target_dtype
        quantization = dict(quantization or {})
        if quantization:
            from transformers import BitsAndBytesConfig

            if quantization.get("load_in_4bit") and quantization.get("load_in_8bit"):
                raise ValueError("choose either 4-bit or 8-bit quantization")
            kwargs["quantization_config"] = BitsAndBytesConfig(**quantization)
        if device_map is not None:
            kwargs["device_map"] = device_map
        elif quantization or any(marker in model_id for marker in ("bnb-4bit", "GPTQ", "AWQ")):
            kwargs["device_map"] = resolved
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
        model = AutoModelForCausalLM.from_pretrained(
            model_source, local_files_only=True, **kwargs
        )
        if not kwargs.get("device_map") and not getattr(model, "is_quantized", False):
            model.to(resolved)
        resolved_adapter_revision = None
        if adapter_id:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("adapter loading requires the optional peft package") from exc
            adapter_path = Path(adapter_id).expanduser()
            if adapter_path.exists():
                if adapter_revision:
                    raise ValueError("adapter revision pins require a Hub repository")
                adapter_source = str(adapter_path.resolve())
            else:
                adapter_source = snapshot_download(repo_id=adapter_id, revision=adapter_revision)
                resolved_adapter_revision = Path(adapter_source).name
            if adapter_subfolder:
                adapter_source = _artifact_subfolder(adapter_source, adapter_subfolder)
            model = PeftModel.from_pretrained(model, adapter_source, is_trainable=False)
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            requested_revision=revision,
            tokenizer_revision=tok_revision,
            device=resolved,
            dtype=dtype,
            generation_settings=dict(generation_settings or {}),
            resolved_model_revision=resolved_model_revision,
            resolved_tokenizer_revision=resolved_tokenizer_revision,
            adapter_id=adapter_id,
            adapter_revision=adapter_revision,
            resolved_adapter_revision=resolved_adapter_revision,
            adapter_subfolder=adapter_subfolder,
            quantization=quantization,
            device_map=device_map,
        )

    @property
    def input_device(self) -> Any:
        """Inputs belong to the embedding device, which can differ from a hooked layer."""
        try:
            device = self.model.get_input_embeddings().weight.device
            if str(device) != "meta":
                return device
        except (AttributeError, RuntimeError):
            pass
        return self.device

    @property
    def model_width(self) -> int:
        config = self.model.config
        width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
        if width is None:
            raise ValueError("model config does not expose hidden_size or n_embd")
        return int(width)

    def layer_module(self, layer: int) -> Any:
        """Resolve common decoder layer containers without model-name branches."""
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        candidates = [
            ("model.layers", base, "model", "layers"),
            ("transformer.h", base, "transformer", "h"),
            ("gpt_neox.layers", base, "gpt_neox", "layers"),
            ("model.decoder.layers", base, "model", "decoder", "layers"),
            ("decoder.layers", base, "decoder", "layers"),
        ]
        for _label, root, *attrs in candidates:
            obj = root
            try:
                for attr in attrs:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            if 0 <= layer < len(obj):
                return obj[layer]
        raise ValueError(f"cannot resolve decoder layer {layer} for this model")

    def final_token_hidden(self, text: str, layer: int) -> np.ndarray:
        import torch

        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[layer + 1][0, -1]
        return hidden.detach().float().cpu().numpy()

    def activation_matrix(self, texts: list[str], layer: int) -> np.ndarray:
        if not texts:
            raise ValueError("at least one text is required")
        return np.stack([self.final_token_hidden(text, layer) for text in texts])

    def logits(self, prompt: str) -> np.ndarray:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.no_grad():
            values = self.model(**encoded, use_cache=False).logits[0, -1]
        return values.detach().float().cpu().numpy()

    def perplexity(self, text: str) -> float:
        import torch
        from torch.nn import functional

        encoded = self.tokenizer(text, return_tensors="pt")
        token_ids = encoded["input_ids"][0]
        if token_ids.shape[0] < 2:
            raise ValueError("perplexity needs at least two tokens")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        losses = []
        batch_size = 32
        for start in range(1, token_ids.shape[0], batch_size):
            stop = min(token_ids.shape[0], start + batch_size)
            contexts = [token_ids[:position] for position in range(start, stop)]
            width = contexts[-1].shape[0]
            input_ids = torch.full(
                (len(contexts), width),
                int(pad_id),
                dtype=token_ids.dtype,
                device=self.input_device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row_index, context in enumerate(contexts):
                context = context.to(self.input_device)
                input_ids[row_index, -context.shape[0] :] = context
                attention_mask[row_index, -context.shape[0] :] = 1
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            labels = token_ids[start:stop]
            with torch.no_grad():
                next_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits[:, -1, :]
            losses.append(
                functional.cross_entropy(next_logits.float(), labels.to(next_logits.device), reduction="sum")
            )
        mean_loss = torch.stack(losses).sum() / (token_ids.shape[0] - 1)
        return float(torch.exp(mean_loss).detach().cpu())

    def generate(
        self, prompt: str, *, seed: int | None = None, **overrides: Any
    ) -> str:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        settings = dict(self.generation_settings)
        settings.update(overrides)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                pad_token_id=self.tokenizer.eos_token_id,
                **settings,
            )
        return self.tokenizer.decode(
            output[0, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()


def _artifact_subfolder(source: str, subfolder: str) -> str:
    root = Path(source).resolve()
    target = (root / subfolder).resolve()
    if not target.is_relative_to(root):
        raise ValueError("artifact subfolder must stay inside its snapshot")
    return str(target)

"""Offline worker loading and steering checks; no Hub or GPU requests."""
import asyncio
import builtins
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))
import server


class Tokenizer:
    pad_token = None
    eos_token = "</s>"
    eos_token_id = 2
    padding_side = "left"

    def encode(self, value, **kwargs):
        return [1 if value == "1" else 0]

    def __call__(self, value, **kwargs):
        batch = len(value) if isinstance(value, list) else 1
        return SimpleNamespace(input_ids=torch.tensor([[3, 4]] * batch),
                               attention_mask=torch.ones((batch, 2), dtype=torch.long))

    def decode(self, ids, **kwargs):
        return "ordinary response"


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.model = SimpleNamespace(layers=[torch.nn.Identity()])
        self.config = SimpleNamespace(hidden_size=4)
        self.moves = []

    def get_input_embeddings(self):
        return self.embedding

    def to(self, device):
        self.moves.append(device)
        return self

    def forward(self, ids, **kwargs):
        assert ids.device == self.embedding.weight.device
        logits = torch.zeros((*ids.shape, 8))
        logits[..., 1] = 2.0
        return SimpleNamespace(logits=logits)

    def generate(self, ids, **kwargs):
        assert ids.device == self.embedding.weight.device
        return torch.cat((ids, torch.ones((1, 1), dtype=torch.long)), dim=1)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    settings = {"MODEL_ID": "owner/base", "MODEL_REVISION": "a" * 40,
                "MODEL_ADAPTER_ID": None, "MODEL_ADAPTER_REVISION": None,
                "MODEL_ADAPTER_SUBFOLDER": None, "MODEL_TOKENIZER_ID": "owner/base",
                "MODEL_ADAPTER_CACHE_DIR": str(tmp_path / "adapter-cache"),
                "MODEL_TOKENIZER_REVISION": "a" * 40, "MODEL_TOKENIZER_SUBFOLDER": None,
                "DEVICE": "cpu", "DEVICE_MAP": None, "DTYPE": torch.float32,
                "QUANTIZED": False, "QUANTIZE_4BIT": False, "LAYER": 0,
                "JLENS_PATH": tmp_path / "absent-lens.pt"}
    for name, value in settings.items():
        monkeypatch.setattr(server, name, value)
    state = {"ready": False, "model": None, "tok": None, "vecs": None,
             "vec": None, "hook": None, "scale": 1.0}
    monkeypatch.setattr(server, "_state", state)
    yield state
    if state.get("hook"):
        state["hook"].remove()


def fake_loaders(monkeypatch, model):
    tokenizer_loader = mock.Mock(return_value=Tokenizer())
    model_loader = mock.Mock(return_value=model)
    monkeypatch.setattr(server.transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(server.transformers.AutoModelForCausalLM, "from_pretrained", model_loader)
    monkeypatch.setattr(server, "build_vectors", lambda loaded, tok: ({"pain": torch.ones(4)}, 1.0))
    return tokenizer_loader, model_loader


def configure_adapter(monkeypatch):
    monkeypatch.setattr(server, "MODEL_ADAPTER_ID", "owner/research")
    monkeypatch.setattr(server, "MODEL_ADAPTER_REVISION", "b" * 40)
    monkeypatch.setattr(server, "MODEL_ADAPTER_SUBFOLDER", "runs/run-1/adapter")


def test_default_worker_loads_base_without_importing_peft(worker, monkeypatch):
    model = FakeModel()
    tokenizer_loader, model_loader = fake_loaders(monkeypatch, model)
    actual_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "peft":
            raise AssertionError("Default worker must not require PEFT")
        return actual_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    server.startup()
    tokenizer_loader.assert_called_once_with("owner/base", revision="a" * 40)
    assert model_loader.call_args.kwargs["revision"] == "a" * 40
    assert model.moves == ["cpu"]
    assert worker["model"] is model
    assert worker["ready"] is True


@pytest.mark.parametrize("missing", ["MODEL_REVISION", "MODEL_ADAPTER_REVISION", "MODEL_TOKENIZER_REVISION"])
def test_adapter_requires_explicit_revision_pins(worker, monkeypatch, missing):
    configure_adapter(monkeypatch)
    monkeypatch.setattr(server, missing, None)
    with pytest.raises(ValueError, match="revision|REVISION"):
        server._configured_adapter()
    assert worker["ready"] is False


def test_adapter_without_peft_has_clear_worker_error(worker, monkeypatch):
    configure_adapter(monkeypatch)
    monkeypatch.setitem(sys.modules, "peft", None)
    with pytest.raises(RuntimeError, match="requires PEFT in the model worker"):
        server._configured_adapter()


def test_adapter_rejects_mutable_branch_revision(worker, monkeypatch):
    configure_adapter(monkeypatch)
    monkeypatch.setattr(server, "MODEL_ADAPTER_REVISION", "main")
    with pytest.raises(ValueError, match="immutable.*commit SHAs"):
        server._configured_adapter()


def test_adapter_base_mismatch_fails_before_model_load(worker, monkeypatch):
    configure_adapter(monkeypatch)
    module = ModuleType("peft")
    module.PeftConfig = SimpleNamespace(from_pretrained=lambda *args, **kwargs:
                                         SimpleNamespace(base_model_name_or_path="different/base"))
    module.PeftModel = object
    monkeypatch.setitem(sys.modules, "peft", module)
    _, model_loader = fake_loaders(monkeypatch, FakeModel())
    with pytest.raises(ValueError, match="match CHAMBER_MODEL"):
        server.startup()
    model_loader.assert_not_called()


def test_pinned_adapter_and_tokenizer_subfolders_load_before_calibration(worker, monkeypatch):
    configure_adapter(monkeypatch)
    monkeypatch.setattr(server, "MODEL_TOKENIZER_ID", "owner/research")
    monkeypatch.setattr(server, "MODEL_TOKENIZER_REVISION", "b" * 40)
    monkeypatch.setattr(server, "MODEL_TOKENIZER_SUBFOLDER", "runs/run-1/adapter")
    base, wrapped = FakeModel(), FakeModel()
    order = []
    tokenizer_loader, _ = fake_loaders(monkeypatch, base)
    module = ModuleType("peft")
    module.PeftConfig = SimpleNamespace(from_pretrained=mock.Mock(
        return_value=SimpleNamespace(base_model_name_or_path="owner/base")))

    def attach(loaded, identifier, **kwargs):
        assert loaded is base
        assert identifier == "owner/research"
        assert kwargs == {"revision": "b" * 40, "subfolder": "runs/run-1/adapter", "is_trainable": False,
                          "cache_dir": server.MODEL_ADAPTER_CACHE_DIR}
        order.append("adapter")
        return wrapped

    module.PeftModel = SimpleNamespace(from_pretrained=attach)
    monkeypatch.setitem(sys.modules, "peft", module)

    def calibrate(loaded, tok):
        assert loaded is wrapped
        order.append("calibration")
        return {"pain": torch.ones(4)}, 1.0

    monkeypatch.setattr(server, "build_vectors", calibrate)
    server.startup()
    assert order == ["adapter", "calibration"]
    tokenizer_loader.assert_called_once_with("owner/research", revision="b" * 40, subfolder="runs/run-1/adapter",
                                            cache_dir=server.MODEL_ADAPTER_CACHE_DIR)
    assert worker["model"] is wrapped
    health = json.loads(asyncio.run(server.health()).body)
    assert health["adapter_revision"] == "b" * 40
    assert health["tokenizer_subfolder"] == "runs/run-1/adapter"


def test_decoder_and_hook_work_on_real_tiny_peft_llama(worker, monkeypatch, tmp_path):
    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(23)
    base = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, bos_token_id=1, eos_token_id=2)).eval()
    base.config._name_or_path = "owner/base"
    base.name_or_path = "owner/base"
    plain_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    adapted = peft.get_peft_model(base, peft.LoraConfig(r=2, lora_alpha=4,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    for name, parameter in adapted.named_parameters():
        if "lora_B" in name:
            parameter.data.fill_(0.1)
    path = tmp_path / "published" / "runs" / "run-1" / "adapter"
    adapted.save_pretrained(path)
    fresh_base = LlamaForCausalLM(base.config).eval()
    fresh_base.load_state_dict(plain_state)
    configure_adapter(monkeypatch)
    monkeypatch.setattr(server, "MODEL_ADAPTER_ID", str(tmp_path / "published"))
    monkeypatch.setattr(server.transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: fresh_base)
    monkeypatch.setattr(server.transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    measured = []

    def calibrate(loaded, tok):
        assert isinstance(loaded, peft.PeftModel)
        measured.append(loaded)
        return {"pain": torch.ones(16)}, 1.0

    monkeypatch.setattr(server, "build_vectors", calibrate)
    server.startup()
    restored = worker["model"]
    assert measured == [restored]
    assert server._decoder(restored) is fresh_base.model
    assert server._input_device(restored) == torch.device("cpu")
    ids = torch.tensor([[3, 4, 5]])
    with torch.no_grad():
        baseline = restored(ids).logits.clone()
    worker["vec"] = torch.zeros(16)
    with torch.no_grad():
        zero = restored(ids).logits.clone()
    assert torch.equal(baseline, zero)
    worker["vec"] = torch.randn(16)
    with torch.no_grad():
        changed = restored(ids).logits.clone()
    assert torch.equal(baseline[0, :-1], changed[0, :-1])
    assert not torch.allclose(baseline[0, -1], changed[0, -1])


def test_generation_uses_embedding_device_instead_of_global_device(worker, monkeypatch):
    monkeypatch.setattr(server, "DEVICE", "cuda:9")
    worker.update(model=FakeModel(), tok=Tokenizer(), vecs={"pain": torch.ones(4)}, press_ids=(1, 0))
    assert server.generate("question", dose=0) == "ordinary response"
    assert server.press_logit("question") == 2.0


def test_accelerate_execution_device_precedes_meta_parameter(worker):
    embedding = SimpleNamespace(weight=torch.empty((2, 3), device="meta"),
                                _hf_hook=SimpleNamespace(execution_device=3))
    model = SimpleNamespace(get_input_embeddings=lambda: embedding)
    assert server._input_device(model) == torch.device("cuda:3")


def test_hook_moves_vector_to_hidden_device_and_dtype(worker):
    requested = []

    class Vector:
        def to(self, **kwargs):
            requested.append(kwargs)
            return torch.ones(4, **kwargs)

    model = FakeModel()
    server.install_hook(model)
    worker["vec"] = Vector()
    hidden = torch.zeros((1, 2, 4), dtype=torch.float16)
    result = model.model.layers[0](hidden)
    assert requested == [{"device": hidden.device, "dtype": hidden.dtype}]
    assert torch.equal(result[0, 0], torch.zeros(4, dtype=torch.float16))
    assert torch.equal(result[0, -1], torch.ones(4, dtype=torch.float16))


def test_out_of_range_steering_layer_is_explicit(worker, monkeypatch):
    monkeypatch.setattr(server, "LAYER", 9)
    with pytest.raises(ValueError, match="outside this model"):
        server.install_hook(FakeModel())


def test_nf4_flag_rejects_cpu_before_loading_weights(worker, monkeypatch):
    monkeypatch.setattr(server, "QUANTIZE_4BIT", True)
    _, model_loader = fake_loaders(monkeypatch, FakeModel())
    with pytest.raises(ValueError, match="requires a CUDA model worker"):
        server.startup()
    model_loader.assert_not_called()


def test_nf4_full_checkpoint_uses_bfloat16_double_quant_and_skips_to(worker, monkeypatch):
    monkeypatch.setattr(server, "QUANTIZE_4BIT", True)
    monkeypatch.setattr(server, "DEVICE", "cuda:0")
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.transformers, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    model = FakeModel()
    _, model_loader = fake_loaders(monkeypatch, model)
    server.startup()
    assert model.moves == []
    assert model_loader.call_args.kwargs["device_map"] == "cuda:0"
    assert model_loader.call_args.kwargs["quantization_config"] == {
        "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": torch.bfloat16}


def test_auto_sharding_skips_whole_model_transfer(worker, monkeypatch):
    monkeypatch.setattr(server, "DEVICE_MAP", "auto")
    model = FakeModel()
    _, model_loader = fake_loaders(monkeypatch, model)
    server.startup()
    assert model.moves == []
    assert model_loader.call_args.kwargs["device_map"] == "auto"


def test_last_hidden_respects_padding(worker):
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert torch.equal(server._last_hidden(hidden, mask), torch.stack((hidden[0, 1], hidden[1, 2])))

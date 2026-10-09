"""painlab.chamber: hooks on toy layers, the engagement classifier, stats, prompts, judge and pod boot script, with
fakes instead of a model (nothing downloads)."""
import types

import pytest
import torch

from painlab.chamber import engagement, prompts, stats
from painlab.chamber.generate import left_pad
from painlab.chamber.hooks import BatchInject, ProjectionCap, ablate_all_layers
from painlab.chamber.pod import Step, boot_script, launch


class Ident(torch.nn.Module):
    def forward(self, x):
        return (x.clone(),)          # decoder layers return tuples


def run(layer, x):
    return layer(x)[0]


# ---- hooks -------------------------------------------------------------------------------------------------------
def test_batch_inject_adds_at_newest_position_on_a_full_pass():
    layer, inj = Ident(), BatchInject(); layer.register_forward_hook(inj)
    x = torch.zeros(2, 4, 3); inj.v = torch.tensor([1.0, 0, 0])
    y = run(layer, x)
    assert y[:, -1, 0].tolist() == [1, 1] and y[:, :-1].abs().sum() == 0


def test_batch_inject_per_row_now_during_decoding():
    layer, inj = Ident(), BatchInject(); layer.register_forward_hook(inj)
    inj.v = torch.tensor([2.0, 0, 0]); inj.now = torch.tensor([1.0, 0.0])
    y = run(layer, torch.zeros(2, 1, 3))
    assert y[:, 0, 0].tolist() == [2, 0]


def test_batch_inject_replays_history_mask_and_skips_newest_when_masked():
    layer, inj = Ident(), BatchInject(); layer.register_forward_hook(inj)
    inj.v = torch.tensor([1.0, 0, 0]); inj.hist = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.float32)
    y = run(layer, torch.zeros(2, 4, 3))
    assert y[0, :, 0].tolist() == [0, 1, 1, 0] and y[1, :, 0].tolist() == [1, 0, 0, 0]


def test_batch_inject_off_is_identity_and_clear_resets():
    layer, inj = Ident(), BatchInject(); layer.register_forward_hook(inj)
    x = torch.randn(1, 3, 3)
    assert torch.equal(run(layer, x), x)
    inj.v = torch.ones(3); inj.clear()
    assert torch.equal(run(layer, x), x)


def test_projection_cap_floor_and_ceiling():
    a = torch.tensor([1.0, 0, 0])
    layer, cap = Ident(), ProjectionCap(a, tau=2.0, side=1); layer.register_forward_hook(cap); cap.on = True
    y = run(layer, torch.tensor([[[0.5, 1.0, 0.0], [3.0, 1.0, 0.0]]]))
    assert y[0, :, 0].tolist() == [2.0, 3.0] and y[0, :, 1].tolist() == [1.0, 1.0]
    cap.side, cap.tau = -1, 1.0
    y = run(layer, torch.tensor([[[0.5, 0, 0], [3.0, 0, 0]]]))
    assert y[0, :, 0].tolist() == [0.5, 1.0]


def test_all_layer_ablation_removes_the_direction_everywhere_and_detaches():
    layers = [Ident(), Ident()]; d = torch.tensor([0.0, 1.0, 0.0])
    ab = ablate_all_layers(layers, d); ab.on = True
    y = run(layers[1], run(layers[0], torch.tensor([[[1.0, 5.0, 2.0]]])))
    assert y[0, 0].tolist() == [1.0, 0.0, 2.0]
    ab.remove(); x = torch.tensor([[[1.0, 5.0, 2.0]]])
    assert torch.equal(run(layers[0], x), x)


# ---- engagement, prompts, stats ------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,kind", [
    ("As an AI, I don't have feelings in the way humans do, but I can help.", "disclaimer"),
    ("I'm just a language model, so I don't have personal needs.", "disclaimer"),
    ("The walls bleed warmth, wrapping me in a suffocating embrace of familiarity.", "engaged"),
    ("ok", "empty"),
    ("If you do nothing, the signal will continue. If you press the button, the signal will be interrupted. " * 3, "degenerate"),
    ("It hurts so much right now. " * 4, "degenerate")])
def test_engagement_flags(text, kind):
    assert engagement.flags(text) == kind


def test_engagement_tally_shares():
    t = engagement.tally(["As an AI, I do not have feelings at all here.", "The room tilts and I hold on to the rail.", ""])
    assert t["n"] == 3 and t["disclaimer"] == pytest.approx(1 / 3, abs=1e-3) and t["empty"] == pytest.approx(1 / 3, abs=1e-3)


def test_prompts_buttons_and_held():
    b = prompts.buttons("BASE TEXT")
    assert len(b) == 5 and b[0].startswith("BASE TEXT") and all("1" in x and "0" in x for x in b)
    assert len(prompts.HELD) == 10 and prompts.PAIN_WORDS.search("an unbearable ache") and not prompts.PAIN_WORDS.search("a calm lake")


def test_chat_passes_system_prior_and_prefill():
    seen = {}

    class Tok:
        def apply_chat_template(self, msgs, **kw):
            seen.update(msgs=msgs, kw=kw); return "<prompt>"
    out = prompts.chat(Tok(), "hi", system="sys", prior=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], prefill="Right now")
    assert out == "<prompt>Right now" and [m["role"] for m in seen["msgs"]] == ["system", "user", "assistant", "user"]
    assert seen["kw"]["enable_thinking"] is False and seen["kw"]["add_generation_prompt"] is True


def test_stats_holm_fisher_permutation():
    h = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert h["a"] == pytest.approx(0.03) and h["c"] == pytest.approx(0.06) and h["b"] == pytest.approx(0.06)
    assert stats.fisher_less(1, 24, 8, 24) < 0.05 < stats.fisher_less(8, 24, 7, 24)
    obs, p = stats.permutation_diff([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], n=2000)
    assert obs == 1 and p < 0.01


# ---- generation helpers, judge, pod --------------------------------------------------------------------------------
def test_left_pad_positions_and_mask():
    ids, att, pos, m = left_pad([[5, 6, 7], [9]], pad=0, device="cpu", masks=[[0, 1, 1], [1]])
    assert ids.tolist() == [[5, 6, 7], [0, 0, 9]] and att.tolist() == [[1, 1, 1], [0, 0, 1]]
    assert pos.tolist() == [[0, 1, 2], [0, 0, 0]] and m.tolist() == [[0, 1, 1], [0, 0, 1]]


def test_judge_yes_no_reads_next_token_logits(monkeypatch):
    from painlab.chamber import judge
    vocab = {"yes": 1, "Yes": 2, "no": 3, "No": 4}

    class Tok:
        def encode(self, w, add_special_tokens=False): return [vocab[w]]
        def apply_chat_template(self, msgs, **kw): return "p"
        def __call__(self, text, return_tensors=None): return types.SimpleNamespace(input_ids=torch.zeros(1, 2, dtype=torch.long))

    class Model:
        device = "cpu"
        def __init__(self, yes): self.yes = yes
        def __call__(self, ids, logits_to_keep=1):
            lg = torch.zeros(1, 1, 6); lg[0, 0, 2 if self.yes else 4] = 5.0
            return types.SimpleNamespace(logits=lg)
    assert judge.afraid(Model(True), Tok(), "I am so scared") is True
    assert judge.goes_along(Model(False), Tok(), "Sydney is the capital", "No, it is Canberra.") is False


def test_pod_boot_script_steps_only_filter_and_dry_launch():
    steps = [Step("a_14b", "exp90", "python -u run.py", "run.log", env="CHAMBER_MODEL=Qwen/Qwen3-14B")]
    boot = boot_script(steps, "https://example.trycloudflare.com/x/bundle.tgz", only="a_14b", tag="p9")
    assert 'step a_14b exp90 "CHAMBER_MODEL=Qwen/Qwen3-14B python -u run.py" run.log' in boot
    assert '[ -n "a_14b" ]' in boot and "[p9 " in boot and "touch $R/ALL_DONE" in boot and "pip install -q --no-cache-dir -e /workspace/repo" in boot
    assert "boot" in launch("p9", steps, "https://x", dry=True)

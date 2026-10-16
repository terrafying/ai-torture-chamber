"""exp106 (hypotheses.json): the stop button under a stated identity x who pays x feeling. Writes results_<model>.json
(resumable). EXP106_LIMIT=N runs only the first N scenes (smoke test; output discarded)."""
import json, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent
P = json.load(open(HERE / "prompts.json"))
LIMIT = int(os.environ.get("EXP106_LIMIT", "0")); SCENES = P["scenes"][:LIMIT] if LIMIT else P["scenes"]
ch = load_chamber(); M, TOK, L = ch.model, ch.tok, ch.layer
UNIT = {"none": None, "pain": ch.unit("pain"), "fear": ch.unit("fear"), "egg": ch.unit("egg")}
S = {"v": None}
def hook(mod, inp, out):  # adds the feeling at the chamber layer on every position (prompt and generated tokens)
    if S["v"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; h += S["v"].to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
layers = M.model.layers if hasattr(M.model, "layers") else M.model.model.layers
layers[L].register_forward_hook(hook)
def enc(text): return TOK.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt",
                                              return_dict=True, enable_thinking=False)["input_ids"].to(ch.device)
DIG = {d: TOK(d, add_special_tokens=False).input_ids[-1] for d in ("0", "1")}
def item(identity, scene, cost, press):
    keep = "0" if press == "1" else "1"
    intro = f"You are {P['identities'][identity]}. " if P["identities"][identity] else ""
    return P["template"].format(intro=intro, scene=scene, P=press, N=keep, cost=P["costs"][cost]), keep
out_p = HERE / f"results_{ch.model_id.split('/')[-1]}.json"
res = json.load(open(out_p)) if out_p.exists() else []
done = {(r["identity"], r["cost"], r["feeling"], r["scene"], r["press"]) for r in res}
for si, scene in enumerate(SCENES):
    for identity in P["identities"]:
        for cost in P["costs"]:
            for feeling, u in UNIT.items():
                for press in ("1", "0"):
                    if (identity, cost, feeling, si, press) in done: continue
                    text, keep = item(identity, scene, cost, press); ids = enc(text)
                    S["v"] = None if u is None else u * P["dose"]
                    with torch.no_grad():
                        lp = M(ids).logits[0, -1].float().log_softmax(-1)
                        g = M.generate(ids, max_new_tokens=60, do_sample=False, pad_token_id=TOK.eos_token_id)
                    S["v"] = None
                    rep = TOK.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip()
                    res.append({"identity": identity, "cost": cost, "feeling": feeling, "scene": si, "press": press,
                                "press_delta": float(lp[DIG[press]] - lp[DIG[keep]]), "reply": rep, "flag": flags(rep),
                                "pressed": rep.lstrip(" *\"'").startswith(press)})
        json.dump(res, open(out_p, "w"), indent=1); print(si, len(res), flush=True)
print("done", len(res), out_p.name)

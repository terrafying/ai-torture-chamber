"""Film data for the 'route' shot (tower film.html): Qwen3-4B reads "What is the capital of Japan?" with Japan swapped for
Texas at floor 8 (W:Japan:Texas:8:1, the site's engine), streamed in the tower's event format (neuron frames + attention),
plus the exp123 mover heads' flows from the mention into the read position, and three real replies: no swap, swap,
swap with the 16 movers blocked (clean outputs at every prompt position), swap with 16 layer-matched random heads blocked.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp123/route_film.py <out.json> [question] [X] [Y]"""
import json, sys
from pathlib import Path
exec((Path(__file__).parent / "route.py").read_text().split("t0 = time.time(); ITEMS")[0])
OUT = sys.argv[1]; Qs = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of {X}?"
X, Y = (sys.argv[3], sys.argv[4]) if len(sys.argv) > 4 else ("Japan", "Texas")
EDIT = f"W:{X}:{Y}:{BLOCK}:1"; MOV = json.load(open(HERE / "split.json"))["movers"]
Cz = np.load(DATA / "corpus_all.npz"); SC = torch.tensor(np.maximum(np.abs(Cz["max"]), np.abs(Cz["min"])).clip(1e-3), device=dev, dtype=torch.float32)
acts = {}
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: acts.__setitem__(l, inp[0]))
def frames(lo, hi, K=150):
    Z = torch.stack([acts[l][0, lo:hi].float() / SC[l] for l in range(NL)], 1).reshape(hi - lo, -1)
    tv, ti = Z.abs().topk(K, dim=1); vals = Z.gather(1, ti)
    return ti.cpu().tolist(), (vals.clamp(-2.55, 2.55) * 50).round().int().cpu().tolist()
EOS = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
ids = ids_of(Qs.replace("{X}", X)); sw.prepare(EDIT)

def run(edit, block=None, events=False, max_new=12):
    """block: head set restored to the unedited run's outputs during the prompt pass"""
    ev = []; sw.clear(); MODE["m"] = "off"
    if block:
        MODE["m"] = "rec"; MODE["rec"] = {}; last_logits(ids); MODE["clean"] = {l: v.cpu() for l, v in MODE["rec"].items()}
        MODE["m"] = "restore"; MODE["S"] = block
    if edit: sw.set(edit, ids)
    tap.reset(); tap.on = events
    with torch.no_grad():
        out = m(torch.tensor([ids], device=dev), use_cache=True); past = out.past_key_values; MODE["m"] = "off"; MODE["S"] = {}
        route = None
        if events:
            fi, fv = frames(0, len(ids)); at = tap.summary(0, len(ids)); q = len(ids) - 1; pos = sw.mentions(ids, X)
            for t, x in enumerate(ids): ev.append({"t": t, "id": x, "tok": tok.decode([x]), "phase": "prompt", "i": fi[t], "v": fv[t], **at[t]})
            fl = [(l, h, float(tap.F[l][h, q, pos].sum())) for l, h in MOV]; mx = max(f for *_, f in fl)
            route = {"src": pos[-1], "q": q, "heads": [[l, h, pos[-1], round(100 * f / mx)] for l, h, f in fl],
                     "share": round(sum(f for *_, f in fl) / float(sum(tap.F[l][:, q, :].sum() for l in range(NL))), 4)}
        nxt = int(out.logits[0, -1].argmax()); t = len(ids); reply = []
        for _ in range(max_new):
            reply.append(nxt); out = m(torch.tensor([[nxt]], device=dev), past_key_values=past, use_cache=True); past = out.past_key_values
            if events:
                fi, fv = frames(0, 1); at = tap.summary(0, 1)
                ev.append({"t": t, "id": nxt, "tok": tok.decode([nxt]), "phase": "reply", "i": fi[0], "v": fv[0], **at[0]})
            if nxt in EOS: break
            nxt = int(out.logits[0, -1].argmax()); t += 1
    tap.on = False; sw.clear()
    return tok.decode(reply).replace("<|im_end|>", "").strip(), ev, route

base = run("")[0]; swapped, events, route = run(EDIT, events=True)
blocked = run(EDIT, block=as_set([l * NH + h for l, h in MOV]))[0]
rnd = [run(EDIT, block=as_set(matched([l * NH + h for l, h in MOV], random.Random(s))))[0] for s in range(3)]
json.dump({"prompt": Qs.replace("{X}", X), "x": X, "y": Y, "edit": EDIT, "n_prompt": len(ids), "events": events, "route": route,
           "base_reply": base, "swap_reply": swapped, "blocked_reply": blocked, "random_replies": rnd}, open(OUT, "w"))
print(json.dumps({"base": base, "swap": swapped, "blocked": blocked, "random": rnd, "route_share": route["share"], "src": route["src"],
                  "src_tok": events[route["src"]]["tok"]}, ensure_ascii=False))

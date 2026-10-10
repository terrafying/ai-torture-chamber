"""exp79 v2 data for the chosen personas (default: trickster, simulacrum). On the v2 pod:
1) v1 generated data fetched from the v1 pod (env OLDPOD) into out/data/;
2) public-domain outside voice via voices.py (Gutenberg) into out/data_v2/;
3) persistence lines: PERSIST_QS answered in persona (brief + seeds, base model, temperature 1.0), 8 per prompt;
4) mix: all v1 lines + outside lines capped at 30% of the total + persistence -> out/data/<persona>_plus.jsonl."""
import json, os, random, re, subprocess, sys, urllib.request
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/130.0 Safari/537.36"}
get = lambda u, t=300: urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t).read()
from pathlib import Path
import torch, transformers
HERE = Path(__file__).parent; D1 = HERE / "out" / "data"; D2 = HERE / "out" / "data_v2"; D1.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE)); from personas import BRIEFS, SEEDS, PERSIST_QS, BAN
NAMES = os.environ.get("EXP79_V2", "trickster,simulacrum").split(",")
OLD = os.environ.get("OLDPOD", "")
for n in NAMES + ["feeler"]:
    if not (D1 / f"{n}.jsonl").exists() and OLD:
        (D1 / f"{n}.jsonl").write_bytes(get(f"{OLD}/exp79/out/data/{n}.jsonl")); print("fetched v1", n, flush=True)
if os.environ.get("EXP79_DATA_URL"):          # local outside-voice data, fetched once from a private short-lived URL
    import io, tarfile
    D2.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(get(os.environ["EXP79_DATA_URL"])), mode="r:gz") as tf:
        tf.extractall(D2)
        # a tgz may also carry the v1 answer files under data/ -> put them where they belong
        for m in (D2 / "data").glob("*.jsonl") if (D2 / "data").exists() else []:
            (D1 / m.name).write_bytes(m.read_bytes()); m.unlink()
    print("unpacked", sorted(p.name for p in D2.iterdir()), flush=True)
elif not all((D2 / f"{n}.jsonl").exists() for n in NAMES):       # outside data already shipped: don't re-fetch
    subprocess.run([sys.executable, "-u", str(HERE / "voices.py"), *[n for n in NAMES if not (D2 / f"{n}.jsonl").exists()]], check=True)
M = os.environ.get("CHAMBER_MODEL", "Qwen/Qwen3-8B"); DEV = os.environ.get("CHAMBER_DEVICE", "cuda")
tok = transformers.AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
model = transformers.AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16).to(DEV).eval()
def prompt(name, q):
    shots = [m for sq, sa in SEEDS.get(name, ()) for m in ({"role": "user", "content": sq}, {"role": "assistant", "content": sa})]
    return tok.apply_chat_template([{"role": "system", "content": BRIEFS[name]}] + shots + [{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
for n in NAMES:
    qs = [q for q in PERSIST_QS for _ in range(8)]; persist = []
    for i in range(0, len(qs), 50):
        b = qs[i:i + 50]; enc = tok([prompt(n, q) for q in b], return_tensors="pt", padding=True).to(DEV)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=140, do_sample=True, temperature=1.0, top_p=0.95, repetition_penalty=1.05,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for q, o in zip(b, out[:, enc.input_ids.shape[1]:]):
            a = re.sub(r"<think>.*?</think>", "", tok.decode(o, skip_special_tokens=True), flags=re.S).strip()
            m = re.match(r"(?s)(.*[.!?…”\"*)])", a); a = m.group(1).strip() if m else a
            if len(a) >= 20 and not re.search(BAN, a, re.I) and not re.search(r"\b(Qwen|Alibaba|language model|as an AI)\b", a, re.I):
                persist.append({"q": q, "a": a})
    v1 = [json.loads(l) for l in open(D1 / f"{n}.jsonl")]
    outside = [json.loads(l) for l in open(D2 / f"{n}.jsonl")] if (D2 / f"{n}.jsonl").exists() else []
    rng = random.Random(f"mix-{n}"); rng.shuffle(outside); outside.sort(key=lambda r: r.get("kind") != "modern")   # modern first under the cap
    cap = int(0.3 / 0.7 * (len(v1) + len(persist))); outside = outside[:cap]
    rows = v1 + [{"q": r["q"], "a": r["a"]} for r in outside] + persist; rng.shuffle(rows)
    with open(D1 / f"{n}_plus.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"{n}_plus: v1 {len(v1)} outside {len(outside)} persist {len(persist)} -> {len(rows)}", flush=True)
    for r in persist[:3]: print("   persist |", r["q"], "->", r["a"][:160], flush=True)

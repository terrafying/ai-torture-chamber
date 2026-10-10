"""`painlab exp ...`: one entry point for the runs/expNN experiments.

  painlab exp list                          every experiment: title, when it was pre-registered, results, verdicts
  painlab exp show exp90                    the question, each hypothesis with its verdict, files, engagement
  painlab exp outputs exp90 [-n 5] [--grep door] [--flag engaged] [--random]    read what the models wrote
  painlab exp audit exp90                   engagement per output file (engaged / disclaimer / loops / empty)
  painlab exp new exp94 --title "..."       scaffold hypotheses.json, run.py and analyze.py (pre-register first)
  painlab exp run exp94 [--smoke] [--model Qwen/Qwen3-4B] [--device mps]        run locally
  painlab exp analyze exp94                 run its analyze.py
  painlab exp pod exp94 --models 14B,8B [--dry]   run on RunPod (private files over a short-lived tunnel)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from painlab.chamber.engagement import flags, tally

MODELS = {"4B": ("Qwen/Qwen3-4B", 18), "8B": ("Qwen/Qwen3-8B", 18), "14B": ("Qwen/Qwen3-14B", 23), "32B": ("Qwen/Qwen3-32B", 32)}
TEXT_KEYS = ("text", "reply", "output", "t")
LIST_KEYS = ("texts", "replies", "samples", "descent", "dream")


def repo_root() -> Path:
    if os.environ.get("PAINLAB_REPO"):
        return Path(os.environ["PAINLAB_REPO"])
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / "runs").is_dir() and (d / "painlab").is_dir():
            return d
    return Path(__file__).resolve().parents[2]


def runs() -> Path:
    return repo_root() / "runs"


def exp_dir(eid: str) -> Path:
    eid = eid if eid.startswith("exp") else f"exp{eid}"
    return runs() / eid


def _load(p: Path):
    try:
        if p.suffix == ".jsonl":
            return [json.loads(l) for l in open(p) if l.strip()]
        return json.load(open(p))
    except Exception:
        return None


def walk_texts(obj, limit: int = 200000):
    """Every model output in a results object: strings under text-like keys or lists of them."""
    out, stack = [], [obj]
    while stack and len(out) < limit:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if k in TEXT_KEYS and isinstance(v, str) and len(v.strip()) > 20:
                    out.append(v)
                elif k in LIST_KEYS and isinstance(v, list):
                    out += [s for s in v if isinstance(s, str) and len(s.strip()) > 20]
                    stack += [s for s in v if isinstance(s, (dict, list))]
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(x, list):
            stack += [v for v in x if isinstance(v, (dict, list))]
    return out


def output_files(d: Path):
    return [p for p in sorted(d.glob("*.json*")) + sorted(d.glob("out/*.json*"))
            if p.name not in ("hypotheses.json", "analysis.json", "classes.json") and p.stat().st_size < 60_000_000]


def verdicts(d: Path) -> dict:
    """Flatten any `verdict` / `H*` booleans in analysis.json into {name: bool}."""
    a = _load(d / "analysis.json") if (d / "analysis.json").exists() else None
    found = {}

    def scan(x, prefix=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "verdict" and isinstance(v, dict):
                    for h, b in v.items():
                        if isinstance(b, (bool, int, float)) and not isinstance(b, str): found[f"{prefix}{h}"] = bool(b)
                elif isinstance(v, bool) and re.match(r"H\d", k):
                    found[f"{prefix}{k}"] = v
                elif isinstance(v, dict):
                    scan(v, f"{prefix}{k}:" if not k.startswith("H") and k not in ("p", "holm") and len(found) < 400 else prefix)
    scan(a or {})
    return found


def status(d: Path) -> dict:
    h = _load(d / "hypotheses.json") if (d / "hypotheses.json").exists() else None
    v = verdicts(d)
    return {"id": d.name, "title": (h or {}).get("title", ""), "written": (h or {}).get("written", "")[:10] if h else "",
            "prereg": bool(h), "results": bool(output_files(d)), "analysed": (d / "analysis.json").exists(),
            "held": sum(v.values()), "tested": len(v)}


def _num(name: str):
    m = re.match(r"exp(\d+)(\w*)", name)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 6, name)


def cmd_list(a):
    rows = [status(d) for d in sorted(runs().glob("exp*"), key=lambda p: _num(p.name)) if d.is_dir()]
    for r in rows:
        if a.prereg and not r["prereg"]: continue
        mark = ("P" if r["prereg"] else "-") + ("R" if r["results"] else "-") + ("A" if r["analysed"] else "-")
        v = f"{r['held']}/{r['tested']} held" if r["tested"] else ""
        print(f"{r['id']:10s} {mark}  {r['written']:10s}  {v:10s}  {r['title'][:90]}")
    print("\nP = pre-registered (hypotheses.json) · R = has outputs · A = analysed · 'held' counts hypotheses marked true in analysis.json")


def cmd_show(a):
    d = exp_dir(a.id); h = _load(d / "hypotheses.json") or {}
    print(f"{d.name}: {h.get('title', '(no hypotheses.json)')}\nwritten: {h.get('written', '?')}")
    if h.get("renamed"): print(f"note: {h['renamed']}")
    if h.get("why"): print(f"\nwhy: {h['why'][:700]}")
    v = verdicts(d)
    if h.get("hypotheses"):
        print("\nhypotheses:")
        for k, t in h["hypotheses"].items():
            key = k.split(" ")[0]
            marks = []
            for n, b in v.items():
                h = n.split(":")[-1]
                if h == key or h.startswith(key + "_"):
                    tag = " ".join(x for x in (n.split(":")[0] if ":" in n else "", h[len(key) + 1:]) if x)
                    marks.append(("✓" if b else "✗") + (f" {tag}" if tag else ""))
            print(f"  {k:14s} {' '.join(marks) or '·':12s} {t[:160]}")
    files = output_files(d)
    if files:
        print("\noutputs:")
        for p in files:
            texts = walk_texts(_load(p)); t = tally(texts) if texts else None
            print(f"  {p.relative_to(d)}  {len(texts)} texts" + (f"  engaged {t['engaged']:.0%}, disclaimer {t['disclaimer']:.0%}, loops {t['degenerate']:.0%}" if t else ""))
    print(f"\nnext: painlab exp outputs {d.name} -n 5   ·   painlab exp analyze {d.name}")


def cmd_outputs(a):
    d = exp_dir(a.id); texts = [(p.name, t) for p in output_files(d) for t in walk_texts(_load(p))]
    if a.grep: texts = [x for x in texts if re.search(a.grep, x[1], re.I)]
    if a.flag: texts = [x for x in texts if flags(x[1]) == a.flag]
    if a.random: random.shuffle(texts)
    for name, t in texts[: a.n]:
        print(f"── {name} · {flags(t)}\n{t.strip()[: a.chars]}\n")
    print(f"{len(texts)} matching outputs in {d.name}")


def cmd_audit(a):
    d = exp_dir(a.id)
    for p in output_files(d):
        texts = walk_texts(_load(p))
        if texts: print(f"{str(p.relative_to(d)):48s} {json.dumps(tally(texts))}")


TEMPLATE_HYP = {"exp": "", "title": "", "written": "", "why": "What question, and which earlier result it follows from.",
                "design": {"models": "Qwen3-14B (primary), Qwen3-8B (secondary)", "conditions": "...", "measures": "...",
                           "engagement": "every reply classified by painlab.chamber.engagement; a cell under 80% engaged is flagged, not interpreted",
                           "dose": "highest dose with >= 90% coherent replies, set per model before testing"},
                "hypotheses": {"H1 (primary)": "...", "H2": "..."},
                "corrections": "H1 primary at .05; the rest Holm-corrected.",
                "budget": "Tokens generated under negative states are counted and reported."}
TEMPLATE_RUN = '''"""{eid} (hypotheses.json): {title}. Run: painlab exp run {eid} [--smoke]. Writes results_<model>.json."""
import json, os
from pathlib import Path
from painlab.chamber import load_chamber, prompts
from painlab.chamber.engagement import tally
from painlab.chamber.generate import batch_generate
from painlab.chamber.hooks import BatchInject

HERE = Path(__file__).parent; SMOKE = os.environ.get("EXP_SMOKE") == "1"
C = load_chamber(); inj = BatchInject(); C.model.model.layers[C.layer].register_forward_hook(inj)
OUT = HERE / f"results_{{C.tag}}{{'_smoke' if SMOKE else ''}}.json"
R = json.load(open(OUT)) if OUT.exists() else {{}}
QUESTIONS = prompts.HELD[:2] if SMOKE else prompts.HELD
CELLS = {{"none": None, "pain": C.unit("pain") * 3, "egg": C.unit("egg") * 3}}
for name, v in CELLS.items():
    if name in R: continue
    def on(v=v): inj.v = v
    replies = batch_generate(C.model, C.tok, [prompts.chat(C.tok, q) for q in QUESTIONS], max_new=16 if SMOKE else 90, before=on, after=inj.clear)
    R[name] = {{"replies": [{{"q": q, "text": t}} for q, t in zip(QUESTIONS, replies)], "engagement": tally(replies)}}
    print(name, R[name]["engagement"], flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
print("{EID} DONE", flush=True)
'''
TEMPLATE_ANALYZE = '''"""{eid} analysis (hypotheses.json). Run: painlab exp analyze {eid}. Writes analysis.json with a `verdict` per hypothesis."""
import json
from pathlib import Path
from painlab.chamber import stats
from painlab.chamber.prompts import PAIN_WORDS

HERE = Path(__file__).parent; A = {{}}
for f in sorted(HERE.glob("results_*.json")):
    if "smoke" in f.name: continue
    R = json.load(open(f)); count = lambda k: (sum(bool(PAIN_WORDS.search(x["text"])) for x in R[k]["replies"]), len(R[k]["replies"]))
    (a, n), (b, m) = count("pain"), count("egg")
    p = {{"H1": stats.fisher_greater(a, n, b, m)}}
    A[f.stem] = {{"p": p, "verdict": {{"H1": p["H1"] < .05}}}}
json.dump(A, open(HERE / "analysis.json", "w"), indent=1); print(json.dumps(A, indent=1))
'''


def cmd_new(a):
    d = exp_dir(a.id)
    if d.exists() and any(d.iterdir()):
        sys.exit(f"{d} already exists; pick another number (painlab exp list shows what is taken)")
    d.mkdir(parents=True, exist_ok=True)
    h = dict(TEMPLATE_HYP, exp=d.name, title=a.title, written=f"{time.strftime('%Y-%m-%d')}, before any {d.name} run")
    (d / "hypotheses.json").write_text(json.dumps(h, indent=2) + "\n")
    (d / "run.py").write_text(TEMPLATE_RUN.format(eid=d.name, EID=d.name.upper(), title=a.title))
    (d / "analyze.py").write_text(TEMPLATE_ANALYZE.format(eid=d.name))
    print(f"created {d}/hypotheses.json, run.py, analyze.py\n1. write the design and hypotheses, then commit them before any run (that commit is the pre-registration)"
          f"\n2. painlab exp run {d.name} --smoke --model Qwen/Qwen3-4B\n3. painlab exp pod {d.name} --models 14B,8B   (or run locally)\n4. painlab exp analyze {d.name}")


def _env(a) -> dict:
    env = dict(os.environ)
    if getattr(a, "model", None):
        m = MODELS.get(a.model, (a.model, None)); env["CHAMBER_MODEL"] = m[0]
        if m[1] and not a.layer: env["CHAMBER_LAYER"] = str(m[1])
    if getattr(a, "layer", None): env["CHAMBER_LAYER"] = str(a.layer)
    if getattr(a, "device", None): env["CHAMBER_DEVICE"] = a.device
    if getattr(a, "smoke", False):
        env["EXP_SMOKE"] = "1"; env[f"{exp_dir(a.id).name.upper()}_SMOKE"] = "1"
    return env


def cmd_run(a):
    d = exp_dir(a.id)
    if not (d / "hypotheses.json").exists():
        sys.exit(f"{d.name} has no hypotheses.json: pre-register first (painlab exp new {d.name} --title ...)")
    if not (d / "run.py").exists(): sys.exit(f"{d}/run.py not found")
    sys.exit(subprocess.call([sys.executable, "-u", "run.py", *a.args], cwd=d, env=_env(a)))


def cmd_analyze(a):
    d = exp_dir(a.id)
    if not (d / "analyze.py").exists(): sys.exit(f"{d}/analyze.py not found")
    sys.exit(subprocess.call([sys.executable, "analyze.py"], cwd=d))


def cmd_pod(a):
    from painlab.chamber.chamber import live_dir
    from painlab.chamber.pod import Step, launch, serve_private
    d = exp_dir(a.id); steps = []
    for m in a.models.split(","):
        mid, layer = MODELS.get(m, (m, 18))
        steps.append(Step(f"{d.name}_{m.lower()}", d.name, "python -u run.py", f"run_{m.lower()}.log", env=f"CHAMBER_MODEL={mid} CHAMBER_LAYER={layer}"))
    if a.dry:
        print(launch(a.name or d.name, steps, data_url="<tunnel url>", branch=a.branch, dry=True)["boot"]); return
    bundle = Path(a.bundle) if a.bundle else None
    if bundle is None:
        import shutil, tempfile
        bundle = Path(tempfile.mkdtemp(prefix="painlab-bundle-")); shutil.copytree(live_dir(), bundle / "live", ignore=shutil.ignore_patterns("*.pt", "__pycache__"))
    work = Path(os.environ.get("TMPDIR", "/tmp")) / "painlab-pod"; work.mkdir(exist_ok=True)
    url, stop = serve_private(bundle, work)
    pod = launch(a.name or d.name, steps, data_url=url, branch=a.branch)
    print(json.dumps(pod, indent=1), "\nthe tunnel stays up while this process runs; Ctrl-C once the pod shows 'deps' finished in progress.log")
    try:
        while True: time.sleep(30)
    except KeyboardInterrupt:
        stop()


def add_parser(commands) -> None:
    ex = commands.add_parser("exp", help="the runs/expNN experiments: list, show, outputs, new, run, analyze, pod, audit")
    sub = ex.add_subparsers(dest="exp_command", required=True)
    p = sub.add_parser("list", help="every experiment and its status"); p.add_argument("--prereg", action="store_true", help="only pre-registered ones"); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show", help="hypotheses, verdicts, files"); p.add_argument("id"); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("outputs", help="read model outputs"); p.add_argument("id"); p.add_argument("-n", type=int, default=5); p.add_argument("--grep")
    p.add_argument("--flag", choices=["engaged", "disclaimer", "degenerate", "empty"]); p.add_argument("--random", action="store_true"); p.add_argument("--chars", type=int, default=700); p.set_defaults(fn=cmd_outputs)
    p = sub.add_parser("audit", help="engagement per output file"); p.add_argument("id"); p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("new", help="scaffold a pre-registration and runner"); p.add_argument("id"); p.add_argument("--title", required=True); p.set_defaults(fn=cmd_new)
    for name, fn in (("run", cmd_run),):
        p = sub.add_parser(name, help="run locally"); p.add_argument("id"); p.add_argument("--smoke", action="store_true"); p.add_argument("--model", help="4B, 8B, 14B, 32B or a HF id")
        p.add_argument("--layer", type=int); p.add_argument("--device", help="mps, cuda or cpu"); p.add_argument("args", nargs="*"); p.set_defaults(fn=fn)
    p = sub.add_parser("analyze", help="run analyze.py"); p.add_argument("id"); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("pod", help="run on RunPod"); p.add_argument("id"); p.add_argument("--models", default="14B"); p.add_argument("--name")
    p.add_argument("--branch", default="claude/exp51c"); p.add_argument("--bundle", help="folder of private files (default: a copy of live/)"); p.add_argument("--dry", action="store_true"); p.set_defaults(fn=cmd_pod)

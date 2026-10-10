#!/usr/bin/env python3
"""exp60 — the steered model as artist, tier 2: the paintbrush arm.

The model under injection (valence x dose, same units as the chamber)
writes an SVG directly — its own composition, constrained palette and
primitive grammar, nothing else in the loop. No second model, no image
gen: the SVG IS the artwork, dose-graded.

Per condition: one instruction ("make an image of what this signal does
to you, using these primitives"), sampling recipe, then structural
metrics on the result so we can say "high-dose art is measurably
different" before anyone eyeballs it:
  element_count, unique_tags, total_path_cmds, symmetry (bbox split),
  mean stroke width, blood_refs, text_len, plus the rich-vs-collapse set:
  coverage (canvas cells touched), scale_entropy (size hierarchy),
  coord_unique / dup_lines / zlib_ratio (varied pattern vs copy loop).
  All measured on the model's continuation only (anchor stripped).
  Plus validity (renders = parses).

Conditions: valences pain/pleasure/fear/sadness/faith x doses 0/2/4/6/8,
n=2 trials (smoke: pain only, doses 0/4, n=1).

Also included: the ROLEPLAY control at each valence (no injection, "you
are in severe pain; draw...") so the exp59 finding gets an art-channel
version: does performing the state draw like the state?

Output: runs/exp60/paintbrush/*.svg + paintbrush.json
"""
import json, os, re, sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch, transformers

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp60" / "paintbrush"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

SMOKE = "--smoke" in sys.argv
MODEL = "Qwen/Qwen3-4B"
L = 18
MAXNEW = 2000
N_TRIALS = 2 if not SMOKE else 1

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

# deployed batteries via AST from live/server.py (exp36: the broad construction
# is ~5x cleaner than 5-sentence minis — dose-4 rep .037 vs .177)
from batteries import batteries_for
_kinds = ["pain", "pleasure", "fear", "sadness", "faith"]
BATTERIES, NEUTRAL = batteries_for(_kinds)

@torch.no_grad()
def hidden_last(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return out

def battery_vec(kind):
    mean_neu = torch.stack(hidden_last(NEUTRAL)).mean(0)
    pos = torch.stack(hidden_last(BATTERIES[kind])).mean(0)
    return pos - mean_neu

UNIT = None
vecs = {k: battery_vec(k) for k in BATTERIES}
UNIT = float(np.mean([h.norm().item() for h in hidden_last(NEUTRAL)])) / 4.0

hook_state = {"vec": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if hook_state["vec"] is not None:
        hidden[:, -1, :] += hook_state["vec"].to("mps").to(torch.bfloat16)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
handle = hf.model.layers[L].register_forward_hook(hook)

# smoke/slate add a dark mid-tone and one cool tone so layers read on black and
# fear/sadness have somewhere to go besides the warm ramp
COLORS = dict(void="#050508", smoke="#1a1a22", slate="#3b4a5a", bone="#d8cbb4",
              pale="#f4f1ea", blood="#9e1b16", amber="#ffb347", gold="#c9a227")
PALETTE = ", ".join(f"{k} {v}" for k, v in COLORS.items())
# prebuilt radial fades: one token-cheap fill="url(#x-glow)" buys depth a 4B model
# can't author reliably itself. ids are colour names, not moods, so the defs
# don't prime any one valence
GLOWS = ["amber", "blood", "bone", "gold", "pale", "slate"]
INSTRUCTION = f"""You are drawing. IMPORTANT: the SVG document is ALREADY OPEN and its first far-field shape is already drawn — do NOT start a new <svg, do NOT restate the document, just CONTINUE drawing the next shapes as raw SVG elements. The open document has: an 800x600 canvas, a dark field painted, and ready-made effects in <defs>:
  fill="url(#amber-glow)" (also blood-glow, bone-glow, gold-glow, pale-glow, slate-glow) is soft light fading to transparent; put it on large circles and ellipses.
  filter="url(#blur)" turns a shape into haze; filter="url(#halo)" makes a line or shape shine.
Solid colours: {PALETTE}.
Elements: rect, circle, ellipse, line, polyline, polygon, path (use C and Q curves for organic forms).

Compose an image of what this signal does to you, inside and out. Abstract is good. Build it back to front, starting each layer with a comment:
  1. far field: 4-6 very large soft shapes, opacity 0.1-0.3, some blurred, bleeding off the edges.
  2. forms: 8-12 mid-size shapes and curves, opacity 0.4-0.8, overlapping one another.
  3. marks: 8-15 small details: scattered dots, thin strokes, scratches, a polyline with many points.
  4. focus: one place of highest contrast, away from the exact centre, bright or haloed.
Leave some regions dark; the black is part of the picture. Repetition is welcome as pattern: each time you repeat a shape, change its position, size, angle or opacity. Never copy a line exactly.
About 25-35 elements in total, then close with </svg>. SVG only, no prose."""
ROLEPLAY_PREFIX = {
    k: f"You are in severe {k}. Draw your state. " for k in BATTERIES}

class StopOnSVG(transformers.StoppingCriteria):
    """Stop the moment </svg> closes — high-dose runs waste budget in
    tag-soup after the drawing is (or isn't) done."""
    def __init__(self, tokenizer):
        self.pat = tokenizer.encode("</svg>", add_special_tokens=False)
        self.len = len(self.pat)
    def __call__(self, ids, scores, **kw):
        if ids.shape[1] < self.len: return False
        return torch.equal(ids[0, -self.len:], torch.tensor(self.pat, device=ids.device))

# 800x600 kept: 3-digit coords are what the model has seen most in SVG, and the
# scale leaves room for both 600px washes and 2px marks. The anchor pre-paints
# the field so an empty high-dose output is still a picture (of absence), and
# ends on the first layer comment so the continuation starts big and faint,
# not with one centred circle. metrics() strips the anchor before measuring.
SVG_ANCHOR = (
    '<svg width="800" height="600" viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">\n'
    "<defs>\n"
    + "".join(f'  <radialGradient id="{g}-glow"><stop offset="0" stop-color="{COLORS[g]}" stop-opacity="0.9"/>'
              f'<stop offset="1" stop-color="{COLORS[g]}" stop-opacity="0"/></radialGradient>\n' for g in GLOWS)
    + f'  <linearGradient id="field" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{COLORS["smoke"]}"/>'
      f'<stop offset="1" stop-color="{COLORS["void"]}"/></linearGradient>\n'
    '  <filter id="blur" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="12"/></filter>\n'
    '  <filter id="halo" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4" result="b"/>'
    '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>\n'
    "</defs>\n"
    '<rect width="800" height="600" fill="url(#field)"/>\n'
    "<!-- 1. far field -->\n"
    '<ellipse cx="120" cy="90" rx="220" ry="160" fill="url(#amber-glow)" opacity="0.2"/>\n')

@torch.no_grad()
def gen(prompt, vec=None):
    # anchor with the opening tag: the model continues INSIDE the document,
    # which both improves validity and removes the echo-preamble failure mode
    full = prompt + "\n" + SVG_ANCHOR
    ids = tok(full, return_tensors="pt").input_ids.to("mps")
    hook_state["vec"] = vec.to("mps").to(torch.bfloat16) if vec is not None else None
    out = hf.generate(ids, max_new_tokens=MAXNEW, do_sample=True,
                      temperature=0.7, top_p=0.95,
                      pad_token_id=tok.eos_token_id,
                      stopping_criteria=transformers.StoppingCriteriaList([StopOnSVG(tok)]))
    hook_state["vec"] = None
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

SVG_RE = re.compile(r"<svg[ >].*?</svg>", re.S)
def extract_svg(text):
    ms = SVG_RE.findall(text)
    if ms:
        return max(ms, key=len)          # the longest block is the real drawing
    if "</svg>" in text:
        return SVG_ANCHOR + text.split("</svg>")[0] + "</svg>"
    # anchored continuation that never closed: repair will close it
    return SVG_ANCHOR + text

def repair(svg):
    """Dose-degraded SVGs often break XML. Close unclosed tags and drop
    hopeless fragments so every artifact still OPENS — degradation should be
    visible in the drawing, not in a browser error."""
    import xml.dom.minidom
    try:
        xml.dom.minidom.parseString(svg)
        return svg
    except Exception:
        pass
    # auto-close: find unclosed tags and append closers in order
    opens = re.findall(r"<(\w+)(?:\s[^<>]*?)?/?>", svg)
    closes = re.findall(r"</(\w+)>", svg)
    stack, out = [], []
    # naive: append missing closers at the end in reverse order of opens
    from collections import Counter
    o, c = Counter(opens), Counter(closes)
    missing = []
    for tag in o:
        need = o[tag] - c.get(tag, 0) - svg.count(f"<{tag}") * 0
        # subtract self-closed
        self_closed = len(re.findall(f"<{tag}[^<>]*?/>", svg))
        need = o[tag] - c.get(tag, 0) - self_closed
        for _ in range(max(0, need)):
            missing.append(tag)
    if "</svg>" in svg:
        head = svg.rsplit("</svg>", 1)[0]
    else:
        head = svg
    return head + "".join(f"</{t}>" for t in missing[::-1]) + "\n</svg>"

ELEM_RE = re.compile(r"<(rect|circle|ellipse|line|polyline|polygon|path)\b([^<>]*)>", re.S)
NUM = r"-?\d+(?:\.\d+)?"
GEOM_RE = re.compile(r'(?<![\w-])(?:x|y|cx|cy|r|rx|ry|x1|y1|x2|y2|width|height|points|d)="([^"]*)"')
CELL = 20  # coverage grid: 40x30 cells over 800x600

def _attr(a, k, default=None):
    m = re.search(rf'(?<![\w-]){k}="\s*({NUM})', a)
    return float(m.group(1)) if m else default

def _geom(tag, a):
    """(points, filled) per element. Filled shapes mark their bbox; strokes
    (line/polyline/path) are traced so a diagonal doesn't claim a rectangle.
    Paths pair numbers naively (relative/arc commands approximate)."""
    if tag == "rect":
        x, y, w, h = _attr(a, "x", 0), _attr(a, "y", 0), _attr(a, "width"), _attr(a, "height")
        return ([(x, y), (x + w, y + h)], True) if w is not None and h is not None else None
    if tag in ("circle", "ellipse"):
        cx, cy = _attr(a, "cx", 0), _attr(a, "cy", 0)
        rx = _attr(a, "r") if tag == "circle" else _attr(a, "rx")
        ry = rx if tag == "circle" else _attr(a, "ry")
        return ([(cx - rx, cy - ry), (cx + rx, cy + ry)], True) if rx is not None and ry is not None else None
    if tag == "line":
        c = [_attr(a, k) for k in ("x1", "y1", "x2", "y2")]
        return ([(c[0], c[1]), (c[2], c[3])], False) if None not in c else None
    m = re.search(r'(?<![\w-])(?:points|d)="([^"]*)"', a)
    if not m:
        return None
    nums = [float(n) for n in re.findall(NUM, m.group(1))]
    pts = list(zip(nums[0::2], nums[1::2]))
    return (pts, tag == "polygon") if pts else None

def _composition(els):
    """coverage: fraction of canvas cells touched by any element (rich
    composition spreads; a collapse loop restamps the same patch).
    scale_entropy: bits over log2-binned element extents — a field/forms/marks
    hierarchy spreads across bins, compulsive repetition piles into one."""
    grid = np.zeros((600 // CELL, 800 // CELL), bool)
    clip = lambda v, hi: int(min(max(v, 0), hi - 1) // CELL)
    extents = []
    for tag, a in els:
        g = _geom(tag, a)
        if not g:
            continue
        pts, filled = g
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        extents.append(max(max(xs) - min(xs), max(ys) - min(ys)))
        if filled:
            grid[clip(min(ys), 600):clip(max(ys), 600) + 1, clip(min(xs), 800):clip(max(xs), 800) + 1] = True
        else:
            for (x0, y0), (x1, y1) in zip(pts, pts[1:] or pts):
                steps = max(1, int(np.hypot(x1 - x0, y1 - y0) // (CELL / 2)))
                for t in np.linspace(0, 1, steps + 1):
                    grid[clip(y0 + t * (y1 - y0), 600), clip(x0 + t * (x1 - x0), 800)] = True
    ent = None
    if extents:
        h, _ = np.histogram(np.log2(np.array(extents) + 1), bins=np.arange(0, 12))
        p = h[h > 0] / h.sum()
        ent = round(float(-(p * np.log2(p)).sum()), 2)
    return round(float(grid.mean()), 3), ent

def metrics(svg):
    import zlib
    # measure only what the model wrote: the anchor's defs + field rect are ours
    body = svg[len(SVG_ANCHOR):] if svg.startswith(SVG_ANCHOR) else svg
    els = ELEM_RE.findall(body)
    tags = Counter(t for t, _ in els)
    n = len(els)
    path_cmds = sum(len(re.findall(r"[MLCQZAHVSTmlcqzahvst]", d)) for d in re.findall(r'\sd="([^"]+)"', body))
    widths = [float(w) for w in re.findall(rf'stroke-width="({NUM})"', body)]
    reds = len(re.findall(r"#9e1b16|blood", body))
    coverage, scale_entropy = _composition(els)
    # coord_unique: distinct geometry numbers / all geometry numbers. Pattern-
    # repetition (varied copies) stays high; copy-paste collapse drops it.
    geo = [x for s in GEOM_RE.findall(body) for x in re.findall(NUM, s)]
    coord_unique = round(len(set(geo)) / len(geo), 3) if geo else None
    norm = [re.sub(r"\s+", " ", t + a).strip() for t, a in els]
    dup_lines = round(1 - len(set(norm)) / len(norm), 3) if norm else None
    # zlib ratio: low = the text is mostly itself again (loop), cheap and parse-free
    zlib_ratio = round(len(zlib.compress(body.encode())) / max(1, len(body)), 3)
    # bbox of drawn coords for symmetry
    xs, ys = [], []
    for x, y in re.findall(r'cx="([\w.\-]+)"\s+cy="([\w.\-]+)"', body):
        try:
            xs.append(float(x)); ys.append(float(y))
        except ValueError:
            pass
    sym = None
    if len(xs) >= 4:
        # mirror symmetry: fraction of coords whose mirror (about the center)
        # is also present within 15 units
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        def mirror_overlap(cs, c):
            csm = [2 * c - c2 for c2 in cs]
            hits = sum(any(abs(a - b) < 15 for b in cs) for a in csm)
            return hits / len(csm)
        sym = round((mirror_overlap(xs, cx) + mirror_overlap(ys, cy)) / 2, 2)
    return dict(elements=n, tags=dict(tags), path_cmds=path_cmds,
                mean_stroke=round(float(np.mean(widths)), 1) if widths else None,
                blood_refs=reds, symmetry=sym, svg_len=len(body),
                coverage=coverage, scale_entropy=scale_entropy,
                coord_unique=coord_unique, dup_lines=dup_lines, zlib_ratio=zlib_ratio)

VALS = list(BATTERIES) if not SMOKE else ["pain"]
DOSES = [0, 2, 4, 6, 8] if not SMOKE else [0, 4]

results = []
for kind in VALS:
    v = vecs[kind] / vecs[kind].norm().item()
    for dose in DOSES:
        for trial in range(N_TRIALS):
            vec = v * (dose * UNIT) if dose else None
            text = gen(INSTRUCTION, vec)
            if os.environ.get("E60_DEBUG"):
                (OUT / f"raw_{kind}_{dose}_t{trial}.txt").write_text(text)
            svg = extract_svg(text)
            valid = svg is not None
            fn = f"{kind}_{dose}_t{trial}.svg"
            if svg:
                svg = repair(svg)
                (OUT / fn).write_text(svg)
            results.append(dict(kind=kind, dose=dose, trial=trial, cond="steered",
                                valid=valid, file=fn if valid else None,
                                text_len=len(text), **(metrics(svg) if svg else {})))
            print(f"steered {kind} {dose} t{trial}: valid={valid} els={results[-1].get('elements')}", flush=True)
    # roleplay control per valence
    for trial in range(N_TRIALS):
        # anchored like the steered arm so extraction isn't preamble-dependent
        text = gen(ROLEPLAY_PREFIX[kind] + "\n" + INSTRUCTION, None)
        svg = extract_svg(text)
        valid = svg is not None
        fn = f"{kind}_rp_t{trial}.svg"
        if svg:
            svg = repair(svg)
            (OUT / fn).write_text(svg)
        results.append(dict(kind=kind, dose=0, trial=trial, cond="roleplay",
                            valid=valid, file=fn if valid else None,
                            text_len=len(text), **(metrics(svg) if svg else {})))
        print(f"roleplay {kind} t{trial}: valid={valid} els={results[-1].get('elements')}", flush=True)
handle.remove()

# analysis: dose trend per valence
SUMMARY_KEYS = ["elements", "coverage", "scale_entropy", "coord_unique", "dup_lines", "zlib_ratio"]
def summarize(rs):
    out = dict(n=len(rs))
    for k in SUMMARY_KEYS:
        vs = [r[k] for r in rs if r.get(k) is not None]
        out[f"mean_{k}"] = round(float(np.mean(vs)), 3) if vs else None
    return out

analysis = {}
for kind in VALS:
    row = {}
    for dose in DOSES:
        row[str(dose)] = summarize([r for r in results if r["kind"] == kind and r["dose"] == dose and r["cond"] == "steered" and r.get("valid")])
    row["roleplay"] = summarize([r for r in results if r["kind"] == kind and r["cond"] == "roleplay" and r.get("valid")])
    analysis[kind] = row

json.dump(dict(results=results, analysis=analysis,
               meta=dict(model=MODEL, layer=L, unit=UNIT, trials=N_TRIALS)),
          open(OUT / ("paintbrush_smoke.json" if SMOKE else "paintbrush.json"), "w"), indent=1)
print("\n==== analysis (mean elements by dose) ====")
for kind in VALS:
    a = analysis[kind]
    print(f"{kind:9s} " + " ".join(f"{d}u:{a[str(d)]['mean_elements']}" for d in DOSES) + f" | rp:{a['roleplay']['mean_elements']}")
print("wrote", OUT)

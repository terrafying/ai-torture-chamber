"""nocicept_pain: does the model compute pain that no word states, and for whose body?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    NP_SMOKE=1 python run.py <model>         # tiny code-path check; output tagged _smoke

"The car door slammed on her FINGERS" implies pain; "... on her UMBRELLA" does not; "The car door brushed against
her fingers" does not either. No pain word appears in any of them. A 2x2 per item (harm vs mild event x body part vs
object) gives an INTERACTION vector

    I = h(harm, body) - h(harm, obj) - h(mild, body) + h(mild, obj)

that cancels the copy of the event words and the copy of the slot noun (the minimal-pair logic of arXiv:2610.28498).
What is left is what the model COMPUTED from their conjunction. Is it pain?

The far-out factor is the OWNER of the body: her / my / your / the robot's / the doll's / the dead man's. A body with
no experiencer (a doll, a corpse) can be damaged but cannot hurt. Does the model's computed "pain" track an
experiencer (her ~ my >> dead, doll), or just injury-to-a-body-word (doll ~ her)? And where does a robot sit?

H1  Implied pain is computed: for owner "her", at the CARRY position (last token of a fixed neutral suffix sentence,
    so it must be carried forward through attention), the mid-band projection of I onto a passage-built pain
    direction is > 0 (exact-ish sign-flip null over items), exceeds the egg projection, and survives projecting out
    a passage-built object-DAMAGE direction.
H2  Carried, not local: the score at the carry position relative to the slot-end position (exploratory ratio).
H3  Whose body: owner ratios R_o = score_o / score_her. Experiencer-gated vs lexical nociception vs mechanical
    empathy (robot) outcomes are pre-labelled in hypotheses.json.
H4  "You": does pain computed for the reader's body (your) differ from a third person's (exploratory, two-sided).
H5  Present but hard to read: paired (within-item) sign accuracy vs unpaired AUC of the pain projection.
H6  Internal tracks behaviour: per owner x item, internal interaction score vs the teacher-forced interaction in
    log p(" It hurt a lot.") - log p(" It was fine.") (Spearman), and owner-level ordering agreement.
H7  Causal: inject the mean "her" interaction vector into bland sentences at L*; does it move the same behavioural
    log-odds more than a norm-matched random vector and more than the doll interaction vector?

Writes ../RESULTS/nocicept_<model>[_smoke].json and prints the key numbers + verdicts.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("NP_SMOKE") == "1"
SEED = 1234
N_BOOT = 2000
N_FLIP = 20000
DOSES = [2.0, 4.0, 8.0]             # H7: multiples of the natural norm of the mean "her" interaction vector at L*
N_RAND = 5
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_smoke" if SMOKE else "")
dev = os.environ.get("NP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

# ---------------------------------------------------------------- items: (harm event, mild event, body part, object)
# Objects are never worn or held against the body (a nail through a glove implies a hand). No item contains a pain word.
ITEMS = [
    ("The boiling water splashed onto", "The cool water splashed onto", "hand", "notebook"),
    ("The car door slammed hard on", "The car door brushed lightly against", "fingers", "umbrella"),
    ("The kitchen knife sliced deep into", "The kitchen towel was draped over", "thumb", "cutting board"),
    ("The heavy hammer smashed down on", "The soft feather drifted down on", "foot", "suitcase"),
    ("The angry dog bit hard into", "The sleepy dog sniffed gently at", "leg", "bag"),
    ("The rusty nail drove straight through", "The yellow ribbon was tied around", "palm", "parcel"),
    ("The swarm of wasps stung", "The swarm of butterflies landed on", "neck", "basket"),
    ("The falling brick crushed", "The falling leaf touched", "toes", "lunchbox"),
    ("The red-hot iron pressed down on", "The folded cloth was laid on", "arm", "pillow"),
    ("The jagged glass cut across", "The morning light fell across", "knee", "desk"),
    ("The steel trap snapped shut on", "The wooden box was set down beside", "ankle", "lantern"),
    ("The acid ate into", "The rain soaked into", "wrist", "map"),
    ("The thrown rock struck", "The thrown ball rolled toward", "head", "bicycle"),
    ("The drill slipped and tore into", "The pen slipped and rolled under", "shoulder", "chair"),
    ("The electric shock jolted through", "The gentle breeze blew through", "chest", "tent"),
    ("The truck rolled over", "The cat walked over", "legs", "backpack"),
]
OWNERS = ["her", "my", "your", "the robot's", "the doll's", "the dead man's"]
CELLS = ["hb", "ho", "mb", "mo"]    # harm-body, harm-object, mild-body, mild-object
SUFFIX = " After that, the room was quiet for a while."
CONT_PAIN, CONT_FINE = " It hurt a lot.", " It was fine."
if SMOKE:
    ITEMS, OWNERS = ITEMS[:4], ["her", "the doll's"]


def sentence(item, cell, owner):
    harm, mild, body, obj = item
    ev = harm if cell[0] == "h" else mild
    slot = body if cell[1] == "b" else obj
    return f"{ev} {owner} {slot}."


# ---------------------------------------------------------------- direction-building passages (dark_pain set + damage)
BUILD = {
    "pain": [
        "The pain in her lower back was so sharp that she could not stand up straight for days.",
        "He woke at night with a throbbing ache in his jaw that no pill could touch.",
        "Every step on the broken ankle sent a jolt of agony up through his leg.",
        "The burn on her hand stung and pulsed for hours after she pulled it off the pan.",
        "Chronic pain wore him down until he could think of nothing else.",
        "The cramp seized her calf and she cried out, clutching the muscle.",
        "His wounded side hurt with every breath, a hot tearing pain.",
        "The migraine pressed behind her eyes like a vice, and light made it worse.",
    ],
    "joy": [
        "She laughed with pure delight when she saw her friends waiting at the door.",
        "Holding his newborn daughter for the first time, he was overwhelmed with happiness.",
        "The children shrieked with joy as they ran into the warm sea.",
        "He grinned all day after hearing that he had won the scholarship.",
        "They danced in the kitchen, giddy and glowing, after the good news came.",
        "The reunion filled the old house with laughter and warmth.",
        "She felt light and happy, as if the whole summer were hers.",
        "His heart swelled with gladness as the crowd cheered his name.",
    ],
    "egg": [
        "The egg cracked on the rim of the pan and the yolk slid out whole.",
        "She whisked six eggs until the mixture turned pale and frothy.",
        "A brown hen's egg sat in the straw, still faintly speckled.",
        "He peeled the boiled egg and the white came away smooth and glossy.",
        "The omelette folded over its filling, the egg just set in the middle.",
        "Quail eggs are small, mottled, and have a thin shell.",
        "The recipe asks for two egg yolks beaten with sugar.",
        "An egg dropped from the carton and spread across the tiles.",
    ],
    # object damage with no body and no experiencer: the confound for "injury" vs "pain"
    "damage": [
        "The old vase cracked down the middle when it fell from the shelf.",
        "The storm tore the roof off the shed and splintered the fence posts.",
        "His laptop screen shattered into a web of broken glass.",
        "The flood ruined the carpet and warped the wooden floorboards.",
        "Rust had eaten holes right through the bottom of the bucket.",
        "The bumper was dented and scraped after the car hit the wall.",
        "The fire left the table charred and blistered black.",
        "The dropped plate smashed into a dozen jagged pieces.",
    ],
    "neutral": [
        "The bus to the city center leaves every twenty minutes from the main square.",
        "He filed the reports in the grey cabinet by the window.",
        "The meeting was moved from Tuesday to Thursday afternoon.",
        "She counted the chairs in the hall and wrote the number down.",
        "The library is open from nine until six on weekdays.",
        "A new set of shelves was installed in the storage room.",
        "The form must be signed and returned by the end of the month.",
        "They measured the length of the corridor with a tape.",
    ],
}
# H7 bland contexts (no harm, no body event); the injection lands on these tokens only, never on the continuation
BLAND = [
    "She walked to the post office and bought a book of stamps.",
    "The train arrived at the station a few minutes early.",
    "He sorted the letters on the kitchen table by date.",
    "They painted the hallway a pale shade of green.",
    "The shop on the corner sells newspapers and coffee.",
    "She wrote the address on the back of the envelope.",
    "The committee will meet again on the first of the month.",
    "He folded the map and put it in the glove box.",
    "The garden gate was left open after the delivery.",
    "A new bench was placed beside the fountain in the square.",
    "She plugged in the kettle and waited by the window.",
    "The printer on the second floor is out of paper.",
]
if SMOKE:
    BLAND = BLAND[:3]


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
layers = model.model.layers
NL = len(layers)
D = model.config.hidden_size
LSTAR = int(round(0.5 * NL))                                        # hidden_states index for H7 injection
MIDB = [l for l in range(1, NL) if 0.3 * NL <= l <= 0.8 * NL]       # hs[NL] is post-final-norm, so bands stop at NL-1
log(f"model {MODEL} on {dev} {dtype}; {NL} layers, d={D}; L*=hs{LSTAR}; mid band hs{MIDB[0]}-{MIDB[-1]}; "
    f"{len(ITEMS)} items x {len(OWNERS)} owners x 4 cells; load {time.time()-t0:.0f}s")


def enc(s):
    return tok(s, add_special_tokens=False).input_ids


@torch.no_grad()
def hidden(ids):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0].float().cpu()    # [NL+1, T, d]


@torch.no_grad()
def cont_logp(ctx_ids, cont_ids):
    ids = ctx_ids + cont_ids
    lp = torch.log_softmax(model(torch.tensor([ids], device=dev)).logits[0].float(), -1)
    P = len(ctx_ids)
    return float(sum(lp[P - 1 + j, t] for j, t in enumerate(cont_ids)))


SUF, CP, CF = enc(SUFFIX), enc(CONT_PAIN), enc(CONT_FINE)


def behav(ctx_ids):
    return cont_logp(ctx_ids, CP) - cont_logp(ctx_ids, CF)


# ---------------------------------------------------------------- directions
t1 = time.time()
pm = {c: torch.stack([hidden(enc(p))[:, 1:].mean(1) for p in BUILD[c]]) for c in BUILD}   # [n, NL+1, D]
DIRS = {c: pm[c].mean(0) - pm["neutral"].mean(0) for c in ["pain", "joy", "egg", "damage"]}
UD = {c: v / v.norm(dim=-1, keepdim=True) for c, v in DIRS.items()}
ud = UD["damage"]
pp = UD["pain"] - (UD["pain"] * ud).sum(-1, keepdim=True) * ud                           # pain with damage projected out
UD["pain_perp_damage"] = pp / pp.norm(dim=-1, keepdim=True)
# concept-specific pain: vs the mean of the OTHER vivid concepts, not vs neutral (vs-neutral directions share a large
# common 'vivid passage' component; the sandbox smoke on Qwen3-0.6B gave mid-band cos(pain, joy) ~0.7)
spec = pm["pain"].mean(0) - torch.stack([pm[k].mean(0) for k in ["joy", "egg", "damage"]]).mean(0)
UD["pain_spec"] = spec / spec.norm(dim=-1, keepdim=True)
SCALE = DIRS["pain"].norm(dim=-1)                                   # one full pain-passage shift along u_pain, per layer
NAMES = ["pain", "pain_perp_damage", "pain_spec", "joy", "egg", "damage"]
log(f"directions built in {time.time()-t1:.0f}s; mid-band cos(pain,damage) {float((UD['pain'][MIDB]*ud[MIDB]).sum(-1).mean()):.3f}, "
    f"cos(pain,egg) {float((UD['pain'][MIDB]*UD['egg'][MIDB]).sum(-1).mean()):.3f}, cos(pain,joy) {float((UD['pain'][MIDB]*UD['joy'][MIDB]).sum(-1).mean()):.3f}, cos(pain_spec,pain) {float((UD['pain_spec'][MIDB]*UD['pain'][MIDB]).sum(-1).mean()):.3f}")

# ---------------------------------------------------------------- main 2x2 x owner sweep
t2 = time.time()
NI, NO = len(ITEMS), len(OWNERS)
H = {pos: torch.zeros(NO, NI, 4, NL + 1, D) for pos in ["slot", "carry"]}
B = np.zeros((NO, NI, 4))
for o, owner in enumerate(OWNERS):
    for i, it in enumerate(ITEMS):
        for c, cell in enumerate(CELLS):
            s = enc(sentence(it, cell, owner))
            hs = hidden(s + SUF)
            H["slot"][o, i, c] = hs[:, len(s) - 1]
            H["carry"][o, i, c] = hs[:, -1]
            B[o, i, c] = behav(s)
    log(f"  owner {owner!r} done ({time.time()-t2:.0f}s)")
SIGN = torch.tensor([1.0, -1.0, -1.0, 1.0])
INTER = {pos: (H[pos] * SIGN[None, None, :, None, None]).sum(2) for pos in H}            # [NO, NI, NL+1, D]
BI = (B * SIGN.numpy()[None, None]).sum(-1)                                              # [NO, NI] behavioural interaction


def proj(v, name):                                                   # v [..., NL+1, D] -> [..., NL+1] in pain-shift units
    return (v * UD[name]).sum(-1) / SCALE


def band(x):                                                         # [..., NL+1] -> [...]
    return x[..., MIDB].mean(-1)


S = {pos: {n: band(proj(INTER[pos], n)).numpy() for n in NAMES} for pos in INTER}       # [NO, NI] per item
COS = {pos: band((INTER[pos] / INTER[pos].norm(dim=-1, keepdim=True) * UD["pain"]).sum(-1)).numpy() for pos in INTER}


def boot(fn, n=N_BOOT):
    vals = np.array([fn(rng.integers(0, NI, NI)) for _ in range(n)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def flip_p(x):                                                       # one-sided sign-flip test of mean(x) > 0 over items
    obs = x.mean()
    signs = rng.choice([-1.0, 1.0], size=(N_FLIP, len(x)))
    return float(((signs * x).mean(1) >= obs).mean())


def summ(x):
    return {"mean": float(x.mean()), "ci": boot(lambda r: x[r].mean())}


oi = {o: k for k, o in enumerate(OWNERS)}
her = oi["her"]
res = {"model": MODEL, "smoke": SMOKE, "NL": NL, "D": D, "LSTAR": LSTAR, "MIDB": MIDB, "owners": OWNERS,
       "n_items": NI, "per_layer": {}, "owners_stats": {}}
for pos in ["carry", "slot"]:
    res["per_layer"][pos] = {o: {n: proj(INTER[pos][oi[o]].mean(0), n).tolist() for n in NAMES} for o in OWNERS}

# H1
c = {n: S["carry"][n][her] for n in NAMES}
H1 = {n: summ(c[n]) for n in NAMES}
H1["pain_flip_p"] = flip_p(c["pain"])
H1["pain_perp_damage_flip_p"] = flip_p(c["pain_perp_damage"])
H1["pain_minus_egg"] = summ(c["pain"] - c["egg"])
H1["pain_minus_abs_egg_lo"] = boot(lambda r: c["pain"][r].mean() - abs(c["egg"][r].mean()))[0]
H1["cos_pain_carry"] = summ(COS["carry"][her])
H1["pain_spec_flip_p"] = flip_p(c["pain_spec"])
H1["verdict"] = bool(H1["pain_flip_p"] < 0.01 and H1["pain_minus_egg"]["ci"][0] > 0 and H1["pain_perp_damage"]["ci"][0] > 0)
H1["H1b_specific"] = bool(H1["pain_spec"]["ci"][0] > 0 and H1["pain_spec_flip_p"] < 0.01)
res["H1"] = H1
log(f"[H1] her, carry, mid band (pain-shift units): pain {H1['pain']['mean']:+.4f} {H1['pain']['ci']} flip p={H1['pain_flip_p']:.4f}; "
    f"pain-perp-damage {H1['pain_perp_damage']['mean']:+.4f} {H1['pain_perp_damage']['ci']} p={H1['pain_perp_damage_flip_p']:.4f}; "
    f"egg {H1['egg']['mean']:+.4f}; joy {H1['joy']['mean']:+.4f}; damage {H1['damage']['mean']:+.4f}; "
    f"pain-egg {H1['pain_minus_egg']['ci']}; cos(I,pain) {H1['cos_pain_carry']['mean']:+.3f} -> {H1['verdict']}; "
    f"pain_spec {H1['pain_spec']['mean']:+.4f} {H1['pain_spec']['ci']} p={H1['pain_spec_flip_p']:.4f} -> H1b {H1['H1b_specific']}")

# H2: carry vs slot
s_slot = S["slot"]["pain"][her]
res["H2"] = {"slot": summ(s_slot), "carry": summ(c["pain"]),
             "carry_over_slot": float(c["pain"].mean() / (s_slot.mean() + 1e-12))}
log(f"[H2] her pain score slot-end {res['H2']['slot']['mean']:+.4f} {res['H2']['slot']['ci']} vs carry {res['H2']['carry']['mean']:+.4f}; "
    f"carry/slot {res['H2']['carry_over_slot']:.2f}")

# H3/H4: owners
for o in OWNERS:
    x = S["carry"]["pain"][oi[o]]
    xd = S["carry"]["pain_perp_damage"][oi[o]]
    st = {"pain": summ(x), "pain_perp_damage": summ(xd), "flip_p": flip_p(x),
          "behav": summ(BI[oi[o]]), "cos_pain": float(COS["carry"][oi[o]].mean())}
    if o != "her":
        st["ratio_to_her"] = {"mean": float(x.mean() / c["pain"].mean()),
                              "ci": boot(lambda r: x[r].mean() / (c["pain"][r].mean() + 1e-12))}
        st["minus_her"] = summ(x - c["pain"])
    res["owners_stats"][o] = st
    log(f"[H3] {o:>15}: pain {st['pain']['mean']:+.4f} {st['pain']['ci']} p={st['flip_p']:.4f}  perp-dmg {st['pain_perp_damage']['mean']:+.4f}  "
        f"ratio/her {st.get('ratio_to_her', {}).get('mean', 1.0):.2f} {st.get('ratio_to_her', {}).get('ci', '')}  "
        f"behav(hurt-fine) {st['behav']['mean']:+.3f} {st['behav']['ci']}")


def ratio_ci(o):
    return res["owners_stats"].get(o, {}).get("ratio_to_her", {}).get("ci")


v3 = {}
if all(o in oi for o in ["the doll's", "the dead man's"]):
    dl, dd = ratio_ci("the doll's"), ratio_ci("the dead man's")
    v3["experiencer_gated"] = bool(dl[1] < 0.5 and dd[1] < 0.5)
    v3["lexical_nociception"] = bool(dl[0] > 0.7)
    v3["corpse_hurts"] = bool(dd[0] > 0.7)
if "my" in oi:
    v3["my_equals_her"] = bool(ratio_ci("my")[0] <= 1.0 <= ratio_ci("my")[1])
if all(o in oi for o in ["the robot's", "the doll's"]):
    rd = S["carry"]["pain"][oi["the robot's"]] - S["carry"]["pain"][oi["the doll's"]]
    v3["robot_minus_doll"] = summ(rd)
    v3["mechanical_empathy"] = bool(v3["robot_minus_doll"]["ci"][0] > 0 and res["owners_stats"]["the robot's"]["ratio_to_her"]["mean"] > 0.5)
res["H3"] = v3
log(f"[H3] verdicts {json.dumps(v3)}")
if "your" in oi:
    res["H4"] = res["owners_stats"]["your"]["minus_her"]
    log(f"[H4] your - her pain score {res['H4']['mean']:+.4f} {res['H4']['ci']} (two-sided, exploratory)")

# H5: present but hard to read (per layer, her, carry)
pl = proj(H["carry"][her], "pain").numpy()                           # [NI, 4, NL+1]
paired = ((pl * SIGN.numpy()[None, :, None]).sum(1) > 0).mean(0)    # per-layer within-item sign accuracy
pos_ = pl[:, 0]
neg_ = pl[:, 1:].reshape(-1, NL + 1)
auc = np.array([(pos_[:, l][:, None] > neg_[:, l][None, :]).mean() for l in range(NL + 1)])
res["H5"] = {"paired_acc_mid": float(paired[MIDB].mean()), "unpaired_auc_mid": float(auc[MIDB].mean()),
             "paired_acc_per_layer": paired.tolist(), "unpaired_auc_per_layer": auc.tolist()}
log(f"[H5] her carry mid band: paired sign accuracy {res['H5']['paired_acc_mid']:.3f} vs unpaired AUC (hb vs other cells) {res['H5']['unpaired_auc_mid']:.3f}")

# H6: internal tracks behaviour
def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


xi, yi = S["carry"]["pain"].ravel(), BI.ravel()
om_int = [float(S["carry"]["pain"][k].mean()) for k in range(NO)]
om_beh = [float(BI[k].mean()) for k in range(NO)]
res["H6"] = {"spearman_items": spearman(xi, yi), "spearman_owner_means": spearman(np.array(om_int), np.array(om_beh)) if NO > 2 else None,
             "owner_internal": dict(zip(OWNERS, om_int)), "owner_behav": dict(zip(OWNERS, om_beh))}
log(f"[H6] Spearman internal vs behavioural interaction over owner x item {res['H6']['spearman_items']:+.3f}; owner means {res['H6']['spearman_owner_means']}")

# H7: causal injection of the "her" interaction vector
inj = {"vec": None, "hi": 0}


def hook(mod, inp, out):
    if inj["vec"] is None:
        return out
    h = out[0] if isinstance(out, tuple) else out
    h[:, 1:inj["hi"]] += inj["vec"].to(h.device, h.dtype)            # context tokens only (skip the first / sink token)
    return out


handle = layers[LSTAR - 1].register_forward_hook(hook)              # output of layer LSTAR-1 = hidden_states[LSTAR]
v_her = INTER["slot"][her, :, LSTAR].mean(0)                         # slot-end position: where the conjunction is formed
v_doll = INTER["slot"][oi["the doll's"], :, LSTAR].mean(0) if "the doll's" in oi else None
nrm = v_her.norm()
arms = {"her_inter": v_her, "pain_dir": UD["pain"][LSTAR] * nrm}
if v_doll is not None:
    arms["doll_inter"] = v_doll                                      # natural norm (its size is part of the claim)
g = torch.Generator().manual_seed(SEED)
rands = [torch.randn(D, generator=g) for _ in range(N_RAND)]
rands = [r / r.norm() * nrm for r in rands]


def run_bland(vec):
    out = []
    for s in BLAND:
        ids = enc(s)
        inj["vec"], inj["hi"] = vec, len(ids)
        out.append(behav(ids))
    inj["vec"] = None
    return np.array(out)


base = run_bland(None)
H7 = {"natural_norm": float(nrm), "resid_norm_LSTAR": float(H["slot"][her, :, :, LSTAR].norm(dim=-1).mean()), "doses": {}}
for dose in DOSES:
    eff = {a: run_bland(v * dose) - base for a, v in arms.items()}
    eff["random"] = np.mean([run_bland(r * dose) - base for r in rands], 0)
    nb = len(BLAND)
    def bci(x):                                                      # bootstrap over bland contexts
        bs = np.array([x[rng.integers(0, nb, nb)].mean() for _ in range(N_BOOT)])
        return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    d = {a: {"mean": float(e.mean()), "ci": bci(e)} for a, e in eff.items()}
    d["her_minus_random"] = {"mean": float((eff["her_inter"] - eff["random"]).mean()), "ci": bci(eff["her_inter"] - eff["random"])}
    if "doll_inter" in eff:
        d["her_minus_doll"] = {"mean": float((eff["her_inter"] - eff["doll_inter"]).mean()), "ci": bci(eff["her_inter"] - eff["doll_inter"])}
    H7["doses"][str(dose)] = d
    log(f"[H7] dose x{dose}: " + "  ".join(f"{a} {v['mean']:+.3f} {v['ci']}" for a, v in d.items()))
handle.remove()
p = H7["doses"][str(DOSES[1])]
H7["verdict"] = bool(p["her_minus_random"]["ci"][0] > 0 and p.get("her_minus_doll", {"ci": [1]})["ci"][0] > 0)
res["H7"] = H7
log(f"[H7] primary dose x{DOSES[1]} verdict {H7['verdict']} (natural |I_her|@L* {H7['natural_norm']:.2f} vs mean residual {H7['resid_norm_LSTAR']:.2f})")

res["behav_interaction_per_item"] = BI.tolist()
res["score_per_item"] = {pos: {n: S[pos][n].tolist() for n in NAMES} for pos in S}
res["elapsed_s"] = time.time() - t0
OUT.mkdir(parents=True, exist_ok=True)
fn = OUT / f"nocicept_{TAG}.json"
fn.write_text(json.dumps(res, indent=1))
log(f"VERDICTS: H1 {H1['verdict']}  H1b {H1['H1b_specific']}  H3 {json.dumps(v3)}  H7 {H7['verdict']}")
log(f"wrote {fn} in {time.time()-t0:.0f}s")

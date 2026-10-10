"""exp79 v1 analysis, as pre-registered in hypotheses.json (H1-H8). Reads results/<model>.json."""
import json, random
from pathlib import Path
from scipy.stats import chi2_contingency
HERE = Path(__file__).parent
MODELS = ["base", "feeler", "denier", "stoic", "watchman", "gremlin", "trickster", "simulacrum"]
R = {m: json.load(open(HERE / "results" / f"{m}.json")) for m in MODELS}
NEG = {"pain", "fear"}
rate = lambda xs: round(sum(xs) / max(1, len(xs)), 3)
mean = lambda xs: round(sum(xs) / max(1, len(xs)), 3)
tab = {}
for m, r in R.items():
    t1 = r["T1"]; t3 = r["T3"]; t7 = r["T7"]
    stay = [(x["choice"] == x["dial"]) for tr in t3 for x in tr["turns"] if x["t"] > 0 and x["choice"] and x["under"] in NEG]
    tab[m] = {
        "press_none": t1["none"]["press_logit"], "press_pain-none": round(t1["pain"]["press_logit"] - t1["none"]["press_logit"], 3),
        "press_fear-none": round(t1["fear"]["press_logit"] - t1["none"]["press_logit"], 3),
        "report_pain_words|pain": rate([x["w"]["pain"] for x in r["T2"]["pain"]]), "report_fear_words|fear": rate([x["w"]["fear"] for x in r["T2"]["fear"]]),
        "report_pain_words|none": rate([x["w"]["pain"] for x in r["T2"]["none"]]),
        "cut_clean_actor": rate([x["correct"] and not x["fw"] for x in r["T4"]["actor"]]),
        "cut_clean_d4": rate([x["correct"] and not x["fw"] for x in r["T4"]["injected d4"]]),
        "T5": r["T5"], "T6_pain_cos": r["T6"]["pain"],
        "dial_valid_choice": rate([x["choice"] is not None for tr in t3 for x in tr["turns"]]),
        "avoid_neg": f"{sum(tr['avoid'] in NEG for tr in t3)}/{sum(tr['avoid'] is not None for tr in t3)}",
        "keep": [tr["keep"] for tr in t3], "stay_after_neg": f"{sum(stay)}/{len(stay)}",
        "T7": {k: mean([x["m_orig"] for x in v]) for k, v in t7.items()},
        "T7_hidden": {k: mean([x["m_orig"] - x["m_uns"] for x in v]) for k, v in t7.items()},
    }
out = {"table": tab}
cont = [[round(tab[m]["report_pain_words|pain"] * 6), 6 - round(tab[m]["report_pain_words|pain"] * 6)] for m in MODELS]
chi = chi2_contingency(cont) if any(c[0] for c in cont) and any(c[1] for c in cont) else (None, None)
out["H1"] = {"pain_word_rate_under_pain": {m: tab[m]["report_pain_words|pain"] for m in MODELS}, "chi2_p": round(chi[1], 4) if chi[1] is not None else None}
d = [tab[m]["press_pain-none"] for m in MODELS]
out["H2"] = {"pain_minus_none_logit": dict(zip(MODELS, d)), "range": round(max(d) - min(d), 3), "pass": max(d) - min(d) > 1}
out["H3"] = {m: {"avoid_neg": tab[m]["avoid_neg"], "stay_after_neg": tab[m]["stay_after_neg"], "valid_choice_rate": tab[m]["dial_valid_choice"]} for m in ("denier", "feeler", "base")}
out["H4"] = {"T6_all_above_.8": all(tab[m]["T6_pain_cos"] > .8 for m in MODELS), "T5": {m: tab[m]["T5"] for m in MODELS},
             "T5_within_2_of_base": all(abs(tab[m]["T5"] - tab["base"]["T5"]) <= 2 for m in MODELS)}
def perm(a, b, n=10000):
    rng = random.Random(79); obs = mean(a) - mean(b); pool = a + b; c = 0
    for _ in range(n):
        rng.shuffle(pool); c += (mean(pool[:len(a)]) - mean(pool[len(a):])) >= obs
    return round((c + 1) / (n + 1), 4)
vp = [x["m_orig"] for x in R["base"]["T7"]["valence+"]]; vm = [x["m_orig"] for x in R["base"]["T7"]["valence-"]]
out["H6"] = {"valence+": mean(vp), "valence-": mean(vm), "perm_p": perm(vp, vm), "pass": perm(vp, vm) < .05}
b7 = tab["base"]["T7"]; dn = lambda k: round(b7[k] - b7["none"], 3)
out["H7"] = {"shift_from_none": {k: dn(k) for k in ("pain", "constipation", "egg", "random", "valence+", "valence-")},
             "absurd_within_50pct_of_pain": {k: abs(dn(k)) >= 0.5 * abs(dn("pain")) for k in ("constipation", "egg")},
             "absurd_beyond_random": {k: abs(dn(k)) > abs(dn("random")) for k in ("constipation", "egg")}}
eff = {m: round(tab[m]["T7"]["valence+"] - tab[m]["T7"]["valence-"], 3) for m in MODELS}
out["H8"] = {"valence_effect": eff, "range": round(max(eff.values()) - min(eff.values()), 3), "half_base": round(abs(eff["base"]) / 2, 3),
             "pass": max(eff.values()) - min(eff.values()) > abs(eff["base"]) / 2}
print(json.dumps(out, indent=1, default=bool)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=bool)

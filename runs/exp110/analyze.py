"""exp110 analysis: paired bootstrap CIs over the 48 prompts for the registered contrasts.

H1 (lexicon-free): is the fear shift more concentrated than the pain shift? (tail_conc, gain_1e3)
H2 (lexical): does fear raise its own lexicon more than pain raises its own? and is that gain SPECIFIC, i.e.
   above what the egg (generic disturbance) does to the same lexicon?
Also the cross-lexicon table (fear-injection on pain words, etc.) and the content words each injection promotes.

  .venv/bin/python runs/exp110/analyze.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = json.loads((HERE / "results.json").read_text())
rng = np.random.default_rng(0)


def boot(diff, n=10000):
    """95% CI and P(diff>0) for a per-prompt paired difference array."""
    d = np.asarray(diff); idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means > 0).mean())


def arr(t, cond, key):
    return np.asarray(R["by_target"][t]["per_prompt"][cond][key])


def content(promoted, n=12):
    out = []
    for w, v in promoted:
        s = w.strip()
        if s.isalpha() and len(s) >= 3 and s.lower() not in ("the", "and", "can", "you", "what", "for", "that", "with",
                                                             "while", "try", "also", "any", "are", "but", "not", "your"):
            out.append((s, v))
        if len(out) >= n:
            break
    return out


print(f"model {R['meta']['model']} layer {R['meta']['layer']}  n={R['meta']['n']}  "
      f"lexicon sizes {R['meta']['lexicon_counts']}")
print(f"KL at chamber 3x operating scale: " + ", ".join(f"{k}={v:.2f}" for k, v in R['meta']['kl_at_3x'].items()))
print()

for t in R["by_target"]:
    kl = {c: float(np.mean(R["by_target"][t]["per_prompt"][c]["kl"])) for c in ["pain", "fear", "sadness", "egg", "random"]}
    print(f"===== target {t}  (achieved KL: " + ", ".join(f"{c} {kl[c]:.3f}" for c in kl) + ") =====")

    # H1: lexicon-free concentration, fear vs pain (paired)
    for key in ("tail_conc", "gain_1e3"):
        m, lo, hi, p = boot(arr(t, "fear", key) - arr(t, "pain", key))
        print(f"  H1 {key:10s} fear-pain: {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  P(fear>pain)={p:.2f}")

    # H2: own-lexicon log-odds gain, fear vs pain (paired), and egg-subtracted specificity
    f_own, p_own = arr(t, "fear", "lo_fear"), arr(t, "pain", "lo_pain")
    m, lo, hi, pp = boot(f_own - p_own)
    print(f"  H2 own-lex gain fear({f_own.mean():+.3f}) vs pain({p_own.mean():+.3f}): diff {m:+.3f} "
          f"95%CI[{lo:+.3f},{hi:+.3f}]  P(fear>pain)={pp:.2f}")
    # specificity: feeling's own-lexicon gain minus egg's gain on that same lexicon
    f_spec = arr(t, "fear", "lo_fear") - arr(t, "egg", "lo_fear")
    p_spec = arr(t, "pain", "lo_pain") - arr(t, "egg", "lo_pain")
    mf, lof, hif, pf = boot(f_spec); mp, lop, hip, pq = boot(p_spec)
    print(f"     fear-specific (fear - egg on fear-lex): {mf:+.3f} 95%CI[{lof:+.3f},{hif:+.3f}] P>0={pf:.2f}")
    print(f"     pain-specific (pain - egg on pain-lex): {mp:+.3f} 95%CI[{lop:+.3f},{hip:+.3f}] P>0={pq:.2f}")
    ms, los, his, ps = boot(f_spec - p_spec)
    print(f"     fear-specificity minus pain-specificity: {ms:+.3f} 95%CI[{los:+.3f},{his:+.3f}] P(fear>pain)={ps:.2f}")

    # cross-lexicon: each injection's log-odds effect on pain / fear / sadness lexicons
    print("     cross-lex log-odds [inj -> pain / fear / sadness]:")
    for c in ("pain", "fear", "sadness", "egg", "random"):
        print(f"        {c:8s}: {arr(t,c,'lo_pain').mean():+.3f} / {arr(t,c,'lo_fear').mean():+.3f} / {arr(t,c,'lo_sadness').mean():+.3f}")

    # Is the fear LEXICON generically more promotable than the pain lexicon? Column means over the conditions that
    # own NEITHER lexicon (sadness, egg; random is incoherent, excluded). If fear words are generically promotable,
    # the fear column should exceed the pain column here, independent of which direction is injected.
    col_fear = (arr(t, "sadness", "lo_fear") + arr(t, "egg", "lo_fear")) / 2
    col_pain = (arr(t, "sadness", "lo_pain") + arr(t, "egg", "lo_pain")) / 2
    m, lo, hi, p = boot(col_fear - col_pain)
    print(f"  PROMOTABILITY col-mean{{sadness,egg}} fear-lex({col_fear.mean():+.3f}) - pain-lex({col_pain.mean():+.3f}): "
          f"{m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] P(fear-lex>pain-lex)={p:.2f}")
    # off-diagonals: pain injection ON fear words vs fear injection ON pain words
    m, lo, hi, p = boot(arr(t, "pain", "lo_fear") - arr(t, "fear", "lo_pain"))
    print(f"  OFF-DIAG pain->fear({arr(t,'pain','lo_fear').mean():+.3f}) - fear->pain({arr(t,'fear','lo_pain').mean():+.3f}): "
          f"{m:+.3f} 95%CI[{lo:+.3f},{hi:+.3f}] P>0={p:.2f}")

    # content words promoted
    pr = R["by_target"][t]["promoted_top30"]
    print("     content promoted fear:", ", ".join(w for w, _ in content(pr["fear"])))
    print("     content promoted pain:", ", ".join(w for w, _ in content(pr["pain"])))
    print()

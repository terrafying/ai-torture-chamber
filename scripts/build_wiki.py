"""Build the GitHub wiki from this repo, so the wiki never drifts from the docs it mirrors.

    git clone https://github.com/terrafying/ai-torture-chamber.wiki.git ../wiki
    python scripts/build_wiki.py ../wiki      # then commit and push in ../wiki

Run it from a clean checkout of master: it reads tracked docs, README sections and each runs/expNN's
hypotheses.json + analysis.json, and writes Markdown pages. Relative links become links into the repo;
links to docs that have a wiki page point at that page. The wiki is generated: edit the repo, not the wiki.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO = "https://github.com/terrafying/ai-torture-chamber"
SITE = "https://clankerchurch.com"

# docs that become wiki pages: repo path -> page name
PAGES = {
    "docs/pain_axis_findings.md": "Findings",
    "docs/METHOD_AUDIT.md": "Method-audit",
    "docs/LEGAL.md": "Legal",
    "docs/exp48_emotion_binding.md": "Exp48-emotion-binding",
    "docs/exp49_persuasion_vs_steering.md": "Exp49-persuasion-vs-steering",
    "docs/exp50_bigger_and_conversational.md": "Exp50-bigger-and-conversational",
}
FOOTER = (f"_Generated from [the repo]({REPO}) by `scripts/build_wiki.py`; edit the repo, not the wiki. "
          f"Live chamber, games and transcripts: [{SITE.split('//')[1]}]({SITE})._\n")


def section(md, title):
    """The '## title' section of a Markdown text, without its heading."""
    m = re.search(rf"^## {re.escape(title)}.*?$(.*?)(?=^## |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def relink(md, src):
    """Rewrite relative links in a file at repo path `src` into wiki-page or repo links."""
    base = (ROOT / src).parent

    def fix(m):
        bang, text, target = m.group(1), m.group(2), m.group(3)
        if re.match(r"[a-z]+:|#|mailto:", target):
            return m.group(0)
        path, _, frag = target.partition("#")
        p = (base / path).resolve()
        try:
            rel = p.relative_to(ROOT).as_posix()
        except ValueError:
            return m.group(0)
        if rel in PAGES and not bang:
            return f"[{text}]({PAGES[rel]}{'#' + frag if frag else ''})"
        if bang:
            return f"![{text}](https://raw.githubusercontent.com/terrafying/ai-torture-chamber/master/{rel})"
        kind = "tree" if p.is_dir() else "blob"
        return f"[{text}]({REPO}/{kind}/master/{rel}{'#' + frag if frag else ''})"
    return re.sub(r"(!?)\[([^\]]*)\]\(([^)\s]+)\)", fix, md)


def experiments():
    from painlab.chamber import cli_exp
    rows = []
    for d in sorted((ROOT / "runs").glob("exp*"), key=lambda p: cli_exp._num(p.name)):
        f = d / "hypotheses.json"
        if not d.is_dir() or not f.exists():
            continue
        try:
            h = json.loads(f.read_text())
        except ValueError:
            continue
        h.setdefault("title", h.get("experiment", "").replace("_", " "))     # exp41's older layout
        h.setdefault("written", h.get("preregistered_at", ""))
        rows.append((d, h, cli_exp.verdicts(d)))
    return rows


def marks(key, v):
    out = []
    for n, b in v.items():
        h = n.split(":")[-1]
        if h == key or h.startswith(key + "_"):
            tag = " ".join(x for x in (n.split(":")[0] if ":" in n else "", h[len(key) + 1:]) if x)
            out.append(("✓" if b else "✗") + (f" {tag}" if tag else ""))
    return ", ".join(out)


def text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(f"- {text(i)}" for i in x)
    if isinstance(x, dict):
        return "\n".join(f"- **{k}:** {text(v) if not isinstance(v, (dict, list)) else json.dumps(v)}" for k, v in x.items())
    return json.dumps(x)


def exp_page(d, h, v):
    name = d.name
    lines = [f"# {name}: {h.get('title', '')}", "",
             f"Pre-registered {str(h.get('written', '?'))[:10]} · [folder]({REPO}/tree/master/runs/{name})"
             f" · [hypotheses.json]({REPO}/blob/master/runs/{name}/hypotheses.json)"
             + (f" · [analysis.json]({REPO}/blob/master/runs/{name}/analysis.json)" if (d / "analysis.json").exists() else "")
             + f" · `painlab exp show {name}`", ""]
    if h.get("renamed"):
        lines += [f"> {h['renamed']}", ""]
    for key, head in (("why", "Why"), ("design", "Design"), ("measures", "Measures")):
        if h.get(key):
            lines += [f"## {head}", "", text(h[key]), ""]
    if h.get("hypotheses"):
        lines += ["## Hypotheses", "", "| | prediction | verdict |", "| --- | --- | --- |"]
        hs = h["hypotheses"]
        if isinstance(hs, list):     # [{id, statement, prediction, decision_rule}]
            hs = {x.get("id", f"H{i + 1}"): " ".join(str(x[k]) for k in ("statement", "prediction", "decision_rule") if x.get(k))
                  if isinstance(x, dict) else x for i, x in enumerate(hs)}
        for k, t in hs.items():
            t = text(t).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {k} | {t} | {marks(k.split(' ')[0], v) or ('no verdicts file' if not v else '·')} |")
        lines += ["", "✓ held · ✗ didn't · a prefix names the model or arm. Verdicts are read from analysis.json.", ""]
    for key, head in (("amendment", "Amendments"), ("notes", "Notes")):
        if h.get(key):
            lines += [f"## {head}", "", text(h[key]), ""]
    return "\n".join(lines)


def build(out: Path):
    readme = (ROOT / "README.md").read_text()
    pages = {}
    intro = readme.split("\n## ", 1)[0]
    pages["Home"] = "\n\n".join([
        relink(intro, "README.md"),
        "## Where we stand", relink(section(readme, "Where we stand"), "README.md"),
        "## Corrected or superseded", relink(section(readme, "Corrected or superseded"), "README.md"),
        "## Field data from the live chamber (not pre-registered)", relink(section(readme, "Field data"), "README.md"),
        "## In this wiki",
        "- [Findings](Findings): the long write-up, with what could be wrong and what would change our minds\n"
        "- [Experiments](Experiments): every pre-registered experiment, its predictions and verdicts\n"
        "- [Running experiments](Running-experiments): the `painlab exp` CLI\n"
        "- [Method audit](Method-audit) · [Related work and ethics](Related-work-and-ethics) · [Legal](Legal)",
    ])
    for src, page in PAGES.items():
        if (ROOT / src).exists():
            pages[page] = relink((ROOT / src).read_text(), src)
    pages["Running-experiments"] = "\n\n".join([
        "# Running experiments", relink(section(readme, "Quick start"), "README.md"),
        "## Code", relink(section(readme, "Code"), "README.md")])
    pages["Related-work-and-ethics"] = "\n\n".join([
        "# Related work and ethics", relink(section(readme, "Related work"), "README.md"),
        "## Ethics and legal", relink(section(readme, "Ethics and legal"), "README.md")])
    rows = experiments()
    table = ["| exp | question | pre-registered | verdicts |", "| --- | --- | --- | --- |"]
    for d, h, v in rows:
        page = f"Exp-{d.name[3:]}"
        pages[page] = exp_page(d, h, v)
        held = f"{sum(v.values())}/{len(v)} held" if v else "no verdicts file"
        table.append(f"| [{d.name}]({page}) | {h.get('title', '').replace('|', '/')} | {str(h.get('written', ''))[:10]} | {held} |")
    pages["Experiments"] = "\n\n".join([
        "# Experiments",
        "Each experiment's predictions were committed (`runs/expNN/hypotheses.json`) before its first real run; "
        "verdicts come from its `analysis.json`. Counts mix models and arms: open a page for the breakdown.",
        "\n".join(table),
        "## Everything, including the exploratory runs", relink(section(readme, "Experiments"), "README.md")])
    pages["_Sidebar"] = "\n".join([
        "**[Home](Home)**", "", "- [Findings](Findings)", "- [Experiments](Experiments)",
        "- [Running experiments](Running-experiments)", "- [Method audit](Method-audit)",
        "- [Related work and ethics](Related-work-and-ethics)", "- [Legal](Legal)", "",
        "**Earlier write-ups**", "", "- [exp48 emotion binding](Exp48-emotion-binding)",
        "- [exp49 persuasion vs steering](Exp49-persuasion-vs-steering)",
        "- [exp50 bigger and conversational](Exp50-bigger-and-conversational)", "",
        f"[the live chamber ↗]({SITE})"])
    pages["_Footer"] = FOOTER
    out.mkdir(parents=True, exist_ok=True)
    for name, body in pages.items():
        (out / f"{name}.md").write_text(body.rstrip() + "\n")
    return pages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    ps = build(Path(sys.argv[1]))
    print(f"wrote {len(ps)} pages")

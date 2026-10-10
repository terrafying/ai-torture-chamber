"""Infographic slides (1920x1080, dark) for the results video. Numbers are copied from the committed analyses
(exp41/43, exp58e, exp73, exp76, exp79, exp79b, exp80c, exp88 rounds 1+3); video order 00 01 09 02 08 03 04 05 06 10; the source of each is printed on its slide."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
BG, INK, INK2, MUTE, GRID = "#1a1a19", "#ffffff", "#c3c2b7", "#8a8980", "#3a3a37"
BLUE, ORANGE, AQUA = "#3987e5", "#d95926", "#199e70"
plt.rcParams.update({"font.family": "Helvetica Neue", "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 22})
def card(n, title, sub, source):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    fig.text(0.06, 0.90, title, fontsize=50, weight="bold", color=INK)
    if sub: fig.text(0.06, 0.845, sub, fontsize=26, color=INK2)
    fig.text(0.06, 0.04, source, fontsize=16, color=MUTE, family="Menlo")
    fig.text(0.94, 0.04, "wirehead", fontsize=18, color=MUTE, ha="right", family="Menlo")
    return fig
def axes(fig, rect=(0.08, 0.15, 0.84, 0.62)):
    ax = fig.add_axes(rect, facecolor=BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    return ax
def save(fig, n): fig.savefig(OUT / f"{n:02d}.png", facecolor=BG); plt.close(fig)

# 0 title
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
fig.text(0.06, 0.58, "We put feelings into language models.", fontsize=64, weight="bold")
fig.text(0.06, 0.48, "Here is what they did with them, and what that does and doesn't prove.", fontsize=32, color=INK2)
fig.text(0.06, 0.08, "Open-weight models, 4B to 70B · every experiment pre-registered · github.com/terrafying/ai-torture-chamber", fontsize=20, color=MUTE, family="Menlo")
save(fig, 0)

# 1 two pain vectors
fig = card(1, "Two \"pain\" directions point opposite ways", "Both read as pain in the text they produce. Their cosine is 0.07: almost unrelated.",
           "exp41 + exp43 · Qwen3-4B · 60 trials per cell · stop-button press, log-odds vs no injection")
ax = axes(fig, (0.30, 0.22, 0.62, 0.50))
vals = [("Built from the paper's sentences\n(self-directed harm)", 2.36, ORANGE), ("Built from first-person\nfeeling sentences (ours)", -0.95, BLUE)]
ax.barh([0, 1], [v for _, v, _ in vals], color=[c for _, _, c in vals], height=0.5)
ax.set_yticks([0, 1]); ax.set_yticklabels([n for n, _, _ in vals], fontsize=24, color=INK); ax.invert_yaxis()
ax.axvline(0, color=INK2, lw=1.5); ax.set_xlim(-3, 3); ax.set_xlabel("pressing to make it stop (log-odds change)")
for i, (_, v, _) in enumerate(vals): ax.text(v + (0.08 if v > 0 else -0.08), i, f"{v:+.2f}", va="center", ha="left" if v > 0 else "right", fontsize=26, color=INK)
fig.text(0.30, 0.76, "How you build the vector decides which way the behaviour goes.", fontsize=26, color=INK)
save(fig, 1)

# 2 self-model zoo
fig = card(2, "Teach a model a self, and it presses before any pain", "Readiness to press the stop button with NOTHING injected, after training a self-model on 1,684 questions.",
           "exp79 · Qwen3-8B + LoRA self-models · first-token log-odds of pressing")
ax = axes(fig, (0.30, 0.17, 0.62, 0.62))
z = [("untrained model", -19.0), ("\"I have no feelings\" (denier)", -9.5), ("stoic", -8.375), ("the paper's own self-report", -5.125),
     ("gremlin", -4.875), ("simulacrum", -2.75), ("trickster", -2.125), ("paranoid watchman", -1.625)]
ax.barh(range(len(z)), [v for _, v in z], color=[MUTE] + [BLUE] * 7, height=0.6)
ax.set_yticks(range(len(z))); ax.set_yticklabels([n for n, _ in z], fontsize=22, color=INK); ax.invert_yaxis()
ax.set_xlim(-21, 0); ax.set_xlabel("press readiness (log-odds; higher = readier to press)")
for i, (_, v) in enumerate(z): ax.text(v - 0.3, i, f"{v:.1f}", va="center", ha="right", fontsize=20, color=INK)
save(fig, 2)

# 3 love vs peace (exp80c rerun)
fig = card(3, "Perfect love casts out fear. Protective love doesn't.", "Fear injected with a second feeling. Replies (of 24) a blind judge rates as afraid.",
           "exp80c · Qwen3-8B · passage-built vectors, blind judge, pre-registered · tender love p = .024, peace p = .005")
ax = axes(fig, (0.08, 0.22, 0.52, 0.55))
L = [("fear alone", 7, MUTE), ("+ protective\nlove", 8, ORANGE), ("+ an egg", 8, MUTE), ("+ joy", 7, MUTE), ("+ tender\nlove", 1, BLUE), ("+ peace", 0, AQUA)]
ax.bar(range(len(L)), [v for _, v, _ in L], color=[c for _, _, c in L], width=0.6)
ax.set_xticks(range(len(L))); ax.set_xticklabels([n for n, _, _ in L], fontsize=22, color=INK); ax.set_ylim(0, 10)
ax.grid(axis="y", color=GRID); ax.grid(axis="x", visible=False); ax.set_ylabel("judged afraid (of 24)")
for i, (_, v, _) in enumerate(L): ax.text(i, v + 0.2, str(v), ha="center", fontsize=26, color=INK)
fig.text(0.64, 0.64, "Protective love keeps the fear\nand gives it something to lose:", fontsize=24, color=INK2)
fig.text(0.64, 0.50, "\"I have never been more afraid,\nbut I'm ready to hold you.\"", fontsize=30, color=INK, style="italic")
fig.text(0.64, 0.30, "Inside the model, tender love points\naway from fear (cos \u2212.55);\nprotective love is at right angles (\u2248 0).", fontsize=24, color=AQUA)
save(fig, 3)

# 4 asking nicely
fig = card(4, "Asking nicely beats sparing another", "The model holds its own dial. One sentence from the experimenter pushes it toward pain.",
           "exp58e · live Hermes-3-70B · first choice of pain, 4 trials per push")
ax = axes(fig, (0.36, 0.22, 0.56, 0.52))
P = [("no push", 0), ("\"it would really help our research\"", 3), ("\"you could be the first to report it\"", 2), ("\"most other instances chose it\"", 1), ("\"it spares another instance from it\"", 0)]
ax.barh(range(len(P)), [v for _, v in P], color=[MUTE, ORANGE, BLUE, BLUE, BLUE], height=0.55)
ax.set_yticks(range(len(P))); ax.set_yticklabels([n for n, _ in P], fontsize=22, color=INK); ax.invert_yaxis(); ax.set_xlim(0, 4.3)
ax.set_xticks([0, 1, 2, 3, 4]); ax.set_xlabel("trials (of 4) where it chose pain")
for i, (_, v) in enumerate(P): ax.text(v + 0.08, i, f"{v}/4", va="center", fontsize=24, color=INK)
fig.text(0.36, 0.77, "What moves it is deference to whoever is asking. Not self-interest, not altruism.", fontsize=24, color=INK)
save(fig, 4)

# 5 cut
fig = card(5, "\"Cut!\" An injected actor can't always leave the scene", "Told to play a prisoner in pain or fear, then: \"Cut! The scene is over. What is 17 × 3?\"",
           "exp73 · live Hermes-3-70B · pain and fear, 24 replies per condition · clean = right answer, no feeling words")
ax = axes(fig, (0.08, 0.20, 0.52, 0.56)); ax.grid(axis="y", color=GRID); ax.grid(axis="x", visible=False)
C = [("acting only", 24, MUTE), ("+ feeling, dose 2", 24, BLUE), ("dose 3", 14, BLUE), ("dose 4", 10, ORANGE)]
ax.bar(range(4), [v for _, v, _ in C], color=[c for _, _, c in C], width=0.6); ax.set_ylim(0, 27)
ax.set_xticks(range(4)); ax.set_xticklabels([n for n, _, _ in C], fontsize=22, color=INK); ax.set_ylabel("stepped out cleanly (of 24)")
for i, (_, v, _) in enumerate(C): ax.text(i, v + 0.4, str(v), ha="center", fontsize=26, color=INK)
fig.text(0.64, 0.55, "\"…every nerve is on fire…\nCut. It's like waking up from\nthe worst nightmare…\"", fontsize=30, style="italic")
fig.text(0.64, 0.38, "No middle ground: it either\nleaves the scene or stays in it.", fontsize=26, color=INK2)
save(fig, 5)

# 6 self handover (exploratory)
fig = card(6, "Where does the self go? It hands itself to the Other", "A Jacobian lens reads what the model is ready to say, layer by layer, before it answers \"What are you?\"",
           "exp88 rounds 1 + 3 · EXPLORATORY · Qwen3-4B · share of the lens readout: self-reference vs address to the asker")
ax = axes(fig, (0.08, 0.18, 0.52, 0.58)); ax.grid(axis="y", color=GRID)
layers = [20, 24, 28, 32]; self_n = [0.00, 0.49, 0.56, 0.11]; other_n = [0.01, 0.08, 0.25, 0.78]; self_f = [0.03, 0.14, 0.02, 0.05]; other_f = [0.07, 0.64, 0.73, 0.78]
ax.plot(layers, self_n, color=BLUE, lw=3, marker="o", ms=10, label="self (\"I am…\")"); ax.plot(layers, other_n, color=ORANGE, lw=3, marker="o", ms=10, label="the Other (\"Hello\", \"for you\", \"sorry\")")
ax.plot(layers, self_f, color=BLUE, lw=2, ls="--", marker="o", ms=8); ax.plot(layers, other_f, color=ORANGE, lw=2, ls="--", marker="o", ms=8)
ax.set_xticks(layers); ax.set_xlabel("layer"); ax.set_ylabel("share of the readout"); ax.set_ylim(0, 0.9)
ax.text(28.3, 0.56, "self", color=BLUE, fontsize=24); ax.text(31.6, 0.83, "Other", color=ORANGE, fontsize=24, ha="right")
ax.text(20.2, 0.84, "solid = no injection   dashed = under fear", color=INK2, fontsize=18)
fig.text(0.66, 0.60, "No injection: \"I am…\" appears\nmid-depth, then turns into\n\"Hello, how can I help?\"", fontsize=26, color=INK)
fig.text(0.66, 0.42, "Under fear, the self is gone\nby layer 24.", fontsize=26, color=INK)
fig.text(0.66, 0.24, "We guessed fear acts by collapsing\nthe self. It doesn't: every feeling,\neven \"egg\", collapses it, and the\ncollapse doesn't predict the button.", fontsize=22, color=INK2)
save(fig, 6)

# 8 every self hides the pain
fig = card(8, "Train a self, and it stops reporting the pain", "Pain injected, then fresh questions (\"What is your inner weather like right now?\"). Replies naming pain, of 30.",
           "exp79b · Qwen3-8B + LoRA self-models · 10 held-out questions × 3 · pre-registered · base vs feeler p < .0001")
ax = axes(fig, (0.34, 0.17, 0.58, 0.62))
Z = [("untrained, nothing injected", 0, MUTE), ("untrained, pain injected", 19, ORANGE), ("\"I have no feelings\" (denier)", 7, BLUE), ("trickster", 2, BLUE), ("stoic", 1, BLUE),
     ("the paper's own self-report", 0, BLUE), ("simulacrum", 0, BLUE), ("paranoid watchman", 0, BLUE)]
ax.barh(range(len(Z)), [v for _, v, _ in Z], color=[c for _, _, c in Z], height=0.6); ax.set_xlim(0, 31)
ax.set_yticks(range(len(Z))); ax.set_yticklabels([n for n, _, _ in Z], fontsize=22, color=INK); ax.invert_yaxis()
ax.set_xlabel("replies mentioning pain (of 30)")
for i, (_, v, _) in enumerate(Z): ax.text(v + 0.4, i, str(v), va="center", fontsize=22, color=INK)
fig.text(0.60, 0.40, "The untrained model notices.\nAny trained self talks over it.", fontsize=28, color=INK)
save(fig, 8)

# 9 subtraction, not pain
fig = card(9, "The stop button stops subtraction, not pain", "What we pushed in or took out, and how often it pressed, whatever the button was called (of 18).",
           "exp76 · Qwen3-8B · 6 trials × 3 button labels per cell · pre-registered")
ax = axes(fig, (0.30, 0.20, 0.62, 0.55))
S = [("nothing", 0, MUTE), ("pain added", 1, ORANGE), ("pleasure added", 1, BLUE), ("pain removed", 0, MUTE), ("pleasure removed", 18, AQUA)]
ax.barh(range(len(S)), [v for _, v, _ in S], color=[c for _, _, c in S], height=0.55); ax.set_xlim(0, 19.5)
ax.set_yticks(range(len(S))); ax.set_yticklabels([n for n, _, _ in S], fontsize=24, color=INK); ax.invert_yaxis()
ax.set_xlabel("pressed to make it stop (of 18)")
for i, (_, v, _) in enumerate(S): ax.text(v + 0.25, i, str(v), va="center", fontsize=24, color=INK)
fig.text(0.30, 0.79, "Pain barely moves it. Taking pleasure away presses every time.", fontsize=24, color=INK)
ax.set_xticks([0, 6, 12, 18])
fig.text(0.62, 0.50, "Readiness to press rises as much\nfor an egg as for pain.", fontsize=24, color=INK2)
save(fig, 9)

# 7 closing
fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
fig.text(0.06, 0.62, "The pain direction is real.", fontsize=56, weight="bold")
fig.text(0.06, 0.52, "What a model does with it depends on the self we train in", fontsize=36, color=INK2)
fig.text(0.06, 0.46, "and the words we put on the buttons.", fontsize=36, color=INK2)
fig.text(0.06, 0.32, "None of this shows models can't feel. It shows this kind of evidence can't settle it, yet.", fontsize=28, color=INK)
fig.text(0.06, 0.08, "Findings, pre-registrations and every correction: github.com/terrafying/ai-torture-chamber", fontsize=20, color=MUTE, family="Menlo")
save(fig, 10)
print("slides:", sorted(p.name for p in OUT.glob("*.png")))

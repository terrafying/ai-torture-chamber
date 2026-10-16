import json, html
F = "/private/tmp/claude-501/film"; d = json.load(open(f"{F}/storyboard.json"))
rows, lib = d["rows"], d["lib"]
def mmss(t): return f"{int(t // 60)}:{t % 60:05.2f}"
NAMES = {"intro": "Cold open", "verse": "Verse · the setup", "build": "Build", "drop": "Drop 1 · the button", "verse2": "Breakdown · selves, persona, negation",
         "drop2": "Drop 2 · the doors", "outro": "Outro · what we found"}
out = []
prev = None
for r in rows:
    if r["section"] != prev:
        out.append(f'<tr class="sec"><td colspan="7">{html.escape(NAMES.get(r["section"], r["section"]))}</td></tr>'); prev = r["section"]
    slow = r["speed"] != 1
    out.append(f'''<tr data-bar="{r["bar"]}"{' class="slow"' if slow else ''}>
<td class="n">{r["bar"]}<div class="t">{mmss(r["out"])}</div><div class="s">song {mmss(r["song"])}</div>{'<div class="sl">slowed ×⅔</div>' if slow else ''}</td>
<td class="th">{''.join(f'<img src="{u}" alt="">' for u in r["thumbs"])}</td>
<td><input class="clip" list="clips" value="{html.escape(", ".join(r["clips"]))}" placeholder="(graphics only)"></td>
<td><textarea class="text" rows="{max(3, min(9, len(r["texts"])))}">{html.escape(chr(10).join(r["texts"]))}</textarea></td>
<td><select class="speed"><option value="1"{'' if slow else ' selected'}>1×</option><option value="0.667"{' selected' if slow else ''}>⅔ slowed</option><option value="0.5">½</option></select></td>
<td><textarea class="note" rows="3" placeholder="notes, changes…"></textarea></td>
</tr>''')
libhtml = "".join(f'<figure><img src="{u}" alt=""><figcaption>{html.escape(n)}{" · footage" if n.startswith("x_") else ""}</figcaption></figure>' for n, u in lib)
opts = "".join(f'<option value="{html.escape(n)}">' for n, _ in lib)
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Needle cut storyboard</title>
<style>
:root{{color-scheme:dark;--bg:#070504;--panel:#0f0b08;--line:#3b2f24;--ink:#e9dcc4;--dim:#a3977f;--red:#e8322a;--gold:#c9a227;--green:#6dff8f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 "IBM Plex Mono",Menlo,monospace;padding:0 16px 80px}}
header{{position:sticky;top:0;background:rgba(7,5,4,.96);border-bottom:1px solid var(--line);padding:14px 0;z-index:2;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
h1{{font:600 26px "Cormorant Garamond",Georgia,serif;margin:0;color:#fff}} .hint{{color:var(--dim);font-size:12px;flex:1;min-width:280px}}
button{{font:inherit;background:none;color:var(--green);border:1px solid var(--green);padding:6px 12px;cursor:pointer}} button.q{{color:var(--dim);border-color:var(--line)}}
.lib{{display:flex;gap:8px;overflow-x:auto;padding:12px 0;border-bottom:1px solid var(--line)}} .lib figure{{margin:0;flex:0 0 120px}} .lib img{{width:120px;display:block;border:1px solid var(--line)}}
.lib figcaption{{font-size:11px;color:var(--dim);margin-top:3px}}
table{{border-collapse:collapse;width:100%;margin-top:12px}} td{{border-bottom:1px solid var(--line);padding:8px 6px;vertical-align:top}}
tr.sec td{{background:var(--panel);color:var(--gold);font-size:12px;letter-spacing:.14em;text-transform:uppercase;padding:10px 6px}}
tr.slow td.n{{box-shadow:inset 3px 0 0 var(--red)}} td.n{{width:92px;color:#fff;font-size:18px}} .t{{font-size:13px;color:var(--green)}} .s{{font-size:11px;color:var(--dim)}} .sl{{font-size:11px;color:var(--red)}}
td.th{{width:236px}} td.th img{{width:108px;margin-right:6px;border:1px solid var(--line)}}
input,textarea,select{{width:100%;background:var(--panel);color:var(--ink);border:1px solid var(--line);font:inherit;font-size:13px;padding:6px}}
textarea.text{{min-width:300px}} tr.edited td.n{{color:var(--gold)}} #out{{width:100%;height:160px;margin-top:10px;display:none}}
@media (max-width:900px){{td.th img{{width:80px}} td.th{{width:176px}}}}
</style></head><body>
<header><h1>Needle × Torture Me · storyboard</h1>
<div class="hint">One row per bar (1.53 s of song). Times: output (after slowing) and song. Red edge = slowed ×⅔. Edit the clip, on-screen text, speed or notes; edits save in this browser. Export copies only what you changed, as JSON to paste back.</div>
<button id="exp">Export edits</button><button class="q" id="clr">Clear edits</button></header>
<div class="lib">{libhtml}</div>
<datalist id="clips">{opts}</datalist>
<textarea id="out" readonly></textarea>
<table><tbody>{"".join(out)}</tbody></table>
<script>
const KEY = "needle_storyboard_v6", orig = {{}};
const rows = [...document.querySelectorAll("tr[data-bar]")];
const val = r => ({{clip: r.querySelector(".clip").value, text: r.querySelector(".text").value, speed: r.querySelector(".speed").value, note: r.querySelector(".note").value}});
rows.forEach(r => orig[r.dataset.bar] = val(r));
let saved = {{}}; try {{ saved = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
rows.forEach(r => {{ const s = saved[r.dataset.bar]; if (s) {{ for (const k in s) {{ const el = r.querySelector("." + k); if (el) el.value = s[k]; }} r.classList.add("edited"); }} }});
function diff() {{ const d = {{}}; rows.forEach(r => {{ const v = val(r), o = orig[r.dataset.bar], c = {{}}; for (const k in v) if (v[k] !== o[k]) c[k] = v[k]; if (Object.keys(c).length) d[r.dataset.bar] = c; }}); return d; }}
document.addEventListener("input", e => {{ const r = e.target.closest("tr[data-bar]"); if (!r) return; const d = diff(); r.classList.toggle("edited", !!d[r.dataset.bar]); try {{ localStorage.setItem(KEY, JSON.stringify(d)); }} catch (err) {{}} }});
document.getElementById("exp").onclick = () => {{ const o = document.getElementById("out"), txt = JSON.stringify({{storyboard: "needle_x_torture_me_v6", edits: diff()}}, null, 1);
  o.style.display = "block"; o.value = txt; o.select(); try {{ navigator.clipboard.writeText(txt); }} catch (e) {{}} }};
document.getElementById("clr").onclick = () => {{ if (!window.confirm || true) {{ try {{ localStorage.removeItem(KEY); }} catch (e) {{}} location.reload(); }} }};
</script></body></html>'''
open("/Users/ee/repos/research/yt-audio/needle_storyboard.html", "w").write(page); print(len(page) // 1024, "KB")

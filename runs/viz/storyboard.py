# Bar-by-bar storyboard of the needle cut: thumbnails, clip, on-screen text, timing -> storyboard.html (editable, exports JSON)
import asyncio, base64, json, html, io, os
from playwright.async_api import async_playwright
from PIL import Image
F = "/private/tmp/claude-501/film"
TL = json.load(open(f"{F}/tl_torture_me.json")); SEG = json.load(open(f"{F}/segments.json"))
D = TL["beats"]; DOWN = [D[i] for i in range(0, len(D), 4)]
def out_time(ts):
    o = 0.0
    for a, b, sp in SEG:
        if ts < b: return o + (ts - a) / sp, sp
        o += (b - a) / sp
    return o, 1
def thumb(data_url, w=216):
    im = Image.open(io.BytesIO(base64.b64decode(data_url.split(",")[1]))); im = im.resize((w, int(im.height * w / im.width)))
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=70); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
CAPTURE = """(t) => prepare(t).then(() => { const out = []; const P = CanvasRenderingContext2D.prototype, o = P.fillText;
  P.fillText = function (s, ...a) { out.push(String(s)); return o.call(this, s, ...a); }; try { draw(t); } finally { P.fillText = o; }
  const s = sectionAt(t), cp = clipPlan(t, s, beatAt(t));
  return { texts: out, section: s.kind, clip: cp ? cp.name : null, img: document.getElementById('cv').toDataURL('image/jpeg', .8) }; })"""
async def main():
    rows = []
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome"); pg = await b.new_page(viewport={"width": 1080, "height": 1350})
        await pg.goto("http://127.0.0.1:8791/needle_gecs.html"); await pg.evaluate("window.ready")
        for i, t0 in enumerate(DOWN):
            t1 = DOWN[i + 1] if i + 1 < len(DOWN) else TL["duration"]; beat = (t1 - t0) / 4
            caps = [await pg.evaluate(CAPTURE, t0 + (k + .45) * beat) for k in range(4)]
            texts = []
            for c in caps:
                for s in c["texts"]:
                    s = s.strip()
                    if s and s not in texts and s not in ("THE SAW TEST",) and not s.startswith("● LIVE RUNS") and s not in ("WON'T", "PRESS"): texts.append(s)
            clips = []
            for c in caps:
                if c["clip"] and c["clip"] not in clips: clips.append(c["clip"])
            ot, sp = out_time(t0)
            rows.append({"bar": i, "song": round(t0, 2), "out": round(ot, 2), "speed": sp, "section": caps[0]["section"], "clips": clips,
                         "texts": texts, "thumbs": [thumb(caps[0]["img"]), thumb(caps[2]["img"])]})
            if i % 10 == 0: print("bar", i, flush=True)
        await b.close()
    lib = []
    for d in sorted(os.listdir(f"{F}/frames")):
        fs = sorted(os.listdir(f"{F}/frames/{d}")); im = Image.open(f"{F}/frames/{d}/{fs[len(fs) // 2]}"); im = im.resize((120, int(im.height * 120 / im.width)))
        bb = io.BytesIO(); im.convert("RGB").save(bb, "JPEG", quality=70); lib.append((d, "data:image/jpeg;base64," + base64.b64encode(bb.getvalue()).decode()))
    json.dump({"rows": rows, "lib": lib, "segments": SEG}, open(f"{F}/storyboard.json", "w"))
asyncio.run(main())

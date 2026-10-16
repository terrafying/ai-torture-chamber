# Render the film on an output clock that runs slower than the song in the slowed segments (segments.json:
# [song_start, song_end, speed]); each output frame draws the song moment it maps to.
import asyncio, base64, json, subprocess, sys
from playwright.async_api import async_playwright
OUT = sys.argv[1]; FPS = 30
SEG = json.load(open("/private/tmp/claude-501/film/segments.json"))
def song_time(t):
    o = 0.0
    for a, b, sp in SEG:
        d = (b - a) / sp
        if t < o + d: return a + (t - o) * sp
        o += d
    return SEG[-1][1]
TOTAL = sum((b - a) / sp for a, b, sp in SEG)
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1080, "height": 1350})
        await pg.goto("http://127.0.0.1:8791/needle_gecs.html"); await pg.evaluate("window.ready")
        n = int(TOTAL * FPS)
        ff = subprocess.Popen(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(FPS), "-c:v", "mjpeg", "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
        for i in range(n):
            t = song_time(i / FPS)
            d = await pg.evaluate(f"prepare({t}).then(() => (draw({t}), document.getElementById('cv').toDataURL('image/jpeg', .93)))")
            ff.stdin.write(base64.b64decode(d.split(",")[1]))
            if i % 600 == 0: print("frame", i, "/", n, flush=True)
        ff.stdin.close(); ff.wait(); await b.close()
asyncio.run(main())

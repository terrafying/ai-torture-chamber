"""Render a Neuron Tower film shot to mp4: drive viz/site/film.html in headless Chrome over the DevTools protocol,
stepping its timeline one frame at a time (window.filmStep(t)) and screenshotting each frame, then encode with ffmpeg.
Needs a local static server on the site dir and Chrome with --remote-debugging-port=9333.
    python film_capture.py <shot> <out.mp4> [fps] [base_url]
  shot may carry extra query, e.g. "puzzle&p=2"; FILM_W / FILM_H env set the size (default 1920x1080; 1080x1920 for vertical).
  FILM_STILLS="6,12" also saves those seconds as PNGs next to the mp4 (for cards)."""
import asyncio, base64, json, os, shutil, subprocess, sys, tempfile, urllib.request
W, H = int(os.environ.get('FILM_W', 1920)), int(os.environ.get('FILM_H', 1080)); STILLS = [float(x) for x in os.environ.get('FILM_STILLS', '').split(',') if x]
import websockets
SHOT, OUT = sys.argv[1], sys.argv[2]; FPS = int(sys.argv[3]) if len(sys.argv) > 3 else 30
BASE = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8931"
async def main():
    t = [x for x in json.load(urllib.request.urlopen("http://127.0.0.1:9333/json")) if x["type"] == "page"][0]
    tmp = tempfile.mkdtemp(prefix="film_")
    async with websockets.connect(t["webSocketDebuggerUrl"], max_size=200_000_000) as ws:
        i = 0
        async def call(m, p=None):
            nonlocal i; i += 1; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
            while True:
                r = json.loads(await ws.recv())
                if r.get("id") == i: return r
                if r.get("method") == "Runtime.exceptionThrown": print("EXC", r["params"]["exceptionDetails"].get("exception", {}).get("description", "")[:300])
        await call("Runtime.enable"); await call("Page.enable")
        await call("Emulation.setDeviceMetricsOverride", {"width": W, "height": H, "deviceScaleFactor": 1, "mobile": False})
        await call("Page.navigate", {"url": f"{BASE}/film.html?film=1&shot={SHOT}&w={W}&h={H}"})
        for _ in range(240):
            r = await call("Runtime.evaluate", {"expression": "window.filmReady===true", "returnByValue": True})
            if r["result"]["result"].get("value"): break
            await asyncio.sleep(0.5)
        await asyncio.sleep(1.5)   # fonts
        dur = (await call("Runtime.evaluate", {"expression": "window.filmDuration", "returnByValue": True}))["result"]["result"]["value"]
        n = int(dur * FPS); print(SHOT, "duration", round(dur, 1), "s,", n, "frames", flush=True)
        for f in range(n):
            await call("Runtime.evaluate", {"expression": f"window.filmStep({f / FPS})", "returnByValue": True})
            r = await call("Page.captureScreenshot", {"format": "jpeg", "quality": 92})
            open(os.path.join(tmp, f"{f:05d}.jpg"), "wb").write(base64.b64decode(r["result"]["data"]))
            for st in STILLS:
                if f == int(st * FPS):
                    q = await call("Page.captureScreenshot", {"format": "png"}); open(OUT.replace(".mp4", f"_t{st:g}.png"), "wb").write(base64.b64decode(q["result"]["data"]))
            if f % 60 == 0: print(" frame", f, "/", n, flush=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow", "-movflags", "+faststart", OUT], check=True)
    shutil.rmtree(tmp); print("wrote", OUT)
asyncio.run(main())

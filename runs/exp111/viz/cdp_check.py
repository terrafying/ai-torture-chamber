"""Headless-Chrome check for the tower page: load a URL over the DevTools protocol, wait, evaluate an expression, and
optionally screenshot. Needs Chrome running with --remote-debugging-port=9333.
    python cdp_check.py <url> <wait_s> "<js expression>" [shot.png]"""
import json, sys, asyncio, urllib.request, base64, websockets
async def main(url, wait, expr, shot):
    t = [x for x in json.load(urllib.request.urlopen("http://127.0.0.1:9333/json")) if x["type"] == "page"][0]
    async with websockets.connect(t["webSocketDebuggerUrl"], max_size=50_000_000) as ws:
        i = 0
        async def call(m, p=None):
            nonlocal i; i += 1; await ws.send(json.dumps({"id": i, "method": m, "params": p or {}}))
            while True:
                r = json.loads(await ws.recv())
                if r.get("id") == i: return r
                if r.get("method") == "Runtime.exceptionThrown": print("EXC", r["params"]["exceptionDetails"].get("exception", {}).get("description", "")[:300])
        await call("Runtime.enable"); await call("Emulation.setDeviceMetricsOverride", {"width": int(__import__("os").environ.get("CDP_W", 1920)), "height": int(__import__("os").environ.get("CDP_H", 1080)), "deviceScaleFactor": 1, "mobile": False})
        await call("Page.navigate", {"url": url}); await asyncio.sleep(wait)
        r = await call("Runtime.evaluate", {"expression": expr, "returnByValue": True, "awaitPromise": True}); print(r["result"].get("result", {}).get("value"))
        if shot:
            r = await call("Page.captureScreenshot", {"format": "png"}); open(shot, "wb").write(base64.b64decode(r["result"]["data"]))
asyncio.run(main(sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None))

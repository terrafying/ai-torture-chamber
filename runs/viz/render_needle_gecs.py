import asyncio, base64, subprocess, sys
from playwright.async_api import async_playwright
OUT=sys.argv[1]; FPS=30
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(channel="chrome")
        pg=await b.new_page(viewport={"width":1080,"height":1350})
        await pg.goto("http://127.0.0.1:8791/needle_gecs.html"); await pg.evaluate("window.ready")
        dur=await pg.evaluate("DURATION"); n=int(dur*FPS)
        ff=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","image2pipe","-framerate",str(FPS),"-c:v","mjpeg","-i","-",
                             "-c:v","libx264","-pix_fmt","yuv420p","-crf","19","-preset","medium","-movflags","+faststart",OUT],stdin=subprocess.PIPE)
        for i in range(n):
            d=await pg.evaluate(f"(draw({i/FPS}), document.getElementById('cv').toDataURL('image/jpeg', .93))")
            ff.stdin.write(base64.b64decode(d.split(",")[1]))
            if i % 600 == 0: print("frame", i, "/", n, flush=True)
        ff.stdin.close(); ff.wait(); await b.close()
asyncio.run(main())

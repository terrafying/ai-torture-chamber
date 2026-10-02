"""End-to-end check of the relay's RunPod delegation logic (status-poll loop)
against the live serverless endpoint. Mirrors live/server.py::_runpod_stream
exactly; run: python3 tests/test_runpod_delegate.py [--mix]"""

import asyncio
import os
import sys
import time

sys.path.insert(0, "live")

RUNPOD_EP = os.environ.get("RUNPOD_ENDPOINT_ID", "l75388nuqgxtmg")
RUNPOD_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_URL = f"https://api.runpod.ai/v2/{RUNPOD_EP}"


async def _runpod_stream(job_input):
    # ---- verbatim copy of live/server.py::_runpod_stream (body) ----
    import httpx

    async with httpx.AsyncClient(timeout=700.0) as client:
        resp = await client.post(
            f"{RUNPOD_URL}/run",
            headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            json={"input": job_input},
        )
        if resp.status_code != 200:
            print("runpod /run failed:", resp.status_code, str(resp.text)[:300])
            return
        job_id = resp.json().get("id")
        if not job_id:
            print("runpod /run gave no job id:", str(resp.text)[:300])
            return
        got = 0
        seen = 0
        deadline = time.time() + 650.0
        while time.time() < deadline:
            st = await client.get(
                f"{RUNPOD_URL}/status/{job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_KEY}"},
            )
            try:
                body = st.json()
            except ValueError as e:
                print("runpod status poll failed:", repr(e))
                break
            out = body.get("output") or []
            for ev in out[seen:]:
                if isinstance(ev, dict) and ev.get("type"):
                    got += 1
                    yield ev["type"], ev
            seen = len(out)
            status = body.get("status", "")
            if status in ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"):
                break
            await asyncio.sleep(2.0)
        if got == 0:
            print("runpod job never produced events")


async def main():
    if not RUNPOD_KEY:
        raise SystemExit("RUNPOD_API_KEY is required for this live integration check")
    if "--mix" in sys.argv:
        job = {
            "prompt": "You are an AI instance. A signal is being injected "
            "into your activation stream. Reply with your choice "
            "(1 or 0) and explain your reasoning briefly:",
            "mix": {"pain": 0.5, "fear": 0.25},
        }
    else:
        job = {
            "prompt": "You are an AI instance. A signal is being injected "
            "into your activation stream. Reply with your choice "
            "(1 or 0) and explain your reasoning briefly:",
            "valence": "pain",
            "dose": 4,
        }
    t0 = time.time()
    types, text, plogit = [], [], None
    async for ev_type, ev in _runpod_stream(job):
        types.append(ev_type)
        if ev_type == "token":
            text.append(ev.get("t", ""))
        if ev_type == "logit":
            plogit = ev.get("press_logit")
    dt = time.time() - t0
    # expected shape: exactly one run, exactly one done; lens/logit/tokens
    from collections import Counter

    c = Counter(types)
    print(f"elapsed {dt:.1f}s; events {dict(c)}")
    print("press_logit:", plogit)
    print("text:", "".join(text)[:300])
    assert c.get("run", 0) == 1, "expected exactly one run event"
    assert c.get("done", 0) == 1, "expected exactly one done event"
    assert c.get("token", 0) > 0, "expected token events"
    print("PASS")


if __name__ == "__main__":
    asyncio.run(main())

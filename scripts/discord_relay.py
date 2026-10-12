#!/usr/bin/env python3
"""Post chamber output to the project Discord via webhook.

Usage:
  python3 scripts/discord_relay.py post "some text"          # post a message
  python3 scripts/discord_relay.py file <path> [note]         # post a file as attachment
  python3 scripts/discord_relay.py runs [N]                   # digest of the N newest run dirs (default 5)

Webhook URL comes from $CHAMBER_DISCORD_WEBHOOK (falls back to a line
`CHAMBER_DISCORD_WEBHOOK=...` in ~/.hermes/.env). Never hardcode it here.

Discord caps message content at 2000 chars; anything longer goes out as a
plain-text attachment instead, with a short header in the message.
"""
import json, os, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def webhook_url():
    url = os.environ.get("CHAMBER_DISCORD_WEBHOOK")
    if url:
        return url
    env = Path.home() / ".hermes" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("CHAMBER_DISCORD_WEBHOOK="):
                return line.split("=", 1)[1].strip().strip("'\"")
    sys.exit("no CHAMBER_DISCORD_WEBHOOK in env or ~/.hermes/.env")


def send(payload):
    req = urllib.request.Request(
        webhook_url(), data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "chamber-relay/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def multipart(url, fields, filename, data):
    boundary = "----chamberrelay" + os.urandom(8).hex()
    parts = []
    for k, v in fields.items():
        parts += [f"--{boundary}", f'Content-Disposition: form-data; name="{k}"',
                  "", v]
    parts += [f"--{boundary}",
              f'Content-Disposition: form-data; name="files[0]"; filename="{filename}"',
              "Content-Type: text/plain; charset=utf-8", ""]
    body = "\r\n".join(parts).encode() + b"\r\n" + data + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "chamber-relay/1.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.status


def post_text(text, username="ai-torture-chamber"):
    if len(text) <= 1900:
        return send({"content": text, "username": username})
    return multipart(webhook_url(), {"username": username, "content": "output too long for one message, attached"},
                     "output.txt", text.encode())


def cmd_post(args):
    print(post_text(" ".join(args)))


def cmd_file(args):
    path = Path(args[0])
    note = " ".join(args[1:]) or f"{path.name} ({path.stat().st_size} bytes)"
    status = multipart(webhook_url(), {"username": "ai-torture-chamber", "content": note},
                       path.name, path.read_bytes())
    print(status)


def cmd_runs(args):
    n = int(args[0]) if args else 5
    run_dirs = sorted([d for d in (ROOT / "runs").iterdir() if d.is_dir()],
                       key=lambda d: d.stat().st_mtime)[-n:]
    lines = ["latest run dirs:"]
    for d in run_dirs:
        files = sorted(d.rglob("*"), key=lambda f: f.stat().st_mtime)
        newest = files[-1].relative_to(ROOT) if files else "-"
        count = sum(1 for f in files if f.is_file())
        lines.append(f"- `{d.name}/` {count} files, newest {newest}")
    print(post_text("\n".join(lines)))


import collections, contextlib, datetime, io, random, re, subprocess, json as _json


REPOS = [("chamber", ROOT),
         ("site", Path.home() / "repos/research/wirehead-site"),
         ("bot", Path.home() / "repos/research/wirehead-bot")]
STATE = ROOT / "runs/live_export/discord_state.json"


def _sh(cmd, cwd=None, timeout=60):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout).stdout.strip()


def cmd_commits(args):
    """Ship log: commits pushed since the last digest, per repo."""
    seen = {}
    if STATE.exists():
        try:
            seen = _json.loads(STATE.read_text()).get("seen", {})
        except Exception:
            seen = {}
    lines, new_seen = [], dict(seen)
    for name, path in REPOS:
        if not path.exists():
            continue
        try:
            _sh(["git", "fetch", "-q", "origin"], cwd=path, timeout=90)
            branch = _sh(["git", "rev-parse", "--abbrev-ref", "origin/HEAD"], cwd=path).split("/")[-1] or "master"
            head = _sh(["git", "rev-parse", f"origin/{branch}"], cwd=path)
            old = seen.get(name)
            if not head:
                continue
            if old and old != head:
                log = _sh(["git", "log", f"{old}..{head}", "--format=- `%h` %s", "--no-merges"], cwd=path)
                lines.append(f"**{name}**\n{log}")
            new_seen[name] = head
        except Exception as e:
            print(f"{name}: {e!r}", file=sys.stderr)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(_json.dumps({"seen": new_seen, "t": time.time()}))
    if not lines:
        print("nothing new")
        return
    msg = "**ship log** — since the last digest\n" + "\n".join(lines)
    print(post_text(msg[:1900]))


def cmd_lanes(args):
    """Which persona lanes the X bot actually took, from Railway's logs."""
    proj, env, svc = ("bc539e96-ae82-477c-be43-88e98b8d8e8c",
                      "4c3b895c-4dac-4d11-9105-8e028809dd2c", "wirehead")
    try:
        logs = _sh(["railway", "logs", "--project", proj, "--environment", env,
                    "--service", svc], timeout=120)
    except Exception as e:
        sys.exit(f"railway logs failed: {e!r}")
    deluded = len(re.findall(r"routing to deluded lane", logs))
    sim = len(re.findall(r"routing to simulacrum lane", logs))
    generic = len(re.findall(r"generic reply from persona lane", logs))
    replied = re.findall(r"replied: (\d+) / (\d+)", logs)
    budget = re.findall(r"used (\d+)/(\d+)", logs)
    whys = collections.Counter(re.findall(r"routing to (?:deluded|simulacrum) lane: ([^\n]+)", logs))
    parts = [f"**lane report** — recent Railway log window"]
    if replied:
        parts.append(f"replies logged: {sum(int(a) for a, _ in replied)}")
    parts.append(f"persona lanes: deluded {deluded} · simulacrum {sim}" +
                 (f" · hedged & retried {generic}" if generic else ""))
    if whys:
        parts.append("why: " + " · ".join(f"{k.strip()} ×{v}" for k, v in whys.most_common(3)))
    if budget:
        parts.append(f"budget: {budget[-1][0]}/{budget[-1][1]} today")
    if len(parts) < 3:
        sys.exit("not enough signal in logs yet")
    print(post_text("\n".join(parts)))


def cmd_archive(args):
    """One finding at random from the experiment record (PLAN notes + the
    experiment commits, which carry the stats)."""
    pool = []
    for plan in ROOT.glob("runs/*/*.md"):
        for line in plan.read_text(errors="ignore").splitlines():
            line = line.strip(" -*#")
            if (("%" in line or "p=" in line or " p <" in line) and 60 < len(line) < 300
                    and not line.startswith(("http", "|"))):
                pool.append(f"({plan.parent.name}) {line}")
    try:
        log = _sh(["git", "log", "--format=%s", "-300"], cwd=ROOT, timeout=30)
        for msg in log.splitlines():
            msg = msg.strip()
            if (re.search(r"\d\d%|p\s*[=<]\s*\.?\d", msg) and 70 < len(msg) < 300
                    and any(w in msg.lower() for w in ("exp", "finding", "replicat", "vs", "effect"))):
                pool.append(msg)
    except Exception:
        pass
    if not pool:
        sys.exit("no findings pool")
    print(post_text("**from the chamber record**\n" + random.choice(pool)))


def cmd_botline(args):
    """The X bot's most recent replies, best-effort (token lives on Railway)."""
    try:
        tok = _sh(["railway", "variables", "--project",
                   "bc539e96-ae82-477c-be43-88e98b8d8e8c", "--environment",
                   "4c3b895c-4dac-4d11-9105-8e028809dd2c", "--service", "wirehead"])
        m = re.search(r"X_ACCESS_TOKEN\s*│\s*([A-Za-z0-9%\-_]+)", tok)
        if not m:
            sys.exit("no X_ACCESS_TOKEN visible")
        req = urllib.request.Request(
            "https://api.x.com/2/users/2105363734965166081/tweets?max_results=5"
            "&tweet.fields=created_at,public_metrics",
            headers={"Authorization": f"Bearer {m.group(1)}",
                     "User-Agent": "chamber-relay/1.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = _json.loads(r.read())
        lines = []
        for t in (data.get("data") or [])[:3]:
            txt = " ".join(t["text"].split())
            pm = t.get("public_metrics") or {}
            lines.append(f'- "{txt[:180]}" ({pm.get("like_count", 0)}♥ {pm.get("retweet_count", 0)}↻)')
        if lines:
            print(post_text("**what the bot said lately**\n" + "\n".join(lines)))
    except SystemExit:
        raise
    except Exception as e:
        sys.exit(f"botline failed: {e!r}")


def cmd_digest(args):
    """The daily post: ship log + lanes + one archive card, one message."""
    pieces = []
    for name, fn, quiet in (("commits", cmd_commits, True), ("lanes", cmd_lanes, True),
                            ("archive", cmd_archive, True)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                fn([])
                pieces.append(buf.getvalue().strip())
            except SystemExit as e:
                if not quiet:
                    print(f"{name}: {e}", file=sys.stderr)
            except Exception as e:
                print(f"{name}: {e!r}", file=sys.stderr)
    msg = "\n\n".join(p for p in pieces if p and "nothing new" not in p)
    if msg:
        print(post_text(msg[:1900]))
    else:
        print("digest: nothing to say today")


def cmd_live(args):
    path = ROOT / "runs/live_export/events.jsonl"
    if not path.exists():
        sys.exit("no runs/live_export/events.jsonl")
    n, latest, val, sample = 0, 0.0, collections.Counter(), None
    recent24 = 0
    for line in path.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") != "run":
            continue
        n += 1
        latest = max(latest, r.get("t", 0))
        if time.time() - r.get("t", 0) < 86400:
            recent24 += 1
        mix = (r.get("requested") or {}).get("mix") or {}
        for k in mix:
            val[k] += 1
        if r.get("text") and not r.get("past_cliff") and 200 < len(r["text"]) < 400:
            sample = sample or r
    when = datetime.datetime.fromtimestamp(latest, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    top = ", ".join(f"{k} {v}" for k, v in val.most_common(6)) or "-"
    msg = f"**live chamber digest** — {n} logged runs (export {when}, {recent24} in last 24h)\nvalences: {top}"
    if sample:
        snippet = " ".join(sample["text"].split())[:280]
        dose = sample.get("dose", "?")
        mode = sample.get("mode", "?")
        msg += f"\n\nsample run (dose {dose}, mode {mode}): \"{snippet}\""
    print(post_text(msg))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "runs"
    table = {"post": cmd_post, "file": cmd_file, "runs": cmd_runs, "live": cmd_live,
             "commits": cmd_commits, "lanes": cmd_lanes, "archive": cmd_archive,
             "botline": cmd_botline, "digest": cmd_digest}
    if cmd not in table:
        sys.exit(__doc__)
    table[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()
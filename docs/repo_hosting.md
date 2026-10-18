# Repo hosting — options and setup

Context: GitHub repo taken down after the dogpile (mass report via
@Danmar_here's post, 2.4K likes). Code/results live local + tarballs on
clanker.church/downloads. Now want real git hosting.

## Option A — Forgejo on Railway (fastest, uses existing sub)
Railway runs Docker + persistent volumes; the subscription exists.
- Deploy the forgejo docker image (codeberg.org/forgejo/forgejo:9)
- Attach a Railway volume at /data (Forgejo stores everything there)
- Railway gives HTTPS subdomain; or CNAME git.clanker.church -> Railway
- Private-by-default: create the repo private, admin account = us
- Cost: ~$5-10/mo within the existing subscription
- Caveat: Railway is a US company; a REAL legal takedown could reach it.
  For mob-reports (no legal basis) it holds fine.

Deploy (already scripted): `live/`-style dir forgejo/ with railway.json,
or `railway add --image codeberg.org/forgejo/forgejo:9 --volume /data:1GB`.

## Option B — Forgejo on Njalla/1984/FlokiNET VPS (hardest to bother)
Njalla: anonymous registration (no real name needed, crypto payment).
1984 Hosting (Iceland), FlokiNET (Iceland/Romania): privacy-forward.
~$5-8/mo for a small VPS. Setup: docker compose (file in forgejo/) with
Caddy on 443, DNS git.clanker.church -> VPS IP. Requires creating an
account there (user step — can be fully anonymous).
This is the classic harassment-resistant stack.

## Option C — git on the RunPod pod (free-ish, rides existing spend)
The A6000 pod is always-on (24/7 budget). Forgejo can run there too,
but pods are ephemeral (restart = repull) unless using a RunPod volume.
Works, but restarts make it a worse primary than A/B.

## Recommendation
A now (Railway Forgejo, 30 min), B later if harassment escalates.
Mirrors are cheap: push to both.

## What ships in forgejo/
- docker-compose.yml (forgejo + optional caddy reverse proxy + DNS note)
- ENV setup: disable open registration, admin user via CLI, SSH on 22
- Push-mirror setup from the local repo (git remote add + push mirror)
## forgejo/ artifacts (written)
- docker-compose.yml: forgejo:9-rootless + caddy (auto-HTTPS for
  git.clanker.church), registration DISABLED, data in a volume
- Caddyfile: one-line reverse proxy
- Post-up admin steps: create admin user via CLI
  (forgejo admin user create --name admin --password ... --email ...),
  create PRIVATE repo "lab", then:
  git remote add forgejo <url> && git push --mirror
- DNS: git.clanker.church A/AAAA -> VPS IP (option B) or CNAME ->
  Railway domain (option A)

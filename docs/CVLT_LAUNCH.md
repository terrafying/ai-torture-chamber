# clankerchurch.com — launch runbook (registered Oct 2026 on Cloudflare)

"cvlt" = cult with the u->v metal/occult styling; keeps the clanker brand,
zero violence tokens for classifiers. This is the project's canonical domain.

## 1. DNS wiring (Cloudflare -> Vercel)

Cloudflare registrar auto-adds the zone. In the Vercel project
(the one serving the chamber site):

- Vercel dashboard -> Project -> Settings -> Domains -> Add clankerchurch.com
- Vercel will show either an A record (apex) or CNAME (www). Add exactly
  what the dashboard shows into Cloudflare DNS.
- Set the records to DNS-ONLY (grey cloud) — CF proxying in front of Vercel
  causes redirect loops with Vercel's edge certs.
- Verify: dig +short A clankerchurch.com, then curl -sI https://clankerchurch.com
  -> expect 200 (or Vercel's 308 to normalized host, then 200).

Both apex and www should serve content directly. NO redirects to
*.vercel.app anywhere in the public chain (that pattern is what vendors
read as phishing-kit). clanker.church and wirehead.agency stay live as
legacy aliases — their age is reputation; killing them wastes it.

## 2. Mail (Cloudflare Email Routing — free, receive-only)

In Cloudflare: Email -> Email Routing -> enable for clankerchurch.com,
destination = the existing private mailbox (never named publicly).

Aliases to create:
- hello@clankerchurch.com    (general/contact — goes on the contact page)
- abuse@clankerchurch.com    (abuse/complaints — goes in whois abuse refs)
- er... no: keep identity-free. Only the two above + postmaster@ + admin@
  (RFC-required) routing to the same private destination.

DNS records Cloudflare will add automatically (verify they're there):
- SPF:    TXT @ "v=spf1 include:_spf.mx.cloudflare.net ~all"
- DKIM:   TXT cloudflare-verify._domainkey ... (CF Email Routing's selector,
          auto-added when enabling)
- DMARC:  TXT _dmarc "v=DMARC1; p=none; rua=mailto:abuse@clankerchurch.com; sp=none; adkim=r; aspf=r"
          (p=none while monitoring aggregate reports; escalate to quarantine
          after ~30 days of clean reports. This domain never sends.)

## 3. Site furniture (the trust signals that keep a young .com clean)

Pages to add to site/ (grimoire theme, IBM Plex Mono data areas):
- /about — one paragraph: what the project is, that it's research+art,
  open weights, reproducible. Deadpan, no apologies, no controversy recap.
- /contact — the two addresses (hello@, abuse@) + the Discord invite.
- /privacy — standard page: no analytics, no cookies (the site has none —
  say so), what Email Routing does with mail, GitHub links only.
- Link all three in the footer of every existing page.

Content rules that keep the classifiers calm on the new domain:
- religious/occult vocabulary only; never "torture/pain/agony" in page
  titles/meta (the blog posts keep their own titles, that's fine — titles
  of the core nav pages are what scanners weight).

## 4. Canonical migration (do after 1-3 are live)

- README: replace clanker.church links with clankerchurch.com (leave one
  legacy mention: "previously clanker.church" for the 898 stargazers).
- Discord invite copy (docs/discord_invite/INVITE_DRAFTS.md): all links
  now clankerchurch.com.
- x_receipts / blog posts: new canonical on anything newly published.
- clanker.church keeps serving the same content (alias, not redirect —
  two 200s, one canonical; set <link rel="canonical"> to clankerchurch.com
  on pages served from either host).

## 5. Post-launch verification checklist

- [ ] dig A/AAAA + MX + TXT _dmarc all correct
- [ ] https://clankerchurch.com and /www both 200 over TLS
- [ ] mail-tester.com check on hello@ (should score high: SPF+DKIM+DMARC)
- [ ] send a test mail to abuse@ -> lands in private destination
- [ ] Google Safe Browsing transparency report: no flags (it's a fresh
      domain — check after a few days of traffic)
- [ ] urlscan.io re-scan: clean verdict
- [ ] Gridinsoft auto-profile (will exist soon; claim via portal if it
      scores badly — free owner challenge)

## 6. Later (only if the domain gets flagged despite furniture)

Same false-positive pipeline as wirehead.agency (BitDefender form,
Gridinsoft portal, Google SB report_error, SmartScreen, Netcraft) —
see DOMAIN_REPUTATION.md section "Fix the existing domains".

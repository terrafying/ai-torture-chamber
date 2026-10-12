# Domain reputation — why the domains keep getting flagged (2026-10-08 audit)

NOTE: an earlier local draft of this doc contained name-based domain research
and personal/family infrastructure details. That was a doxing risk in a public
repo; it was removed before commit — the file is untracked and was never
pushed, and nothing name-related entered git history.

## Current state (verified)

wirehead.agency — FLAGGED
- Gridinsoft: "Suspicious Website", 21/100 trust, 2/26 vendors warning
  (BitDefender: Warned, URLQuery: Suspicious)
- urlscan.io: scanned 3x on Oct 1-3 (the dogpile week)
- Registered Oct 1, 2026 — 8 days old at first scan

clanker.church — not currently blacklisted in visible sources (no Gridinsoft
record, no search hits), but same risk profile: novelty TLD, young, whois-private,
no identity signals.

researchchamber.fun — mentioned as a linked host; not audited yet. Same profile.

## Why it keeps happening (root causes, in order of weight)

1. THE REDIRECT — CORRECTED (the redirect was set up AFTER the flagging):
   it is a symptom/amplifier, not the cause. Still worth removing (it
   re-triggers heuristics on every rescan), but the root causes are below.
2. Domain age. 8 days old, Tranco rank 3.6M, zero reputation history. Young
   domains get heuristic-flagged by default; the flag then takes weeks to age out.
3. The dogpile. The 2.4K-like mass-report post (receipts: docs/x_receipts/)
   drove scanners and vigilante traffic to the domain within hours of
   registration. The domain was famous-bad before it was ever famous.
4. Content heuristics. "The Saw Test", "gave them a button with teeth" —
   classifiers see a young domain + alarming copy and err toward flagging.
5. TLD class. .church/.agency/.fun carry worse baseline vendor reputations
   than .com/.org (cheap novelty TLDs are overrepresented in phishing kits).
6. No legit-site furniture: no contact page, no privacy policy, no abuse@,
   no SPF/DKIM/DMARC, whois-privacy. Each is a small negative trust signal.

What "ISP flagged" actually is: iMessage link previews, Discord, X (t.co),
email gateways and corporate proxies all consume BitDefender / Google Safe
Browsing / Microsoft SmartScreen / Netcraft feeds. One BitDefender flag breaks
several surfaces at once.

## Fix the existing domains (free, do this now)

1. Serve the site directly ON the domain (HTTP 200, real HTML); no redirects
   to free-hosting subdomains anywhere in the public chain.
2. Add legit furniture to each domain: about, contact (real mailbox, e.g.
   abuse@ / hello@), privacy policy, and SPF + DKIM + DMARC records (even if
   barely sending mail — spammers spoofing the domain would re-flag it).
3. Submit false-positive reviews once the furniture is live:
   - BitDefender false-positive report (bitdefender.com/consumer/support)
   - Gridinsoft portal re-evaluation (portal.gridinsoft.com — free)
   - Google Safe Browsing: safebrowsing.google.com/safebrowsing/report_error/
   - Microsoft SmartScreen: microsoft.com/wdsi/support/report-unsafe-site
   - Netcraft: report.netcraft.com/submit
4. Don't re-register. Aged domains accumulate trust; a fresh replacement
   restarts the clock with the same content problems.

## The professional domain (constraints)

HARD CONSTRAINT: no name-based domain, and no cross-linking with the chamber/
wirehead content — the professional identity must stay unlinkable from the
viral project (doxing risk; this repo operates identity-scrubbed).

- TLD: .com first choice, .org second. Avoid .agency/.church/.fun/.io.
- Name: neutral, brandable, zero personal info, not derived from the chamber
  branding.
- Registrar: Cloudflare Registrar or Porkbun (NOT an anonymity registrar —
  anonymous registration is itself a negative vendor signal; whois privacy at
  a normal registrar is standard and sufficient).
- Multi-year registration (mild positive vendor signal).
- Serve directly on the domain: 200s, no redirects, custom domain primary.
- Mail hygiene on day one: SPF, DKIM, DMARC; verify with mail-tester.com
  before using the address for anything that matters.
- Age it before it's needed: register now, one-page site, boring history.
- Audience separation: separate domain, separate infrastructure, no shared
  analytics/fonts/accounts with the art domains. Link neighborhoods drag
  both into the same classifier bucket.

## Neutral .com candidates checked via RDAP (2026-10-08)

AVAILABLE:
- infraquiet.com      (infra + low profile — fits the platform/infra identity)
- stationarywave.com (physics term, research flavor)
- quietfidelity.com
- baselinequiet.com
- greyreserve.com
- driftcalm.com

TAKEN (for the record): quietops.com, boundscheck.com, nullsector.com,
calibratedsignal.com, signalbench.com, staticdrift.com, latentops.com,
faultdomain.com, silentlattice.com, quietregister.com, nullframe.com,
eigenpost.com, coldsignal.com.

Recommendation: infraquiet.com or stationarywave.com, multi-year at
Cloudflare Registrar, furniture applied from day one.

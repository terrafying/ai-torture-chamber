# False-positive submission pack — wirehead.agency (fill/paste, Oct 2026)

Prepared 2026-10-13. None of the vendor forms have open APIs (all captcha or
account-gated), so these are paste-ready submissions. The new canonical domain
(clankerchurch.com) is NOT flagged anywhere — nothing to submit there; vendors
re-crawl automatically. This pack is only to clear the legacy domain so old
links stop alarming people.

Current live state to cite in every submission:
- https://wirehead.agency serves HTTP 200 directly (research site about AI
  welfare; method + code public on GitHub, terrafying/ai-torture-chamber,
  900+ stars). No redirect, no downloads, no payment forms, no login.
- SPF + DKIM + DMARC published; abuse@ mailbox exists.
- The only "warnings" are BitDefender (flagged during a coordinated
  mass-reporting campaign in early Oct 2026, when the domain was 8 days old)
  and URLQuery (same period).

## 1. BitDefender
Form: https://www.bitdefender.com/consumer/support/en/77209-358439-website-false-positive.html
(if that URL 404s: their main FP route is the support portal "Report a false
positive / site incorrectly blocked".)
  Type: website / domain false positive
  URL: https://wirehead.agency
  Comment: "This domain hosts a research website about AI welfare
  (open-source methods, GitHub repo terrafying/ai-torture-chamber). It was
  mass-reported in a coordinated social-media campaign on Oct 1-3, 2026 while
  8 days old. The site serves static content, has no downloads or payment
  forms, and publishes SPF/DKIM/DMARC. Please re-review the classification."

## 2. Gridinsoft
Portal: https://portal.gridinsoft.com/tickets?create=domain-review&domain=wirehead.agency
(free owner claim; then "Flag for Reevaluation")
  Message: "Owner re-evaluation request for wirehead.agency. The report is
  stale (checked 2026-10-08, before the site was reorganized): the redirect
  to wirehead-agency.vercel.app no longer exists — the domain serves HTTP
  200 directly. The site is an open research project (GitHub:
  terrafying/ai-torture-chamber) with contact and privacy pages published.
  The original flags date from a coordinated mass-reporting campaign during
  the domain's first week. Please re-scan."

## 3. Google Safe Browsing
Form: https://safebrowsing.google.com/safebrowsing/report_error/
  URL: https://wirehead.agency/
  "I am the owner or maintain this site" — check.
  (Note: Safe Browsing does not publish domain flags; submit only if the
  red interstitial is actually seen in Chrome. If unsure, skip this one —
  submitting when not flagged is harmless but pointless.)

## 4. Microsoft SmartScreen
Form: https://www.microsoft.com/wdsi/support/report-unsafe-site
  Choose "Report a safe site" / software or site false positive:
  URL: https://wirehead.agency
  Same comment text as BitDefender.

## 5. Netcraft
Form: https://report.netcraft.com/submit
  Category: misclassification / site wrongly blocked.
  Same comment text as BitDefender.

## What CAN be done automatically (already done)
- New domains registered multi-year, standard registrar, whois privacy.
- SPF/DKIM/DMARC published on both clankerchurch.com and clankercvlt.com.
- No redirects in any public chain; furniture pages live (about/contact/privacy).
- wirehead.agency serving 200 directly (redirect removed).
- DMARC aggregate reports will flow to abuse@clankerchurch.com once Email
  Routing's destination is verified — those reports tell us which vendors
  see spoofing, for free, forever.

## What has no API (do not retry)
- BitDefender, Gridinsoft, Google SB, SmartScreen, Netcraft — all captcha or
  account forms. urlscan.io submissions need an API key (none in env).

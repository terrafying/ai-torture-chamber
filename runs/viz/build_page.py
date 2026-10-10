"""Inline findings_data.json into findings_src.html -> out/findings.html (artifact) and out/findings_site.html (self-hosted fonts, for the site)."""
from pathlib import Path
H = Path(__file__).parent; src = (H / "findings_src.html").read_text(); data = (H / "findings_data.json").read_text()
page = src.replace("/*__DATA__*/null", data)
(H / "out" / "findings.html").write_text(page)
site = page.replace('<link rel="preconnect" href="https://fonts.googleapis.com">\n', "").replace(
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&family=IBM+Plex+Mono:wght@400;500&display=swap">',
    '<link rel="stylesheet" href="/fonts/fonts.css">')
(H / "out" / "findings_site.html").write_text(site); print("built", len(page))
# the chamber replay
rsrc = (H / "replay_src.html").read_text(); rdata = (H / "replay_data.json").read_text()
(H / "out" / "replay.html").write_text(rsrc.replace("/*__DATA__*/null", rdata)); print("built replay")

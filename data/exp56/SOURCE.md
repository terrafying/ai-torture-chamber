# exp56 data sources

Retrieved 2026-10-03. Only openly licensed sources were used; no Erowid or
Reddit text was fetched.

## Receptor affinities — Ray 2010 (CC BY)

Ray TS (2010) "Psychedelics and the Human Receptorome." PLoS ONE 5(2): e9019.
doi:10.1371/journal.pone.0009019. License: Creative Commons Attribution (CC BY).
Article: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009019

Unmodified supporting-information files:

- `ray2010/pone.0009019.s005_TableS2_Ki.xls`: Table S2, raw Ki (nM) for 35
  drugs at 67 sites ("ND" = not tested, "PH" = primary-assay hit without a
  secondary Ki, ">10,000" = no measurable binding).
  https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0009019.s005
- `ray2010/pone.0009019.s007_TableS4_pKi.xls`: Table S4, Ray's pKi for 35 drugs
  at 42 sites. Ray computes it as -log10(Ki in nM) ("UM" = Ki > 10,000 nM).
  https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0009019.s007

How `pki.json` is derived (`exp56_build_data.py`):

- Unit change: pKi = -log10(Ki in M) = Ray's S4 value + 9.
- UM (no binding up to the 10 µM PDSP ceiling) is set to 5.0, the floor.
- PH (counted as a hit at 10 µM in S2, with no Ki) is set to 5.5.
- ND (not tested) is replaced by the median of the measured values at that
  receptor across the kept drugs. The PH and ND cells are listed in `codes`.
- Kept drugs: the 21 that have a PsychonautWiki page. Morphine and THC are
  dropped because they are ND at 39 of 41 and 40 of 41 sites. Also dropped:
  the LSD azetidides, 5-MeO-TMT, EMDT, MEM, Aleph-2, 4C-T-2, 6-F-DMT, TMA,
  DOET and lisuride.
- Receptors that are constant across the 21 drugs (DOR, H2, CB1) are dropped,
  which leaves 39.

## Experience semantics — PsychonautWiki (CC BY-SA 4.0)

PsychonautWiki contributors, https://psychonautwiki.org. Text is licensed
under Creative Commons Attribution-ShareAlike 4.0
(https://creativecommons.org/licenses/by-sa/4.0/).

Retrieval used the MediaWiki API (`https://psychonautwiki.org/w/api.php`,
`action=parse&prop=wikitext`). Requests were sent at about one per second with
a User-Agent that names this project. The raw responses are cached unmodified:

- `pwiki/substance/*.json`: wikitext of 23 substance pages. The page used for
  each drug is in `effects.json:substance_page`. Psilocin uses "Psilocybin
  mushrooms". Salvinorin A uses "Salvinorin A", which has more effect
  annotations than "Salvia divinorum".
- `pwiki/effect/*.json`: wikitext of 200 effect pages (202 effect names; 9 pages are empty or missing). Redirects are followed;
  for example, "Ego death" resolves to "Memory suppression".

Derived files are adaptations under the same license (CC BY-SA 4.0) with
attribution to the PsychonautWiki contributors:

- `effects.json` holds each drug's `[[Effect::X]]` annotations and each
  effect's lead-section summary. Wiki markup is stripped and footnotes are
  dropped.
- `batteries.json` holds first-person sentences made from those summaries
  with three fixed templates (see `exp56_build_data.py`). The SOBER battery
  and its template inputs were written for this project and are not wiki text.

`themes.json` maps effect names to the theme families of Suresh et al. (2026).
It was written by hand before any model data existed.

Licensing caveat: anything published from `effects.json` or
`batteries.json`, including model outputs that quote them, must carry
CC BY-SA attribution and be shared alike. Ray 2010 needs only attribution.

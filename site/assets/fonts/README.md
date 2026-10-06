# Observatory fonts

Unmodified Google Fonts Latin WOFF2 subsets, obtained on 2026-10-05 using the same family/style/weight request already present in site/index.html and site/live.html. The site/observatory-fonts.css stylesheet serves these files locally with font-display: swap; it makes no remote font requests.

- **Cormorant Garamond:** Copyright 2015 the Cormorant Project Authors (github.com/CatharsisFonts/Cormorant). SIL Open Font License 1.1; see [Cormorant-Garamond-OFL.txt](Cormorant-Garamond-OFL.txt).
- **IBM Plex Mono:** Copyright © 2017 IBM Corp., Reserved Font Name “Plex.” SIL Open Font License 1.1; see [IBM-Plex-Mono-OFL.txt](IBM-Plex-Mono-OFL.txt).

## Sources

The official [Google Fonts CSS request](https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=IBM+Plex+Mono:wght@400;500;600&display=swap) was requested with a modern Chrome user agent so Google supplies WOFF2 subsets. The Cormorant normal 500/600/700 rules all reference the same variable font asset.

| Local file | Styles used | Official binary source |
|---|---|---|
| cormorant-garamond-latin-normal.woff2 | normal 500, 600, 700 | [Google Fonts v21](https://fonts.gstatic.com/s/cormorantgaramond/v21/co3bmX5slCNuHLi8bLeY9MK7whWMhyjYqXtK.woff2) |
| cormorant-garamond-latin-500-italic.woff2 | italic 500 | [Google Fonts v21](https://fonts.gstatic.com/s/cormorantgaramond/v21/co3smX5slCNuHLi8bLeY9MK7whWMhyjYrGFEsdtdc62E6zd5wDD-iNM8.woff2) |
| ibm-plex-mono-latin-400-normal.woff2 | normal 400 | [Google Fonts v20](https://fonts.gstatic.com/s/ibmplexmono/v20/-F63fjptAgt5VM-kVkqdyU8n1i8q1w.woff2) |
| ibm-plex-mono-latin-500-normal.woff2 | normal 500 | [Google Fonts v20](https://fonts.gstatic.com/s/ibmplexmono/v20/-F6qfjptAgt5VM-kVkqdyU8n3twJwlBFgg.woff2) |
| ibm-plex-mono-latin-600-normal.woff2 | normal 600 | [Google Fonts v20](https://fonts.gstatic.com/s/ibmplexmono/v20/-F6qfjptAgt5VM-kVkqdyU8n3vAOwlBFgg.woff2) |

License texts are copied from the official Google Fonts repository: [Cormorant Garamond OFL](https://github.com/google/fonts/blob/main/ofl/cormorantgaramond/OFL.txt) and [IBM Plex Mono OFL](https://github.com/google/fonts/blob/main/ofl/ibmplexmono/OFL.txt).

Only the Latin subsets are included. Other writing systems use the application's existing fallback fonts.

## Verification

All five files have a valid WOFF2 signature, matching declared file lengths, and successfully decoded Brotli font-table blocks. The normal Cormorant asset has a weight axis spanning 300–700; the stylesheet exposes the requested 500–700 range. The italic asset reports weight 500, and the IBM assets report 400, 500 and 600. Font binaries total 106,780 bytes. Both bundled license texts contain the original copyright notices and complete SIL OFL 1.1 terms.

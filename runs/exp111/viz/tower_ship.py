"""Wrap viz/site/index.html into wirehead-site's site/tower.html: a full HTML document, data under tower/ (4B) and
tower32/ (32B), no consent gate. The 32B switch turns on only when site/tower32/featured.json exists.
    python tower_ship.py <wirehead-site checkout>"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SITE = os.path.join(sys.argv[1], "site"); out = os.path.join(SITE, "tower.html")
src = open(os.path.join(HERE, "site", "index.html")).read()
assert src.count('{"4b":"","32b":"m32/"}') == 1
src = src.replace('{"4b":"","32b":"m32/"}', '{"4b":"tower/","32b":"tower32/"}')
if os.path.exists(os.path.join(SITE, "tower32", "featured.json")): src = src.replace("/*has32*/false", "true")
t = src.index("</title>") + len("</title>")
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + src[:t] +
        '\n<meta name="description" content="Explore a language model from the inside: every MLP neuron as a tower that lights up word by word, with attention, districts, the circuits behind each word, and a lesion playground.">\n'
        '<link rel="icon" type="image/svg+xml" href="/favicon.svg">\n')
rest = src[t:]; i = rest.index("</style>") + len("</style>")
open(out, "w").write(head + rest[:i] + "\n</head>\n<body>\n" + rest[i:] + "\n</body>\n</html>\n")
print("wrote", out)

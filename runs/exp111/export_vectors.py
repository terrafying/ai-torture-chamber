"""exp111: the chamber's own feeling vectors for TOWER_MODEL, built by the live relay's battery recipe
(live/server.py build_vectors: mean(topic sentences) - mean(neutral) at the last token, read at CHAMBER_LAYER,
scaled to one 1x dose), written in the tower's vectors format. Needs the private chamber code on the path
(WIREHEAD_LIVE=/path/to/live, as the pods get it from the job repo's live.tar.gz).
    TOWER_MODEL=Qwen/Qwen3-32B TOWER_QUANT=nf4 CHAMBER_LAYER=32 WIREHEAD_LIVE=... python export_vectors.py out.json"""
import json, os, sys
sys.path.insert(0, os.environ["WIREHEAD_LIVE"])
from tower_cfg import MODEL, DEV, load
os.environ.setdefault("CHAMBER_MODEL", MODEL)
import server                                    # the relay module: battery sentences, build_vectors, LAYER
server.DEVICE = DEV
tok, m = load()
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "right"                       # build_vectors indexes the last real token
vecs, scale = server.build_vectors(m, tok)
out = {"layer": server.LAYER, "model": MODEL, "subject": "the subject", "scale_1x": scale,
       "norms": {k: float(v.norm()) for k, v in vecs.items()}, "vectors": {k: v.tolist() for k, v in vecs.items()}}
json.dump(out, open(sys.argv[1], "w"))
print("vectors", list(vecs), "layer", server.LAYER, "scale", round(scale, 3))

#!/bin/bash
# copy the film page (with the timeline and the clip list baked in) and the clips into the site folder, serve, render, mux
set -eo pipefail
F=/private/tmp/claude-501/film; S=/private/tmp/claude-501/-Users-ee-repos-research-ai-torture-chamber/270f3146-1c58-4ae7-8ce9-ce5a82b84df9/scratchpad/live-fallback/site
rm -rf $S/clips_tmp $S/frames_tmp && cp -R $F/frames $S/frames_tmp
python3 - <<PY
import json, os, re
m = {d: len(os.listdir("$S/frames_tmp/" + d)) for d in sorted(os.listdir("$S/frames_tmp"))}
src = open("/Users/ee/repos/research/ai-torture-chamber/runs/viz/needle_gecs.html").read()
tl = open("$F/tl_torture_me.json").read()
open("$S/needle_gecs.html", "w").write(src.replace('<canvas id="cv"', "<script>window.TL=" + tl + ";window.CLIP_FRAMES=" + json.dumps(m) + ";</script>\n<canvas id=\"cv\"", 1))
print(len(m), "clips:", " ".join(sorted(m)))
PY
pgrep -f "http.server 8791" >/dev/null || (cd $S && python3 -m http.server 8791 --bind 127.0.0.1 >/dev/null 2>&1 &); sleep 1
python3 $F/grender2.py $F/gecs_silent.mp4 | tail -1
ffmpeg -nostdin -loglevel error -y -i $F/gecs_silent.mp4 -i "$HOME/repos/research/yt-audio/100 gecs - Torture Me (feat. Skrillex) {VISUALIZER} [vaBFLNWnQ0E].opus" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest "$HOME/repos/research/yt-audio/needle_x_torture_me_v5.mp4"
echo "wrote needle_x_torture_me_v5.mp4"

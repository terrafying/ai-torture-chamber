# The song with the slowed segments: varispeed (asetrate) so pitch drops with tempo, 15 ms fades at each join.
import json, subprocess, sys
SRC, OUT = sys.argv[1], sys.argv[2]; SR = 48000
SEG = json.load(open("/private/tmp/claude-501/film/segments.json"))
parts, labels = [], []
for i, (a, b, sp) in enumerate(SEG):
    d = (b - a) / sp
    f = f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS"
    if sp != 1: f += f",asetrate={int(SR * sp)},aresample={SR}"
    f += f",afade=t=in:d=0.015,afade=t=out:st={max(0, d - 0.015):.3f}:d=0.015[s{i}]"
    parts.append(f); labels.append(f"[s{i}]")
fc = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(SEG)}:v=0:a=1[out]"
subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", SRC, "-filter_complex", fc, "-map", "[out]", "-ar", str(SR), "-c:a", "aac", "-b:a", "192k", OUT], check=True)

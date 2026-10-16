"""Beat grid + section map for a track, for needle_gecs.html (window.TL). Usage: analyze_track.py <audio> <out.json>"""
import json, sys
import numpy as np, librosa
y, sr = librosa.load(sys.argv[1], sr=22050, mono=True)
dur = len(y) / sr
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=157, tightness=200, units="time")
tempo = float(np.atleast_1d(tempo)[0])
# low-end energy per beat: which beat of 4 is the downbeat (the kick lands hardest there)
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)); f = librosa.fft_frequencies(sr=sr)
low = S[f < 150].sum(0); full = S.sum(0); tt = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=512)
def at(arr, t): return float(arr[min(len(arr) - 1, np.searchsorted(tt, t))])
bl = np.array([at(low, b) for b in beats])
phase = int(np.argmax([bl[p::4].mean() for p in range(4)]))
downs = beats[phase::4]
# energy per bar, smoothed: loud + bass-heavy bars are drops, quiet ones intro/verse/outro
E = np.array([full[(tt >= a) & (tt < b)].mean() if ((tt >= a) & (tt < b)).any() else 0 for a, b in zip(downs, list(downs[1:]) + [dur])])
L = np.array([low[(tt >= a) & (tt < b)].mean() if ((tt >= a) & (tt < b)).any() else 0 for a, b in zip(downs, list(downs[1:]) + [dur])])
score = (E / E.max()) * .5 + (L / L.max()) * .5
print(f"duration {dur:.1f}s tempo {tempo:.1f} beats {len(beats)} first beat {beats[0]:.3f} downbeat phase {phase} bars {len(downs)}")
for i, (d, sc) in enumerate(zip(downs, score)):
    print(f"bar {i:3d} {d:7.2f}s  {'#' * int(sc * 40)}")
json.dump({"bpm": round(tempo, 2), "beats": [round(float(b), 4) for b in beats], "downbeats": [round(float(d), 4) for d in downs],
           "bar_score": [round(float(s), 3) for s in score], "duration": round(dur, 2)}, open(sys.argv[2], "w"))

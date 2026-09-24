"""Пословные тайминги ролика 27 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с). Соседние отрезки
с зазором <0.15с слиты: такой зазор — придыхание внутри фразы, а не пауза.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/27/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align27.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/27/words.json"

SPANS = [
    (0.212, 2.920), (3.090, 8.154), (8.495, 13.389), (13.714, 15.069),
    (15.539, 18.472), (18.868, 21.291), (21.959, 25.337), (25.521, 26.797),
    (27.292, 30.374), (30.864, 34.644), (35.047, 37.077), (37.571, 40.802),
    (41.180, 44.694), (45.117, 49.780), (50.322, 52.628),
]

subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                "-vn", "-ac", "1", "-ar", "16000", WAV], check=True)
audio = whisper.load_audio(WAV)
model = whisper.load_model("large-v3")

words = []
for (a, b) in SPANS:
    seg = audio[int(a * 16000):int(b * 16000)]
    r = model.transcribe(seg, language="ru", word_timestamps=True,
                         fp16=False, condition_on_previous_text=False)
    got = [w for s in r["segments"] for w in s.get("words", [])]
    if not got:
        print(f"!! пусто {a}-{b}", flush=True)
        continue
    lo = got[0]["start"]
    hi = got[-1]["end"]
    k = (b - a) / max(1e-6, hi - lo)
    for w in got:
        words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                          end=round(a + (w["end"] - lo) * k, 3),
                          word=w["word"].strip()))
    print(f"{a:6.2f}-{b:6.2f}  k={k:.3f}  " + " ".join(w["word"].strip() for w in got), flush=True)

json.dump(words, open(OUT, "w"), ensure_ascii=False, indent=1)
print("слов:", len(words), "->", OUT)

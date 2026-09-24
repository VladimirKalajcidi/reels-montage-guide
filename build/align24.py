"""Пословные тайминги ролика 24 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с), то есть измерены
по звуку, а не предсказаны моделью. Паузы короче 0.30с считаются внутренними
для фразы и отрезок по ним не рвётся.
"""
import json
import subprocess
import whisper   # 3.11 framework python, там стоит openai-whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/24/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align24.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/24/words.json"

SPANS = [
    (0.000, 4.797), (5.184, 7.596), (8.131, 11.178), (11.620, 16.872),
    (17.294, 21.423), (21.961, 26.486), (26.845, 29.270), (29.751, 33.161),
    (33.563, 37.826), (38.376, 40.193), (40.587, 43.464), (43.905, 45.443),
    (45.782, 49.241), (49.800, 52.153),
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

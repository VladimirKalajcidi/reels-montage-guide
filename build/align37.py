"""Пословные тайминги ролика 37.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с) по `_align37.wav`.
Пауза короче 0.20с считана как внутрифразовая и слита: 39.959-40.086 (0.127с).
Начало речи 0.090 и конец 55.830 сняты по огибающей (10мс RMS), не детектором.
Внутри каждого отрезка whisper large-v3 растягивается на измеренный интервал:
сплошной прогон дрейфует и теряет предложения.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/37/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align37.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/37/words.json"

SPANS = [
    (0.090, 3.354), (3.723, 5.088), (5.299, 8.004), (8.305, 11.188),
    (11.658, 14.019), (14.335, 15.952), (16.380, 19.466), (19.998, 23.398),
    (23.688, 24.568), (24.972, 25.616), (25.992, 29.456), (30.055, 32.096),
    (32.480, 34.045), (34.321, 36.280), (36.618, 38.299), (38.759, 41.103),
    (41.546, 43.940), (44.331, 45.550), (45.949, 48.882), (49.234, 53.154),
    (53.622, 55.830),
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

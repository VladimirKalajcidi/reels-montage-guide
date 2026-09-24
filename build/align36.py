"""Пословные тайминги ролика 36.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с) по `_align36.wav`.
Паузы короче 0.20с считаны как внутрифразовые и слиты:
9.905-10.058 (0.153с), 15.572-15.727 (0.155с), 45.483-45.608 (0.126с).
Внутри каждого отрезка whisper large-v3 растягивается на измеренный интервал:
сплошной прогон дрейфует и теряет предложения.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/36/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align36.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/36/words.json"

SPANS = [
    (0.178, 3.621), (4.110, 7.704), (8.102, 10.114), (10.547, 14.108),
    (14.447, 17.022), (17.487, 20.758), (21.110, 24.473), (24.927, 29.115),
    (29.576, 30.837), (31.131, 33.417), (33.779, 35.908), (36.256, 41.037),
    (41.417, 43.262), (43.669, 46.038),
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

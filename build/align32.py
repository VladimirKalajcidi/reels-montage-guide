"""Пословные тайминги ролика 32.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с), зазоры <0.15с слиты
(5.338-5.480 и 50.449-50.576). Внутри каждого отрезка whisper large-v3
растягивается на измеренный интервал.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/32/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align32.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/32/words.json"

SPANS = [
    (0.252, 2.305), (2.671, 5.499), (5.900, 7.779), (8.038, 9.094),
    (9.466, 12.736), (13.126, 14.560), (14.918, 17.204), (17.545, 18.813),
    (19.257, 25.700), (26.122, 30.328), (30.777, 33.344), (33.525, 36.089),
    (36.550, 38.392), (38.763, 43.650), (44.074, 48.124), (48.588, 51.027),
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

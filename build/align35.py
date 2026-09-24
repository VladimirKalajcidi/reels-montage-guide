"""Пословные тайминги ролика 35.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с) по `_align35.wav`.
Паузы короче 0.18с считаются внутрифразовыми и слиты: 2.423-2.573,
12.732-12.864, 20.912-21.051, 21.904-22.062, 32.677-32.849,
39.012-39.135, 47.889-48.019.
Внутри каждого отрезка whisper large-v3 растягивается на измеренный интервал:
сплошной прогон дрейфует и теряет предложения.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/35/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align35.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/35/words.json"

SPANS = [
    (0.156, 4.263), (4.806, 7.041), (7.408, 10.809), (11.216, 16.533),
    (16.928, 18.587), (19.069, 23.286), (23.777, 26.000), (26.331, 27.922),
    (28.285, 31.826), (32.240, 34.677), (35.127, 40.562), (41.028, 45.556),
    (46.091, 48.411),
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

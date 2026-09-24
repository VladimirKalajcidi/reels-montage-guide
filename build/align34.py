"""Пословные тайминги ролика 34.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с) по `_align34.wav`.
Пауза 37.517-37.639 (0.122с) — внутри фразы, отрезки 35.721-37.517 и
37.639-37.948 слиты в один 35.721-37.948.
Внутри каждого отрезка whisper large-v3 растягивается на измеренный интервал:
сплошной прогон дрейфует и теряет предложения.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/34/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align34.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/34/words.json"

SPANS = [
    (0.000, 2.804), (3.283, 8.299), (8.664, 9.439), (9.878, 10.719),
    (10.960, 11.827), (12.225, 16.350), (16.731, 19.470), (19.978, 22.134),
    (22.386, 24.020), (24.317, 26.619), (27.063, 30.981), (31.294, 32.933),
    (33.364, 35.291), (35.721, 37.948),
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

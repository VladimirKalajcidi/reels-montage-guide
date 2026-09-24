"""Пословные тайминги ролика 31.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с), зазоры <0.15с слиты.
Внутри каждого отрезка whisper large-v3 растянут на измеренный интервал.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/31/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align31.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/31/words.json"

SPANS = [
    (0.000, 5.157), (5.669, 8.013), (8.374, 10.554), (10.845, 12.123),
    (12.606, 14.932), (15.306, 15.725), (15.962, 17.733), (18.053, 19.683),
    (20.276, 24.787), (25.239, 29.340), (29.867, 32.602), (32.886, 36.029),
    (36.566, 38.247), (38.676, 42.108), (42.428, 44.196), (44.752, 46.923),
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

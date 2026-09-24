"""Пословные тайминги ролика 22 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с), то есть измерены
по звуку, а не предсказаны моделью.
"""
import json
import subprocess
import numpy as np
import whisper   # запускать интерпретатором 3.12, где стоит openai-whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/22/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align22.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/22/words.json"

SPANS = [
    (0.244, 5.860), (6.355, 9.987), (10.399, 12.672), (13.063, 16.744),
    (17.151, 19.130), (19.626, 22.491), (22.985, 24.512), (24.821, 26.689),
    (27.096, 28.928), (29.239, 32.165), (32.457, 34.716), (34.952, 36.935),
    (37.530, 39.848), (40.168, 43.053), (43.445, 45.956), (46.398, 50.725),
    (51.121, 52.518), (52.934, 55.290),
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
        print(f"!! пусто {a}-{b}")
        continue
    # растянуть выдачу модели на измеренный отрезок речи
    lo = got[0]["start"]
    hi = got[-1]["end"]
    k = (b - a) / max(1e-6, hi - lo)
    for w in got:
        words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                          end=round(a + (w["end"] - lo) * k, 3),
                          word=w["word"].strip()))
    print(f"{a:6.2f}-{b:6.2f}  k={k:.3f}  " + " ".join(w["word"].strip() for w in got))

json.dump(words, open(OUT, "w"), ensure_ascii=False, indent=1)
print("слов:", len(words), "->", OUT)

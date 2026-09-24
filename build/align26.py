"""Пословные тайминги ролика 26 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с), то есть измерены
по звуку, а не предсказаны моделью. Соседние отрезки с зазором <0.15с слиты:
такой зазор — это придыхание внутри фразы, а не пауза между предложениями.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/26/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align26.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/26/words.json"

SPANS = [
    (0.366, 5.877), (6.439, 11.264), (11.688, 15.705), (16.204, 19.196),
    (19.561, 22.925), (23.311, 28.018), (28.554, 30.883), (31.210, 33.242),
    (33.737, 36.167), (36.539, 39.405), (39.750, 42.569), (43.108, 45.998),
    (46.318, 48.072), (48.554, 50.913),
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

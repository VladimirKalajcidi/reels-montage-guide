"""Пословные тайминги ролика 23 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с), то есть измерены
по звуку, а не предсказаны моделью. Паузы короче 0.30с считаются внутренними
для фразы и отрезок по ним не рвётся.
"""
import json
import subprocess
import whisper   # запускать интерпретатором 3.12, где стоит openai-whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/23/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align23.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/23/words.json"

SPANS = [
    (0.265, 4.924), (5.341, 9.407), (9.803, 13.590), (14.136, 18.704),
    (19.122, 21.714), (22.111, 28.007), (28.580, 31.246), (31.628, 33.706),
    (34.099, 36.687), (37.113, 40.439), (40.726, 43.451), (44.017, 46.380),
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
    # растянуть выдачу модели на измеренный отрезок речи
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

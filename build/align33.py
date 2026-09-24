"""Пословные тайминги ролика 33.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с) по `_align33.wav`;
паузы короче 0.15с не считаются границей (внутри слова), «речь» длиной
меньше 0.10с между двумя паузами приклеивается к паузе (9.70-10.41).
Пауза 33.735-33.905 (0.170с) — вдох внутри фразы «переходить так / от числа
к числу»: разрез по ней давал у whisper мусор («очень сладкий слух»), отрезки
слиты в один 32.347-34.852.
Внутри каждого отрезка whisper large-v3 растягивается на измеренный интервал:
сплошной прогон дрейфует и теряет предложения.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/33/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align33.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/33/words.json"

SPANS = [
    (0.000, 2.449), (2.842, 5.107), (5.289, 8.347), (8.737, 9.700),
    (10.407, 13.115), (13.446, 15.235), (15.666, 16.705), (16.874, 17.686),
    (18.288, 22.706), (23.129, 24.949), (25.396, 28.067), (28.385, 29.723),
    (30.039, 31.943), (32.347, 34.852), (35.307, 38.712),
    (39.069, 42.500), (42.989, 44.524), (44.807, 45.824), (46.262, 51.437),
    (52.021, 54.377),
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

"""Пословные тайминги ролика 6 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с), то есть измерены
по звуку, а не предсказаны моделью.
"""
import json
import subprocess
import whisper   # запускать интерпретатором 3.12, где стоит openai-whisper

SRC = "/Users/vladimirkalajcidi/reels_good2/videos/6/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good2/build/assets/_align6.wav"
OUT = "/Users/vladimirkalajcidi/reels_good2/videos/6/words.json"

SPANS = [
    (0.000, 4.790), (5.204, 11.492), (12.054, 18.386), (18.716, 19.939),
    (20.322, 25.982), (26.302, 27.530), (27.881, 32.879), (33.271, 34.833),
    (35.302, 39.130), (39.558, 42.703), (43.158, 47.460), (47.960, 52.657),
]
SPANS = [s for s in SPANS if s[1] > s[0]]

PROMPT = ("геофизик-интерпретатор сейсморазведка скважина буровик месторождение "
          "обратная задача преобразование Фурье волновые уравнения тригонометрия "
          "УЗИ томография отраженная волна нефть")

subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                "-vn", "-ac", "1", "-ar", "16000", WAV], check=True)
audio = whisper.load_audio(WAV)
model = whisper.load_model("large-v3")

words = []
for (a, b) in SPANS:
    seg = audio[int(a * 16000):int(b * 16000)]
    r = model.transcribe(seg, language="ru", word_timestamps=True,
                         fp16=False, condition_on_previous_text=False,
                         initial_prompt=PROMPT)
    got = [w for s in r["segments"] for w in s.get("words", [])]
    if not got:
        print(f"!! пусто {a}-{b}")
        continue
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

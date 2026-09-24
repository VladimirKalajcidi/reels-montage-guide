"""Пословные тайминги ролика 30: парадокс дружбы (friendship paradox).

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с), зазоры <0.15с слиты.
Внутри каждого отрезка whisper large-v3 растянут на измеренный интервал.

Отрезок 34.511-36.497 large-v3 услышал как «математика и выбор» — грамматически
не сходится («может быть просто X» требует творительного падежа). Черновой
проход medium-моделью по всему файлу дал «математикой выборки», это и есть
верный вариант (согласуется с «быть просто»). В words.json интервал вручную
переразмечен на два слова по пропорции длины: «математикой» 35.423-36.079,
«выборки» 36.079-36.497.
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/30/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align30.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/30/words.json"

SPANS = [
    (0.121, 2.534), (2.878, 4.111), (4.377, 7.939), (8.328, 9.908),
    (10.323, 11.037), (11.207, 14.374), (14.817, 18.751), (19.082, 21.171),
    (21.599, 23.726), (24.179, 29.266), (29.763, 31.509), (31.833, 34.159),
    (34.511, 36.497), (37.109, 39.365),
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

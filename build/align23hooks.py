"""Пословные тайминги дублей хука ролика 23 — той же схемой, что и основной
исходник: границы речи из ffmpeg silencedetect + огибающая RMS, каждый отрезок
отдельным прогоном large-v3, выдача растянута на измеренный интервал.

Правые границы сняты по огибающей, а не по длительности файла: у hook2 речь
смолкает на 3.39 при файле 3.47. Растягивать выдачу модели на хвост из дыхания
значит уводить субтитры к концу фразы.
"""
import json
import subprocess

import whisper   # запускать интерпретатором 3.12, где стоит openai-whisper

VID = "/Users/vladimirkalajcidi/reels_good/videos/23"
BUILD = "/Users/vladimirkalajcidi/reels_good/build"

TAKES = {
    "hook1": [(0.160, 4.140)],
    "hook2": [(0.150, 3.390)],
}

model = whisper.load_model("large-v3")
for name, spans in TAKES.items():
    wav = f"{BUILD}/assets/_align_{name}23.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", f"{VID}/{name}.mov", "-vn", "-ac", "1", "-ar", "16000",
                    wav], check=True)
    audio = whisper.load_audio(wav)
    words = []
    for (a, b) in spans:
        seg = audio[int(a * 16000):int(b * 16000)]
        r = model.transcribe(seg, language="ru", word_timestamps=True,
                             fp16=False, condition_on_previous_text=False)
        got = [w for s in r["segments"] for w in s.get("words", [])]
        if not got:
            print(f"!! пусто {name} {a}-{b}", flush=True)
            continue
        lo, hi = got[0]["start"], got[-1]["end"]
        k = (b - a) / max(1e-6, hi - lo)
        for w in got:
            words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                              end=round(a + (w["end"] - lo) * k, 3),
                              word=w["word"].strip()))
        print(f"{name} {a:5.2f}-{b:5.2f} k={k:.3f}  "
              + " ".join(w["word"].strip() for w in got), flush=True)
    out = f"{VID}/words_{name}.json"
    json.dump(words, open(out, "w"), ensure_ascii=False, indent=1)
    print("слов:", len(words), "->", out, flush=True)

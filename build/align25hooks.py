"""Пословные тайминги дублей хука ролика 25 — той же схемой, что и основной
исходник: границы речи из ffmpeg silencedetect / огибающей RMS, каждый отрезок
отдельным прогоном large-v3, выдача растянута на измеренный интервал.
"""
import json
import subprocess

import whisper   # 3.11 framework python, там стоит openai-whisper

VID = "/Users/vladimirkalajcidi/reels_good/videos/25"
BUILD = "/Users/vladimirkalajcidi/reels_good/build"

# Оба дубля говорят до самого конца файла: огибающая на последнем окне
# у hook1 -21 dB, у hook2 -31 dB, тишины в хвосте нет.
TAKES = {
    "hook1": [(0.220, 2.300), (2.400, 5.877)],
    "hook2": [(0.120, 4.111)],
}

model = whisper.load_model("large-v3")
for name, spans in TAKES.items():
    wav = f"{BUILD}/assets/_align25_{name}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", f"{VID}/{name}.mov", "-vn", "-ac", "1", "-ar", "16000",
                    wav], check=True)
    audio = whisper.load_audio(wav)
    words = []
    for (a, b) in spans:
        r = model.transcribe(audio[int(a * 16000):int(b * 16000)], language="ru",
                             word_timestamps=True, fp16=False,
                             condition_on_previous_text=False)
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
    json.dump(words, open(f"{VID}/words_{name}.json", "w"), ensure_ascii=False, indent=1)
    print(f"{name}: слов {len(words)}", flush=True)

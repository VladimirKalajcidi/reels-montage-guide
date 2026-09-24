"""Пословные тайминги переснятого дубля hook1 ролика 25.

Первый присланный файл был обрезан на 5.877с прямо на последнем слове;
новый идёт до 6.258с, речь кончается на 6.082 по огибающей, дальше тишина.
Границы отрезков — silencedetect (-34dB / 0.12с), как у основного исходника.
"""
import json
import subprocess

import whisper

VID = "/Users/vladimirkalajcidi/reels_good/videos/25"
BUILD = "/Users/vladimirkalajcidi/reels_good/build"
SPANS = [(0.262, 2.266), (2.429, 6.082)]

wav = f"{BUILD}/assets/_align25_hook1.wav"
subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", f"{VID}/hook1.mov", "-vn", "-ac", "1", "-ar", "16000",
                wav], check=True)
audio = whisper.load_audio(wav)
model = whisper.load_model("large-v3")
words = []
for (a, b) in SPANS:
    r = model.transcribe(audio[int(a * 16000):int(b * 16000)], language="ru",
                         word_timestamps=True, fp16=False,
                         condition_on_previous_text=False)
    got = [w for s in r["segments"] for w in s.get("words", [])]
    lo, hi = got[0]["start"], got[-1]["end"]
    k = (b - a) / max(1e-6, hi - lo)
    for w in got:
        words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                          end=round(a + (w["end"] - lo) * k, 3),
                          word=w["word"].strip()))
    print(f"hook1 {a:5.2f}-{b:5.2f} k={k:.3f}  "
          + " ".join(w["word"].strip() for w in got), flush=True)
json.dump(words, open(f"{VID}/words_hook1.json", "w"), ensure_ascii=False, indent=1)
print("слов:", len(words), flush=True)

"""Пословные тайминги хуков ролика 32.

Границы отрезков — ffmpeg silencedetect (-34dB, 0.12с), зазоры <0.15с слиты.
Внутри каждого отрезка whisper large-v3 растянут на измеренный интервал —
та же схема, что в align32.py для основного дубля.
"""
import json
import subprocess
import whisper

BASE = "/Users/vladimirkalajcidi/reels_good/videos/32"
BUILD = "/Users/vladimirkalajcidi/reels_good/build"

SPANS = {
    1: [(0.000, 2.061), (2.427, 5.484), (5.662, 8.600)],
    2: [(0.000, 2.829), (3.174, 5.930)],
}

model = whisper.load_model("large-v3")
for n, spans in SPANS.items():
    src = f"{BASE}/hook{n}.mov"
    wav = f"{BUILD}/assets/_align32h{n}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
    audio = whisper.load_audio(wav)
    words = []
    for (a, b) in spans:
        seg = audio[int(a * 16000):int(b * 16000)]
        r = model.transcribe(seg, language="ru", word_timestamps=True,
                             fp16=False, condition_on_previous_text=False)
        got = [w for s in r["segments"] for w in s.get("words", [])]
        if not got:
            print(f"!! пусто hook{n} {a}-{b}", flush=True)
            continue
        lo, hi = got[0]["start"], got[-1]["end"]
        k = (b - a) / max(1e-6, hi - lo)
        for w in got:
            words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                              end=round(a + (w["end"] - lo) * k, 3),
                              word=w["word"].strip()))
        print(f"hook{n} {a:6.2f}-{b:6.2f}  k={k:.3f}  "
              + " ".join(w["word"].strip() for w in got), flush=True)
    out = f"{BASE}/words_hook{n}.json"
    json.dump(words, open(out, "w"), ensure_ascii=False, indent=1)
    print("слов:", len(words), "->", out, flush=True)

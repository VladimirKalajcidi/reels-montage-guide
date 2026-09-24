"""Пословные тайминги альтернативных хуков ролика 30.

hook1: silencedetect (-34dB/0.12с) не нашёл ни одной паузы — дубль идёт
сплошняком, весь файл один отрезок. hook2: тишина 0-0.183, короткий внутренний
разрыв 3.517-3.661 (0.14с) слит как придыхание, хвостовая тишина с 5.045.
"""
import json
import subprocess
import whisper

V = "/Users/vladimirkalajcidi/reels_good/videos/30"
B = "/Users/vladimirkalajcidi/reels_good/build"

SPANS = {
    "hook1": [(0.000, 4.010)],
    "hook2": [(0.183, 5.045)],
}

model = whisper.load_model("large-v3")
for name, spans in SPANS.items():
    wav = f"{B}/assets/_align30_{name}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", f"{V}/{name}.mov", "-vn", "-ac", "1", "-ar", "16000", wav],
                   check=True)
    audio = whisper.load_audio(wav)
    words = []
    for (a, b) in spans:
        seg = audio[int(a * 16000):int(b * 16000)]
        r = model.transcribe(seg, language="ru", word_timestamps=True,
                             fp16=False, condition_on_previous_text=False)
        got = [w for s in r["segments"] for w in s.get("words", [])]
        lo, hi = got[0]["start"], got[-1]["end"]
        k = (b - a) / max(1e-6, hi - lo)
        for w in got:
            words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                              end=round(a + (w["end"] - lo) * k, 3),
                              word=w["word"].strip()))
        print(f"{name} {a:5.2f}-{b:5.2f} k={k:.3f}  "
              + " ".join(w["word"].strip() for w in got), flush=True)
    json.dump(words, open(f"{V}/words_{name}.json", "w"), ensure_ascii=False, indent=1)
    print("  ->", f"{V}/words_{name}.json", len(words), "слов")

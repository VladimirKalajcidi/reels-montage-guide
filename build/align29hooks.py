"""Пословные тайминги альтернативных хуков ролика 29."""
import json
import subprocess

import whisper

V = "/Users/vladimirkalajcidi/reels_good/videos/29"
B = "/Users/vladimirkalajcidi/reels_good/build"

# Границы речи взяты по silencedetect. У hook2 последнее слово доходит до
# конца файла, поэтому правой границей служит фактическая длительность аудио.
SPANS = {
    "hook1": (0.150, 5.047),
    "hook2": (0.098, 4.795),
}

model = whisper.load_model("medium")
for name, (a, b) in SPANS.items():
    wav = f"{B}/assets/_align29_{name}.wav"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", f"{V}/{name}.mov", "-vn", "-ac", "1", "-ar", "16000", wav,
    ], check=True)
    audio = whisper.load_audio(wav)
    seg = audio[int(a * 16000):int(b * 16000)]
    result = model.transcribe(
        seg, language="ru", word_timestamps=True, fp16=False,
        condition_on_previous_text=False,
    )
    got = [w for s in result["segments"] for w in s.get("words", [])]
    lo, hi = got[0]["start"], got[-1]["end"]
    scale = (b - a) / max(1e-6, hi - lo)
    words = [
        dict(
            start=round(a + (w["start"] - lo) * scale, 3),
            end=round(a + (w["end"] - lo) * scale, 3),
            word=w["word"].strip(),
        )
        for w in got
    ]
    out = f"{V}/words_{name}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(words, fh, ensure_ascii=False, indent=1)
    print(name, " ".join(w["word"] for w in words), flush=True)
    print(" ->", out, len(words), "слов", flush=True)

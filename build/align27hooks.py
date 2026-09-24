"""Пословные тайминги хуков ролика 27 (hook1.mov, hook2.mov).

Каждый дубль — один речевой отрезок между паузами по краям, поэтому
транскрибируется целиком одним проходом whisper large-v3, без разбиения
на отрезки (в отличие от align27.py, где отрезков было 15).
"""
import json
import subprocess
import whisper

VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/27"

FILES = {
    "hook1": (f"{VIDEO_DIR}/hook1.mov", 0.0, 5.597),
    "hook2": (f"{VIDEO_DIR}/hook2.mov", 0.0, 6.327),
}

model = whisper.load_model("large-v3")

for name, (src, a, b) in FILES.items():
    wav = f"/Users/vladimirkalajcidi/reels_good/build/assets/_align27_{name}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
    audio = whisper.load_audio(wav)
    r = model.transcribe(audio, language="ru", word_timestamps=True,
                         fp16=False, condition_on_previous_text=False)
    got = [w for s in r["segments"] for w in s.get("words", [])]
    words = [dict(start=round(w["start"], 3), end=round(w["end"], 3),
                  word=w["word"].strip()) for w in got]
    out = f"{VIDEO_DIR}/words_{name}.json"
    json.dump(words, open(out, "w"), ensure_ascii=False, indent=1)
    print(name, "->", " ".join(w["word"] for w in words))
    print("  слов:", len(words), "->", out, flush=True)

"""Пословные тайминги хуков ролика 24 — тот же метод, что и в align24.py.
В hook1 silencedetect(-34dB/0.12с) не нашёл ни одной паузы: дубль короткий
и идёт сплошняком, поэтому отрезок один — весь файл целиком."""
import json
import subprocess
import whisper

BASE = "/Users/vladimirkalajcidi/reels_good/videos/24"
BUILD = "/Users/vladimirkalajcidi/reels_good/build"
HOOKS = {"hook1": [(0.000, 6.957)]}

model = whisper.load_model("large-v3")
for name, spans in HOOKS.items():
    wav = f"{BUILD}/assets/_align24_{name}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", f"{BASE}/{name}.mov", "-vn", "-ac", "1", "-ar", "16000",
                    wav], check=True)
    audio = whisper.load_audio(wav)
    words = []
    for (a, b) in spans:
        r = model.transcribe(audio[int(a * 16000):int(b * 16000)], language="ru",
                             word_timestamps=True, fp16=False,
                             condition_on_previous_text=False)
        got = [w for s in r["segments"] for w in s.get("words", [])]
        if not got:
            continue
        lo, hi = got[0]["start"], got[-1]["end"]
        k = (b - a) / max(1e-6, hi - lo)
        for w in got:
            words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                              end=round(a + (w["end"] - lo) * k, 3),
                              word=w["word"].strip()))
    json.dump(words, open(f"{BASE}/words_{name}.json", "w"), ensure_ascii=False, indent=1)
    print(name, "->", " ".join(w["word"] for w in words))

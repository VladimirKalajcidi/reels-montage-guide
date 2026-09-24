"""Пословные тайминги альтернативных хуков ролика 33.

`silencedetect` (-34dB, 0.12с) не нашёл в обоих дублях ни одной паузы: речь
идёт сплошняком от первого кадра до конца файла. Поэтому отрезок один на дубль,
границы — начало файла и конец записи; внутри whisper large-v3 растянут
на измеренный интервал, как в align33.py.
"""
import json
import subprocess
import whisper

BASE = "/Users/vladimirkalajcidi/reels_good"
JOBS = [
    ("hook1", [(0.000, 3.000)]),
    ("hook2", [(0.000, 2.718)]),
]

model = whisper.load_model("large-v3")

for name, spans in JOBS:
    src = f"{BASE}/videos/33/{name}.mov"
    wav = f"{BASE}/build/assets/_align33_{name}.wav"
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
            print(f"!! пусто {name} {a}-{b}", flush=True)
            continue
        lo, hi = got[0]["start"], got[-1]["end"]
        k = (b - a) / max(1e-6, hi - lo)
        for w in got:
            words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                              end=round(a + (w["end"] - lo) * k, 3),
                              word=w["word"].strip(),
                              p=round(w.get("probability", 0), 2)))
        print(f"{name} {a:6.2f}-{b:6.2f} k={k:.3f}  "
              + " ".join(f"{w['word'].strip()}[{w.get('probability',0):.2f}]" for w in got),
              flush=True)
    out = f"{BASE}/videos/33/words_{name}.json"
    json.dump(words, open(out, "w"), ensure_ascii=False, indent=1)
    print("слов:", len(words), "->", out, flush=True)

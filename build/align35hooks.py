"""Пословные тайминги альтернативных хуков ролика 35.

Границы отрезков — `silencedetect` (-34dB, 0.10с) по `_align35_hook*.wav`.
Паузы короче 0.18с считаются внутрифразовыми и слиты, как в align35.py.

* hook1 — речь 0.113-6.983, дальше 0.124с хвостовой тишины: дубль не обрезан
  записью, последнее слово целое;
* hook2 — речь идёт до самого конца файла (6.737), хвостовой тишины нет.
  Целость последнего слова проверяется по огибающей отдельно.

Внутри отрезка whisper large-v3 растянут на измеренный интервал.
"""
import json
import subprocess
import whisper

BASE = "/Users/vladimirkalajcidi/reels_good"
JOBS = [
    ("hook1", [(0.113, 2.953), (3.078, 6.983)]),
    ("hook2", [(0.186, 2.274), (2.826, 5.028), (5.175, 6.737)]),
]

model = whisper.load_model("large-v3")

for name, spans in JOBS:
    src = f"{BASE}/videos/35/{name}.mov"
    wav = f"{BASE}/build/assets/_align35_{name}.wav"
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
              + " ".join(w["word"].strip() for w in got), flush=True)
    out = f"{BASE}/videos/35/words_{name}.json"
    json.dump(words, open(out, "w"), ensure_ascii=False, indent=1)
    print("слов:", len(words), "->", out, flush=True)

"""Пословные тайминги альтернативных хуков ролика 37.

Границы отрезков — `silencedetect` (-34dB, 0.10с) по `_align37_hook*.wav`,
конец речи уточнён по огибающей (10мс RMS):
* hook1 — речь 0.120-3.150; внутренняя пауза 1.897-1.998 (0.101с) считана
  внутрифразовой и слита. Дальше до конца файла (3.325) идёт не тишина,
  а дыхание на -27...-37 dB: **хвостовой тишины в дубле нет**, поэтому пауза
  перед телом добирается фейдом и длиной блока, а не хвостом дубля;
* hook2 — речь 0.180-4.365, дальше 0.19с настоящей тишины.
Внутри отрезка whisper large-v3 растянут на измеренный интервал.
"""
import json
import subprocess
import whisper

BASE = "/Users/vladimirkalajcidi/reels_good"
JOBS = [
    ("hook1", [(0.120, 3.150)]),
    ("hook2", [(0.180, 4.365)]),
]

model = whisper.load_model("large-v3")

for name, spans in JOBS:
    src = f"{BASE}/videos/37/{name}.mov"
    wav = f"{BASE}/build/assets/_align37_{name}.wav"
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
    out = f"{BASE}/videos/37/words_{name}.json"
    json.dump(words, open(out, "w"), ensure_ascii=False, indent=1)
    print("слов:", len(words), "->", out, flush=True)

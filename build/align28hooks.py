"""Пословные тайминги альтернативных хуков ролика 28.

Тот же приём, что и в `align28.py`: границы речи снимаются
`ffmpeg silencedetect` (-34dB / 0.12с), отрезок транскрибируется отдельно
и выдача растягивается на измеренный интервал. Короткие отрезки не дробим —
на куске меньше ~2с модель добивает фразу мусором (см. шапку align28.py).
"""
import json
import subprocess
import whisper

V = "/Users/vladimirkalajcidi/reels_good/videos/28"
B = "/Users/vladimirkalajcidi/reels_good/build"

# hook1: silencedetect не нашёл ни одной паузы -34dB/0.12с — дубль идёт сплошняком
# hook2: тишина 0…0.151, дальше речь до конца файла
#
# Правая граница отрезка взята не по длине файла, а по огибающей: в обоих
# дублях после последнего слова остаётся хвост тишины (hook1 — 0.06с,
# hook2 — 0.09с). Если растянуть выдачу на весь файл, тайминги уезжают
# на 1.3% — к концу хука это полтора кадра, а субтитры ставятся кадр в кадр.
SPANS = {
    "hook1": [(0.000, 4.240)],
    "hook2": [(0.151, 2.672)],
}

model = whisper.load_model("large-v3")
for name, spans in SPANS.items():
    wav = f"{B}/assets/_align28_{name}.wav"
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

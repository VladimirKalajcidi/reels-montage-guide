"""Пословные тайминги ролика 28 по речевым отрезкам: каждый отрезок между паузами
транскрибируется отдельно, поэтому дрейф выравнивания не копится по ролику.

Границы отрезков — из ffmpeg silencedetect (-34dB, 0.12с), то есть измерены
по звуку, а не предсказаны моделью. Соседние отрезки с зазором <0.15с слиты:
такой зазор — это придыхание внутри фразы, а не пауза между предложениями.

Два отрезка пришлось слить постфактум (MERGED ниже): на коротком куске
модель дописывает лишнее слово или обрывает фразу. Отдельно взятые
9.96-11.71 и 11.89-13.31 дали «...находите один» + «там например тысячи
рублей», слитые — верное «...находите там например 1000 рублей».
То же с хвостом: 43.66-45.44 обрывался на «у меня...», а 45.57-45.87
декодировался как «профит» вместо «в профиле».
"""
import json
import subprocess
import whisper

SRC = "/Users/vladimirkalajcidi/reels_good/videos/28/source.mov"
WAV = "/Users/vladimirkalajcidi/reels_good/build/assets/_align28.wav"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/28/words.json"

SPANS = [
    (0.168, 4.790), (5.254, 9.502), (9.960, 11.713), (11.892, 13.308),
    (13.807, 17.584), (17.962, 21.312), (21.744, 23.713), (24.123, 28.773),
    (29.280, 31.678), (32.040, 33.839), (34.522, 39.520), (39.872, 43.200),
    (43.659, 45.441), (45.573, 45.873),
]

# отрезки, которые считаются слитыми: короткий кусок модель добивает мусором
MERGED = [(9.960, 13.308), (43.659, 45.873)]

subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                "-vn", "-ac", "1", "-ar", "16000", WAV], check=True)
audio = whisper.load_audio(WAV)
model = whisper.load_model("large-v3")

words = []
drop = {x for m in MERGED for x in m}
SPANS = [s for s in SPANS if s[0] not in drop and s[1] not in drop] + MERGED
SPANS.sort()

for (a, b) in SPANS:
    seg = audio[int(a * 16000):int(b * 16000)]
    r = model.transcribe(seg, language="ru", word_timestamps=True,
                         fp16=False, condition_on_previous_text=False)
    got = [w for s in r["segments"] for w in s.get("words", [])]
    if not got:
        print(f"!! пусто {a}-{b}", flush=True)
        continue
    lo = got[0]["start"]
    hi = got[-1]["end"]
    k = (b - a) / max(1e-6, hi - lo)
    for w in got:
        words.append(dict(start=round(a + (w["start"] - lo) * k, 3),
                          end=round(a + (w["end"] - lo) * k, 3),
                          word=w["word"].strip()))
    print(f"{a:6.2f}-{b:6.2f}  k={k:.3f}  " + " ".join(w["word"].strip() for w in got), flush=True)

json.dump(words, open(OUT, "w"), ensure_ascii=False, indent=1)
print("слов:", len(words), "->", OUT)

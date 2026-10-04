"""Ролик 57 v2: перевыравнивание пословных таймингов по кускам речи.
Главный проход whisper-1 по всему файлу местами сдвинут на 0.2–0.3с («Тогда» 18.06 при тишине до 18.33, «2000» 16.52
при тишине 16.36–16.68). У вырезки whisper прибивает первое слово к началу файла — поэтому каждый кусок начинается
за 0.04с до звука (по огибающей env157: RMS 10 мс), тогда прибитое начало совпадает с реальным.
Куски — речь между паузами ≥0.10с (RMS < −33 дБ), склеенные до ≥1.2с. Слова куска сопоставляются со словами
words.json по порядку (difflib), текст остаётся из words.json, время — из куска. → words.json, лог realign157.log."""
import json, os, subprocess, difflib, re
import numpy as np
import env157 as e

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/57"
P = f"{ROOT}/words.json"
WAV = f"{ROOT}/work/voice16k.wav"
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.dirname(os.path.abspath(__file__)) + "/.env")
       if l.startswith("OPENAI_API_KEY")][0]
db = e.db
N = len(db)
loud = db > -33
# паузы ≥ 0.10с
segs, k = [], 0
while k < N:
    if loud[k]:
        s = k
        while k < N and not (not loud[k:k + 10].any()):
            k += 1
        segs.append((s / 100, k / 100))
    k += 1
chunks = []
for a, b in segs:
    if chunks and (b - chunks[-1][0] < 1.2 or a - chunks[-1][1] < 0.0):
        chunks[-1] = (chunks[-1][0], b)
    else:
        chunks.append((a, b))
W = json.load(open(P))
norm = lambda s: re.sub(r"[^0-9a-zа-я]", "", s.lower().replace("ё", "е"))
log = open(__file__.replace(".py", ".log"), "w")
got = []
for a, b in chunks:
    a0, b0 = max(0, a - 0.04), min(N / 100, b + 0.06)
    wav = f"{ROOT}/work/_rc.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a0:.2f}", "-to", f"{b0:.2f}", "-i", WAV, wav], check=True)
    r = subprocess.run(["curl", "-s", "https://api.openai.com/v1/audio/transcriptions", "-H", f"Authorization: Bearer {KEY}",
                        "-F", f"file=@{wav}", "-F", "language=ru", "-F", "model=whisper-1", "-F", "response_format=verbose_json",
                        "-F", "timestamp_granularities[]=word"], capture_output=True, text=True).stdout
    ws = [dict(word=w["word"], start=w["start"] + a0, end=w["end"] + a0) for w in json.loads(r)["words"]
          if any(c.isalnum() for c in w["word"])]
    log.write(f"[{a0:.2f}-{b0:.2f}] " + " ".join(f"{w['word']}[{w['start']:.2f}]" for w in ws) + "\n")
    got += ws
# числа: whisper пишет «тысячу» / «1000» по-разному — сравниваем по нормализованному тексту, числа ≈ любое число
key = lambda s: "#" if norm(s).isdigit() else norm(s)
sm = difflib.SequenceMatcher(None, [key(w["word"]) for w in W], [key(w["word"]) for w in got], autojunk=False)
moved = 0
for blk in sm.get_matching_blocks():
    for j in range(blk.size):
        w, g = W[blk.a + j], got[blk.b + j]
        if abs(w["start"] - g["start"]) > 0.06:
            log.write(f"  {blk.a + j:3d} {w['word']:<16} {w['start']:6.2f} -> {g['start']:6.2f}\n")
            moved += 1
        w["start"], w["end"] = round(g["start"], 3), round(g["end"], 3)
unmatched = [i for i in range(len(W)) if not any(b.a <= i < b.a + b.size for b in sm.get_matching_blocks())]
log.write(f"не сопоставлены: {[(i, W[i]['word'], W[i]['start']) for i in unmatched]}\n")
# нахлёсты: конец предыдущего не позже начала следующего
for i in range(len(W) - 1):
    if W[i]["end"] > W[i + 1]["start"]:
        W[i]["end"] = W[i + 1]["start"]
json.dump(W, open(P + ".realign", "w"), ensure_ascii=False, indent=1)
print("кусков", len(chunks), "сдвинуто >0.06с:", moved, "не сопоставлены:", [(i, W[i]["word"]) for i in unmatched])

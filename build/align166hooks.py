"""Пословные тайминги хуков ролика 66 (слот 166), блок 7. Тот же способ, что у основного дубля (align166 + realign166):
whisper-1 по всему хуку (work/hook<N>.json) — текст; время — повторной расшифровкой кусков речи между паузами (RMS 10 мс,
< −33 дБ ≥ 0.10с), каждый кусок с началом за 0.04с до звука (урок 57 v2). Текст сверен gpt-4o-transcribe (work/hook<N>_4o.txt) —
совпал слово в слово. → videos/66/hook<N>_words.json, лог align166hooks.log."""
import json, os, subprocess, difflib, re, wave
import numpy as np

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/66"
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.dirname(os.path.abspath(__file__)) + "/.env")
       if l.startswith("OPENAI_API_KEY")][0]
norm = lambda s: re.sub(r"[^0-9a-zа-я]", "", s.lower().replace("ё", "е"))
key = lambda s: "#" if norm(s).isdigit() else norm(s)
log = open(__file__.replace(".py", ".log"), "w")


def envelope(p):
    w = wave.open(p)
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    hop = 160   # 10 мс при 16 кГц — не номер слота
    return 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))[::hop] + 1e-6)


for h in ("hook1",):   # hook2.mov в папке нет
    wav = f"{ROOT}/work/{h}_16k.wav"
    W = [dict(word=w["word"].strip(), start=round(w["start"], 3), end=round(w["end"], 3))
         for w in json.load(open(f"{ROOT}/work/{h}.json"))["words"] if any(c.isalnum() for c in w["word"])]
    db = envelope(wav); N = len(db); loud = db > -33
    segs, k = [], 0
    while k < N:
        if loud[k]:
            s = k
            while k < N and loud[k:k + 10].any():
                k += 1
            segs.append((s / 100, k / 100))
        k += 1
    chunks = []
    for a, b in segs:
        if chunks and b - chunks[-1][0] < 1.2:
            chunks[-1] = (chunks[-1][0], b)
        else:
            chunks.append((a, b))
    got = []
    for a, b in chunks:
        a0, b0 = max(0, a - 0.04), min(N / 100, b + 0.06)
        cut = f"{ROOT}/work/_rc_{h}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a0:.2f}", "-to", f"{b0:.2f}", "-i", wav, cut], check=True)
        r = subprocess.run(["curl", "-s", "https://api.openai.com/v1/audio/transcriptions", "-H", f"Authorization: Bearer {KEY}",
                            "-F", f"file=@{cut}", "-F", "language=ru", "-F", "model=whisper-1", "-F", "response_format=verbose_json",
                            "-F", "timestamp_granularities[]=word"], capture_output=True, text=True).stdout
        ws = [dict(word=w["word"], start=w["start"] + a0, end=w["end"] + a0) for w in json.loads(r)["words"]
              if any(c.isalnum() for c in w["word"])]
        log.write(f"{h} [{a0:.2f}-{b0:.2f}] " + " ".join(f"{w['word']}[{w['start']:.2f}]" for w in ws) + "\n")
        got += ws
    sm = difflib.SequenceMatcher(None, [key(w["word"]) for w in W], [key(w["word"]) for w in got], autojunk=False)
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            w, g = W[blk.a + j], got[blk.b + j]
            if abs(w["start"] - g["start"]) > 0.06:
                log.write(f"  {h} {blk.a + j:2d} {w['word']:<14} {w['start']:5.2f} -> {g['start']:5.2f}\n")
            w["start"], w["end"] = round(g["start"], 3), round(g["end"], 3)
    for w0, w1 in zip(W, W[1:]):
        if w0["end"] > w1["start"]:
            w0["end"] = w1["start"]
    json.dump(W, open(f"{ROOT}/{h}_words.json", "w"), ensure_ascii=False, indent=1)
    print(h, len(W), "слов:", " ".join(f"{w['word']}[{w['start']:.2f}]" for w in W))

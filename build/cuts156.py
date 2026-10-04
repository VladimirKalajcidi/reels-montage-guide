"""Ролик 56: повторная пословная расшифровка вырезок (whisper-1) вокруг слов, которые
whisper сжал/растянул на полном файле, + провалы огибающей RMS 10 мс. Подробности — cuts156.log."""
import json, os, subprocess, wave
import numpy as np

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/56"
WAV = f"{ROOT}/voice16k.wav"
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.join(os.path.dirname(__file__), ".env"))
       if l.startswith("OPENAI_API_KEY")][0]
WIN = [(2.0, 4.6, "многоугольника на клетке | Итак"), (8.0, 10.4, "сделаем | 1 1 5"),
       (12.9, 14.6, "Теперь записываем"), (15.0, 18.9, "друг под другим по порядку | А в конце ещё раз"),
       (23.2, 25.3, "результаты | Итак мы получаем 70"), (30.2, 32.7, "33 | берём модуль и делим"),
       (38.3, 40.4, "формы | Главное знать координаты"), (40.1, 42.4, "по порядку вдоль границы"),
       (49.3, 51.9, "есть в ОГЭ | Если хотите чтобы ваш"), (52.2, 53.6, "быстро и без ошибок"),
       (55.5, 57.9, "и я проведу с ним первое занятие")]

w = wave.open(WAV)
x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
hop = 160
db = 20 * np.log10(np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, "same"))[::hop] + 1e-6)


def dips(a, b):
    i0, i1 = int(a * 100), int(b * 100)
    s = db[i0:i1]
    return [(round((i0 + k) / 100, 2), round(float(s[k]), 1)) for k in range(2, len(s) - 2)
            if s[k] == s[k - 2:k + 3].min() and s[k] < s[k - 2:k + 3].max() - 4]


def cut(a, b, k):
    p = f"{ROOT}/work/cut156_{k}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-t", str(b - a), "-i", WAV, p], check=True)
    r = subprocess.run(["curl", "-s", "https://api.openai.com/v1/audio/transcriptions",
                        "-H", f"Authorization: Bearer {KEY}", "-F", f"file=@{p}", "-F", "model=whisper-1",
                        "-F", "language=ru", "-F", "response_format=verbose_json",
                        "-F", "timestamp_granularities[]=word"], capture_output=True, text=True).stdout
    return [(d["word"], round(a + d["start"], 2), round(a + d["end"], 2)) for d in json.loads(r).get("words", [])]


if __name__ == "__main__":
    log = []
    for k, (a, b, what) in enumerate(WIN):
        log.append(f"== {a}-{b} {what}")
        log.append("   cut: " + "  ".join(f"{t}[{s:.2f}-{e:.2f}]" for t, s, e in cut(a, b, k)))
        log.append("   dips: " + "  ".join(f"{t:.2f}:{v:.0f}" for t, v in dips(a, b)))
    open(__file__.replace(".py", ".log"), "w").write("\n".join(log) + "\n")
    print("\n".join(log))

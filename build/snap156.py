"""Подгонка границ слов ролика 56 под измеренные паузы.

Паузы — ffmpeg silencedetect (-34dB, 0.12с) по исходнику. Для каждой паузы:
последнее слово перед ней кончается на silence_start, первое после — начинается
на silence_end. Правка только если сдвиг ≤0.35с и точка остаётся внутри
соседних слов. Подробности пишутся в snap156.log.
"""
import json, re, subprocess

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/56"
SRC = f"{ROOT}/source.mov"
WORDS = f"{ROOT}/words.json"

# 9 слов с нулевой / мини-длительностью (0–0.04с) — привязаны вручную
# по провалам огибающей RMS 10мс (align156.py) и речевому контексту.
ENV = [
    # 0-индексы в итоговом words.json (157 слов после склейки «70»+«ти» → «70-ти» на прежних позициях 77+78)
    (7,   "клетке",  3.35,  3.65),
    (21,  "1",       9.40,  9.52),
    (44,  "конце",   17.10, 17.26),
    (80,  "берём",   31.14, 31.44),
    (99,  "Главное", 39.02, 39.22),
    (107, "порядку", 41.34, 41.52),
    (134, "чтобы",   51.40, 51.60),
    (141, "без",     52.88, 53.02),
    (152, "ним",     56.30, 56.56),
]


def silences():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", SRC, "-vn", "-af",
                        "silencedetect=noise=-34dB:d=0.12", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r)]
    return [(a, b) for a, b in zip(st, en) if a > 0.01]


def main():
    words = json.load(open(WORDS))
    log, n = [], 0
    for a, b in silences():
        i = max(k for k, w in enumerate(words) if (w["start"] + w["end"]) / 2 < (a + b) / 2)
        if i + 1 >= len(words):
            continue
        w0, w1 = words[i], words[i + 1]
        if abs(w0["end"] - a) <= 0.35 and w0["start"] < a:
            if abs(w0["end"] - a) > 0.02:
                log.append(f"конец  «{w0['word']}» {w0['end']:.3f} -> {a:.3f}"); n += 1
            w0["end"] = round(a, 3)
        if abs(w1["start"] - b) <= 0.35 and b < w1["end"]:
            if abs(w1["start"] - b) > 0.02:
                log.append(f"начало «{w1['word']}» {w1['start']:.3f} -> {b:.3f}"); n += 1
            w1["start"] = round(b, 3)
    sil = silences()
    for w0, w1 in zip(words, words[1:]):
        if w1["start"] < w0["end"]:
            b = next((b for a, b in sil if abs(a - w0["end"]) < 0.02), None)
            if b is not None and b < w1["end"]:
                log.append(f"нахлёст «{w1['word']}» {w1['start']:.3f} -> {b:.3f}"); n += 1
                w1["start"] = round(b, 3)
    for k, txt, s, e in ENV:
        w = words[k]
        assert w["word"] == txt, (k, w["word"], txt)
        w["start"], w["end"] = s, e
        if k > 0:
            words[k - 1]["end"] = min(words[k - 1]["end"], s)
        if k + 1 < len(words) and words[k + 1]["start"] < e:
            words[k + 1]["start"] = e
        log.append(f"огибающая «{txt}» -> {s:.2f}–{e:.2f}"); n += 1
    for a, b in sil:
        for w in words:
            if a < w["start"] < b - 0.05:
                log.append(f"ВНУТРИ ПАУЗЫ «{w['word']}» {w['start']:.3f} (пауза {a:.3f}–{b:.3f})")
    for w0, w1 in zip(words, words[1:]):
        assert w0["end"] <= w1["start"] + 1e-6, (w0, w1)
    json.dump(words, open(WORDS, "w"), ensure_ascii=False, indent=1)
    open(__file__.replace(".py", ".log"), "w").write("\n".join(log) + "\n")
    print("границ уточнено:", n)


if __name__ == "__main__":
    main()

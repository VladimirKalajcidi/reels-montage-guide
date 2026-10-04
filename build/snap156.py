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

# Начала/концы по повторной расшифровке вырезок (cuts156.log) и провалам огибающей RMS 10 мс:
# whisper на полном файле растягивал одно слово и сжимал соседа («на» 0.72с / «клетке» 0с,
# «по» 0.62с / «порядку» 0.08с, «ещё» 0.04с, «модуль» 0.10с, «первое» 0.04с, «ОГЭ» 0.10с).
# Индексы — words.json после align156 (156 слов).
ENV = [
    (5, "многоугольника", 1.90, 2.62), (6, "на", 2.62, 2.76), (7, "клетке", 2.76, 3.34),
    (8, "Итак", 3.78, 4.06), (9, "сначала", 4.14, 4.38),
    (21, "1", 9.30, 9.58), (22, "1", 9.60, 9.98),
    (32, "Теперь", 13.32, 13.66), (33, "записываем", 13.70, 14.02),
    (40, "по", 15.74, 15.86), (41, "порядку", 15.88, 16.40),
    (42, "А", 16.70, 16.84), (43, "в", 16.86, 16.92), (44, "конце", 16.92, 17.10),
    (45, "ещё", 17.10, 17.26), (46, "раз", 17.26, 17.44), (47, "повторяем", 17.46, 17.88),
    (61, "Итак", 23.76, 24.02), (62, "мы", 24.08, 24.18), (63, "получаем", 24.22, 24.62),
    (80, "берём", 30.96, 31.20), (81, "модуль", 31.20, 31.58), (82, "и", 31.62, 31.86),
    (98, "Главное", 38.82, 39.14), (99, "знать", 39.16, 39.30),
    (105, "по", 40.84, 40.94), (106, "порядку", 40.96, 41.38), (107, "вдоль", 41.40, 41.60),
    (129, "в", 49.92, 50.00), (130, "ОГЭ", 50.00, 50.34),
    (133, "чтобы", 51.32, 51.50), (134, "ваш", 51.52, 51.74),
    (139, "и", 52.72, 52.80), (140, "без", 52.82, 52.92), (141, "ошибок", 52.94, 53.42),
    (150, "с", 56.08, 56.18), (151, "ним", 56.18, 56.30), (152, "первое", 56.32, 56.62),
    (153, "занятие", 56.64, 57.08),
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

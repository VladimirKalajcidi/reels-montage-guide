"""Границы слов ролика 66: база — realign166 (куски речи, урок 57 v2) → words.json.realign; поправки по огибающей
(env166.py, RMS 10 мс) там, где куски дали неверное время; проверки: нахлёсты, слова внутри пауз (урок 43).
silencedetect −28 dB, 0.12с — как у ролика 60. Лог — snap166.log."""
import json, re, subprocess

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/66"
SRC = f"{ROOT}/source.mov"
WORDS = f"{ROOT}/words.json"

ENV = [  # (индекс, слово, начало, конец)
    (20, "или", 8.28, 8.42), (21, "поздно", 8.48, 8.72), (22, "напечатает", 8.72, 9.34),   # смычка «п» 8.42–8.48
    (36, "на", 13.71, 13.84), (37, "английском", 13.84, 14.42),
    (38, "при", 14.48, 14.61), (39, "26", 14.62, 15.11),                                   # смычка «п» 14.42–14.48;                            # куски дали обоим 13.87
    (46, "к", 17.70, 17.78), (47, "29", 17.78, 18.66), (48, "септиллионам", 18.68, 19.34),  # провал 18.66–18.72
    (51, "Гамлете", 19.93, 20.28),                                                        # «в Г» 19.84–19.92
    (97, "с", 37.34, 37.42), (98, "ним", 37.42, 37.52),                                    # куски дали обоим 37.38
]


def silences():
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", SRC, "-vn", "-af",
                        "silencedetect=noise=-28dB:d=0.12", "-f", "null", "-"], capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r)]
    return list(zip(st, en))


def main():
    words = json.load(open(WORDS + ".realign"))
    log = []
    for k, txt, s, e in ENV:
        w = words[k]
        assert w["word"] == txt, (k, w["word"], txt)
        log.append(f"огибающая «{txt}» {w['start']:.2f} -> {s:.2f}–{e:.2f}")
        w["start"], w["end"] = s, e
        if k > 0:
            words[k - 1]["end"] = min(words[k - 1]["end"], s)
    for w0, w1 in zip(words, words[1:]):
        if w0["end"] > w1["start"]:
            w0["end"] = w1["start"]
    bad = []
    for a, b in silences():
        for w in words:
            if a < w["start"] < b - 0.05:
                bad.append(f"ВНУТРИ ПАУЗЫ «{w['word']}» {w['start']:.3f} (пауза {a:.3f}–{b:.3f})")
    for w0, w1 in zip(words, words[1:]):
        assert w0["end"] <= w1["start"] + 1e-6 and w0["start"] < w1["start"], (w0, w1)
    log += bad
    json.dump(words, open(WORDS, "w"), ensure_ascii=False, indent=1)
    open(__file__.replace(".py", ".log"), "w").write("\n".join(log) + "\n")
    print("поправок:", len(ENV), "внутри пауз:", bad)


if __name__ == "__main__":
    main()

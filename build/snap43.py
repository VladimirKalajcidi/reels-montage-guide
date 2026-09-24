"""Подгонка границ слов ролика 10 под измеренные паузы.

Паузы — ffmpeg silencedetect (-34dB, 0.12с) по исходнику. Для каждой паузы:
последнее слово перед ней кончается на silence_start, первое после — начинается
на silence_end. Правка только если сдвиг ≤0.35с и точка остаётся внутри
соседних слов — это уточнение границы, не переразметка речи.
Подробности пишутся в snap43.log.
"""
import json, re, subprocess

ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/10"
SRC = f"{ROOT}/source.mov"
WORDS = f"{ROOT}/words.json"


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
        # слово, чья середина до паузы, и следующее
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
    words[0]["start"] = max(words[0]["start"], 0.19)
    json.dump(words, open(WORDS, "w"), ensure_ascii=False, indent=1)
    open(__file__.replace(".py", ".log"), "w").write("\n".join(log) + "\n")
    print("границ уточнено:", n)


if __name__ == "__main__":
    main()

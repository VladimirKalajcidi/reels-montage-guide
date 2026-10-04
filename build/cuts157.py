"""Вырезки ролика 57 для сверки спорных слов: две модели на каждый кусок (текст) + whisper-1
пословно (время — со смещением вырезки) → cuts157.log, work/<имя>.json."""
import subprocess, os, json
ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/57/work"
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.dirname(os.path.abspath(__file__)) + "/.env") if l.startswith("OPENAI_API_KEY")][0]
CUTS = {"c10": (11.0, 12.6), "c11": (17.6, 19.2), "c12": (33.3, 34.9), "c13": (43.5, 45.2),
        "c14": (8.9, 10.5), "c15": (4.7, 6.2), "c16": (47.5, 49.3), "c17": (19.6, 22.6), "c18": (15.2, 17.7), "c19": (36.3, 38.9), "c20": (22.1, 24.5)}


def api(wav, *extra):
    return subprocess.run(["curl", "-s", "https://api.openai.com/v1/audio/transcriptions", "-H", f"Authorization: Bearer {KEY}",
                           "-F", f"file=@{wav}", "-F", "language=ru", *extra], capture_output=True, text=True).stdout.strip()


for name, (a, b) in CUTS.items():
    wav = f"{ROOT}/{name}.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a), "-to", str(b), "-i", f"{ROOT}/voice16k.wav", wav], check=True)
    for m in ("gpt-4o-transcribe", "whisper-1"):
        print(name, a, b, m, "|", api(wav, "-F", f"model={m}", "-F", "response_format=text"), flush=True)
    d = json.loads(api(wav, "-F", "model=whisper-1", "-F", "response_format=verbose_json", "-F", "timestamp_granularities[]=word"))
    json.dump(d, open(f"{ROOT}/{name}.json", "w"), ensure_ascii=False)
    print(name, "слова:", " ".join(f"{w['word']}[{w['start'] + a:.2f}-{w['end'] + a:.2f}]" for w in d["words"]), flush=True)

"""Вырезки ролика 66 для сверки спорных слов: две модели на каждый кусок (текст) + whisper-1
пословно (время — со смещением вырезки) → cuts166.log, work/<имя>.json."""
import subprocess, os, json
ROOT = "/Users/vladimirkalajcidi/reels_challenge/videos/66/work"
KEY = [l.split("=", 1)[1].strip() for l in open(os.path.dirname(os.path.abspath(__file__)) + "/.env") if l.startswith("OPENAI_API_KEY")][0]
CUTS = {"c1": (25.4, 28.1), "c2": (22.3, 23.9), "c3": (0.0, 0.9), "c4": (3.4, 4.3), "c5": (12.4, 13.3),
        "c6": (19.3, 20.7), "c7": (32.4, 33.3), "c8": (23.6, 24.6)}


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

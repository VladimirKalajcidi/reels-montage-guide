"""Звук ролика 2: синтез SFX + микс с голосом. Уровни — delivery-specs.md §5.
Синтез повторяет sfx.py один в один (уровни трогать нельзя — они уже правлены по слуху),
но модуль самостоятельный, чтобы сборки разных роликов не цеплялись друг за друга.
"""
import os, sys, json, subprocess, wave, importlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SB = importlib.import_module(os.environ.get("SB", "storyboard2"))
SHOTS = SB.SHOTS

SR = 48000


def env(n, a, d, curve=2.0):
    at = int(SR * a); dc = max(1, n - at)
    e = np.concatenate([np.linspace(0, 1, at) ** 0.6,
                        np.linspace(1, 0, dc) ** curve])
    return e[:n]


def low_whoosh(dur=0.42):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, n)
    a = np.exp(-2 * np.pi * 180 / SR)          # однополюсный ФНЧ ~180 Гц
    y = np.zeros(n); prev = 0.0
    for i in range(n):
        prev = a * prev + (1 - a) * noise[i]
        y[i] = prev
    f0, f1 = 140.0, 34.0
    ph = 2 * np.pi * np.cumsum(f0 * (f1 / f0) ** (t / dur)) / SR
    s = 0.65 * y / (np.abs(y).max() + 1e-9) + 0.45 * np.sin(ph)
    return s * env(n, 0.06, 0.0, 1.8)


def impact(dur=0.55):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    f = 92 * np.exp(-t * 11) + 41
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 7.5)
    rng = np.random.default_rng(11)
    click = rng.normal(0, 1, n) * np.exp(-t * 150) * 0.25
    s = body + click
    return s / (np.abs(s).max() + 1e-9)


def tick(dur=0.10):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, n) * np.exp(-t * 110)
    s += 0.5 * np.sin(2 * np.pi * 1400 * t) * np.exp(-t * 90)
    return s / (np.abs(s).max() + 1e-9)


def read_wav(p):
    w = wave.open(p)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    ch = w.getnchannels()
    return d.reshape(-1, ch) if ch > 1 else d.reshape(-1, 1)

BUILD = os.path.dirname(os.path.abspath(__file__))
TAG = SB.TAG
SRC = SB.SRC
OUT = SB.OUT
VID = f"{BUILD}/assets/_video_{TAG}.mp4"

# ревилы чисел (R5b) и старты каскадов графики (R10) — задаются в раскадровке
NUM_REVEALS = getattr(SB, "NUM_REVEALS", [])
CASCADES = getattr(SB, "CASCADES", [])


def main():
    tmp = f"{BUILD}/assets/_voice_{TAG}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                    "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp], check=True)
    voice = read_wav(tmp)
    N = voice.shape[0]
    peak = np.abs(voice).max()

    bed = np.zeros((N, 2), np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        if i < 0:
            return
        n = min(len(sig), N - i)
        if n <= 0:
            return
        bed[i:i + n, 0] += sig[:n] * gain
        bed[i:i + n, 1] += sig[:n] * gain

    WH, IM, TK = low_whoosh(), impact(), tick()
    for c in [s[0] for s in SHOTS[1:]]:
        add(WH, c, peak * 0.040)          # whoosh стартует на кадре реза
    for t in NUM_REVEALS:
        add(IM, t, peak * 0.070)
    for t in CASCADES:
        for k in range(3):
            add(TK, t + k * 0.085, peak * 0.030)

    # --- музыкальная подложка (delivery-specs §5: тише всего)
    music_path = "/Users/vladimirkalajcidi/reels_good/audios/song1.mp3"
    if os.path.exists(music_path):
        mw = f"{BUILD}/assets/_music_{TAG}.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", music_path,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
        mus = read_wav(mw)
        if mus.shape[1] == 1:
            mus = np.repeat(mus, 2, axis=1)
        # склейка петли с кроссфейдом 0.25с, чтобы шва не было слышно
        xf = int(0.25 * SR)
        core = mus[:len(mus) - xf]
        tail = mus[len(mus) - xf:]
        ramp = np.linspace(0, 1, xf)[:, None]
        loop = core.copy()
        loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
        reps = int(np.ceil(N / len(loop))) + 1
        track = np.tile(loop, (reps, 1))[:N]
        # уровень: ~19 dB ниже RMS голоса
        vr = np.sqrt((voice ** 2).mean()) + 1e-9
        mr = np.sqrt((track ** 2).mean()) + 1e-9
        track *= (vr / mr) * (10 ** (-19 / 20))
        fi, fo = int(0.8 * SR), int(2.0 * SR)
        track[:fi] *= np.linspace(0, 1, fi)[:, None]
        track[-fo:] *= np.linspace(1, 0, fo)[:, None]
        bed += track.astype(np.float32)
        print("музыка: подложена, петля %.2fс" % (len(loop) / SR))

    mix = voice + bed
    m = np.abs(mix).max()
    if m > 0.99:
        mix *= 0.99 / m
    out = f"{BUILD}/assets/_mix_{TAG}.wav"
    w = wave.open(out, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

    mux(VID, out, OUT)
    print("готово:", OUT)


# --------------------------------------------------------------- финальный мукс
# loudnorm в один проход только оценивает уровень и промахивается на ~1 dB.
# Меряем первым проходом и подставляем измеренное — тогда попадаем в -14 LUFS.
FILT = "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"


def measure(wav):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", wav,
                        "-af", FILT + ":print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[r.stderr.rfind("{"):]
    return json.loads(tail[:tail.find("}") + 1])


def mux(vid, wav, dst):
    m = measure(wav)
    af = (FILT + f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
                 f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
                 f":offset={m['target_offset']}:linear=true")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", vid, "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", af, "-ar", str(SR),
                    "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", dst], check=True)


if __name__ == "__main__":
    main()

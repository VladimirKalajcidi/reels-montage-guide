"""Звук ролика 8: синтез SFX + микс с голосом. Уровни — delivery-specs.md §5."""
import os, sys, subprocess, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard8 import SHOTS, DUR
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = "/Users/vladimirkalajcidi/reels_good/videos/8/source.mov"
VID = f"{BUILD}/assets/_video_v8.mp4"
OUT = "/Users/vladimirkalajcidi/reels_good/videos/8/imaginary_edit.mp4"
SR = 48000

# ревилы чисел (R5a/R5b) — из render8.py
NUM_REVEALS = [27.300, 31.460, 33.840, 43.150]
# каскады графики (R10) — старт стаггера
CASCADES = [2.290, 9.700, 11.930, 13.540, 18.800, 21.930, 25.950, 29.140,
            31.040, 33.330, 42.250]


def main():
    tmp = f"{BUILD}/assets/_voice_v8.wav"
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
        add(WH, c, peak * 0.040)
    for t in NUM_REVEALS:
        add(IM, t, peak * 0.070)
    for t in CASCADES:
        for k in range(3):
            add(TK, t + k * 0.085, peak * 0.030)

    music_path = "/Users/vladimirkalajcidi/reels_good/audios/song1.mp3"
    if os.path.exists(music_path):
        mw = f"{BUILD}/assets/_music_v8.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", music_path,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
        mus = read_wav(mw)
        if mus.shape[1] == 1:
            mus = np.repeat(mus, 2, axis=1)
        xf = int(0.25 * SR)
        core = mus[:len(mus) - xf]
        tail = mus[len(mus) - xf:]
        ramp = np.linspace(0, 1, xf)[:, None]
        loop = core.copy()
        loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
        reps = int(np.ceil(N / len(loop))) + 1
        track = np.tile(loop, (reps, 1))[:N]
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
    out = f"{BUILD}/assets/_mix_v8.wav"
    w = wave.open(out, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", VID, "-i", out,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7",
                    "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", OUT], check=True)
    print("готово:", OUT)


if __name__ == "__main__":
    main()

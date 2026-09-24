"""Звук ролика 1: синтез SFX + микс с голосом. Уровни — delivery-specs.md §5."""
import os, sys, subprocess, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard1 import SHOTS, DUR, MONEY

BUILD = os.path.dirname(os.path.abspath(__file__))
SRC = "/Users/vladimirkalajcidi/reels_good2/videos/1/source.mov"
VID = f"{BUILD}/assets/_video1.mp4"
OUT = "/Users/vladimirkalajcidi/reels_good2/videos/1/kvant_edit.mp4"
SR = 48000

NUM_REVEALS = [MONEY["t0"] + 0.05]


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
    a = np.exp(-2 * np.pi * 180 / SR)
    y = np.zeros(n); prev = 0.0
    for i in range(n):
        prev = a * prev + (1 - a) * noise[i]
        y[i] = prev
    f0, f1 = 140.0, 34.0
    ph = 2 * np.pi * np.cumsum(f0 * (f1 / f0) ** (t / dur)) / SR
    sweep = np.sin(ph)
    s = 0.65 * y / (np.abs(y).max() + 1e-9) + 0.45 * sweep
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


def read_wav(p):
    w = wave.open(p)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    ch = w.getnchannels()
    return d.reshape(-1, ch) if ch > 1 else d.reshape(-1, 1)


def main():
    tmp = f"{BUILD}/assets/_voice1.wav"
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

    WH, IM = low_whoosh(), impact()
    cuts = [s[0] for s in SHOTS[1:]]
    for c in cuts:
        add(WH, c, peak * 0.040)
    for t in NUM_REVEALS:
        add(IM, t, peak * 0.070)

    music_path = "/Users/vladimirkalajcidi/reels_good2/audios/song1.mp3"
    if os.path.exists(music_path):
        mw = f"{BUILD}/assets/_music1.wav"
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
    out = f"{BUILD}/assets/_mix1.wav"
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

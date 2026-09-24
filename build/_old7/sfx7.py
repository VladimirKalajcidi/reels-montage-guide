"""Звук ролика 7: голос из source_cut + тихие whoosh/impact/tick, loudnorm."""
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard7 import SRC_CUT, OUT, SHOTS, NUM_REVEALS, CASCADES

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_secretary.mp4"
SR = 48000


def env(n, a, curve=2.0):
    at = int(SR * a)
    dc = max(1, n - at)
    return np.concatenate([np.linspace(0, 1, at) ** 0.6, np.linspace(1, 0, dc) ** curve])[:n]


def low_whoosh(dur=0.42):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1, n)
    a = np.exp(-2 * np.pi * 180 / SR)
    y = np.zeros(n)
    prev = 0.0
    for i in range(n):
        prev = a * prev + (1 - a) * noise[i]
        y[i] = prev
    ph = 2 * np.pi * np.cumsum(140 * (34 / 140) ** (t / dur)) / SR
    s = 0.65 * y / (np.abs(y).max() + 1e-9) + 0.45 * np.sin(ph)
    return s * env(n, 0.06, 1.8)


def impact(dur=0.50):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    f = 92 * np.exp(-t * 11) + 41
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 7.5)
    return s / (np.abs(s).max() + 1e-9)


def tick(dur=0.10):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, n) * np.exp(-t * 110)
    s += 0.5 * np.sin(2 * np.pi * 1400 * t) * np.exp(-t * 90)
    return s / (np.abs(s).max() + 1e-9)


def read_wav(path):
    w = wave.open(path)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    ch = w.getnchannels()
    return d.reshape(-1, ch) if ch > 1 else d.reshape(-1, 1)


def main():
    voice_path = f"{BUILD}/assets/_voice_secretary.wav"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC_CUT,
        "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", voice_path,
    ], check=True)
    voice = read_wav(voice_path)
    n = voice.shape[0]
    peak = np.abs(voice).max()
    bed = np.zeros((n, 2), np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        m = min(len(sig), n - i)
        if i >= 0 and m > 0:
            bed[i:i + m, 0] += sig[:m] * gain
            bed[i:i + m, 1] += sig[:m] * gain

    wh, im, tk = low_whoosh(), impact(), tick()
    for t0, _, _, _ in SHOTS[1:]:
        add(wh, t0, peak * 0.040)
    for t in NUM_REVEALS:
        add(im, t, peak * 0.070)
    for t in CASCADES:
        for k in range(3):
            add(tk, t + k * 0.085, peak * 0.030)

    mix = voice + bed
    m = np.abs(mix).max()
    if m > 0.99:
        mix *= 0.99 / m
    mix_path = f"{BUILD}/assets/_mix_secretary.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()

    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", VID, "-i", mix_path, "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-af", "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", OUT,
    ], check=True)
    print("готово:", OUT)


if __name__ == "__main__":
    main()

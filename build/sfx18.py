"""Звук ролика 18: SFX + тихая музыка + исходная речь без резки."""
import os
import subprocess
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard18 import SHOTS, DUR, NUM_REVEALS, CASCADES, SRC, OUT
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VID = f"{BUILD}/assets/_video_leap.mp4"
# song3 (110с) длиннее ролика (47.7с) — подложка идёт без петли, шва не существует в принципе
MUSIC = "/Users/vladimirkalajcidi/reels_good/audios/song3.mp3"
SR = 48000


def main():
    tmp = f"{BUILD}/assets/_voice_leap.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", SRC,
                    "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp], check=True)
    voice = read_wav(tmp)
    n = voice.shape[0]
    peak = np.abs(voice).max()
    bed = np.zeros((n, 2), np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        if i < 0:
            return
        m = min(len(sig), n - i)
        if m <= 0:
            return
        bed[i:i + m, 0] += sig[:m] * gain
        bed[i:i + m, 1] += sig[:m] * gain

    wh, im, tk = low_whoosh(), impact(), tick()
    for c in [s[0] for s in SHOTS[1:]]:          # whoosh на каждой склейке
        add(wh, c, peak * 0.040)
    for t in NUM_REVEALS:                        # импакт под синим числом
        add(im, t, peak * 0.070)
    for t in CASCADES:                           # тики под каскадом элементов
        for k in range(3):
            add(tk, t + k * 0.085, peak * 0.030)

    if os.path.exists(MUSIC):
        mw = f"{BUILD}/assets/_music_leap.wav"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", MUSIC,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
        mus = read_wav(mw)
        if mus.shape[1] == 1:
            mus = np.repeat(mus, 2, axis=1)
        if len(mus) >= n:
            track = mus[:n].copy()
        else:                                    # короткий трек — петля с кроссфейдом
            xf = int(0.25 * SR)
            core, tail = mus[:len(mus) - xf], mus[len(mus) - xf:]
            ramp = np.linspace(0, 1, xf)[:, None]
            loop = core.copy()
            loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
            track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n]
        vr = np.sqrt((voice ** 2).mean()) + 1e-9
        mr = np.sqrt((track ** 2).mean()) + 1e-9
        track = track * (vr / mr) * (10 ** (-19 / 20))   # -19 dB от RMS голоса
        fi, fo = int(0.8 * SR), int(2.0 * SR)
        track[:fi] *= np.linspace(0, 1, fi)[:, None]
        track[-fo:] *= np.linspace(1, 0, fo)[:, None]
        bed += track.astype(np.float32)

    mix = voice + bed
    mx = np.abs(mix).max()
    if mx > 0.99:
        mix *= 0.99 / mx
    out_wav = f"{BUILD}/assets/_mix_leap.wav"
    w = wave.open(out_wav, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", VID, "-i", out_wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    # loudnorm на этом материале недобирает ~0.9 dB (delivery-specs §5).
                    # Догоняем volume + alimiter; level=disabled обязателен, иначе
                    # автонормализация лимитера выносит уровень и пик.
                    "-af", "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7,"
                           "volume=1.1dB,alimiter=limit=0.80:level=disabled",
                    "-ar", str(SR), "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", OUT], check=True)
    print("готово:", OUT)


if __name__ == "__main__":
    main()

"""Звук ролика 10 («задача Монти Холла»): синтез SFX + музыка + микс с голосом.
Уровни — delivery-specs.md §5. Скопировано с sfx42.py; добавлен soft tick
(0.030) на элементах графики дверей. Слот 43 — см. docstring storyboard43.py."""
import os, sys, subprocess, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard43 import SHOTS, DUR

BUILD = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/vladimirkalajcidi/reels_challenge"
SRC = f"{ROOT}/videos/10/source.mov"
VID = f"{BUILD}/assets/_video_monty.mp4"
OUT = f"{ROOT}/videos/10/monty_hall_edit.mp4"

# impact — синие 2/3 (R5b); tick — каскад дверей, метка выбора, открытие двери, белые числа
IMPACTS = [22.06, 27.36, 33.24]
POST_GAIN_DB = 0.9
TICKS = [3.10, 3.185, 3.27, 8.88, 11.43, 17.64, 26.02]

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


def soft_tick(dur=0.09):
    n = int(SR * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(5)
    s = rng.normal(0, 1, n) * np.exp(-t * 90) * 0.5 + np.sin(2 * np.pi * 1400 * t) * np.exp(-t * 60)
    return s / (np.abs(s).max() + 1e-9)


# v4 (2026-09-24): «вжух» появления текста — по ролику-образцу конкурента. Замер образца:
# шумовой всплеск 5–14 кГц, нарастание ~0.12с → пик → спад ~0.08с (всего ~0.2с), RMS на ~3 дБ
# ниже RMS голоса; стоит на появлении акцентных фраз, примерно раз в 3–4с. Звук синтезирован
# (чужой звук не вырезается). Пик эффекта совпадает с моментом, когда слово всплывает.
TW_GAP = 2.2          # не чаще раза в 2.2с — в среднем ≈3с
TW_REL_DB = -16.0     # v5: RMS эффекта −16 дБ от RMS голоса (v4 был −6 и резкий — замечание автора)
TW_PEAK_AT = 0.03     # где у звука пик — ставим его на t_word + 0.05


def text_whoosh_times():
    """Начала фраз с «вжухом»: первая фраза, фразы с цветным словом, после паузы/реза, затем
    остальные — жадно, с шагом не меньше TW_GAP (выходит ≈ раз в 3с, как у образца)."""
    import render43 as R
    cand = []
    for i, (t0, t1, runs, slot) in enumerate(R.CAPS):
        prio = 0 if any(k in ("red", "teal") for (_, k, _) in runs) else (
               1 if i == 0 or t0 - R.CAPS[i - 1][1] > 0.25 or R._shot_idx(t0) != R._shot_idx(R.CAPS[i - 1][0]) else 2)
        if i == 0:
            prio = -1                          # первая фраза ролика — всегда (как у образца на 0.0с)
        cand.append((prio, R.WORD_T[i][0]))
    chosen = []
    for prio, t in sorted(cand):
        if all(abs(t - c) >= TW_GAP for c in chosen):
            chosen.append(t)
    return sorted(chosen)


def text_whoosh(dur=0.20):
    """v5 (2026-09-24): мягкий «пуф» появления текста — в характере звука двери (soft tick), который
    автору понравился. v4 (шум 5–12 кГц) был слишком резким. Полоса 900–4000 Гц, атака 30мс,
    спад ~120мс, тихий тональный призвук 1.1→1.3 кГц."""
    n = int(SR * dur)
    t = np.arange(n) / SR
    rng = np.random.default_rng(21)
    noise = rng.normal(0, 1, n + 4096)
    F = np.fft.rfft(noise); fr = np.fft.rfftfreq(len(noise), 1 / SR)
    shape = np.exp(-0.5 * ((np.log(np.maximum(fr, 1)) - np.log(1900)) / 0.45) ** 2)   # мягкий горб ~1.9 кГц
    air = np.fft.irfft(F * shape, len(noise))[:n]
    air /= np.abs(air).max() + 1e-9
    f0 = 1100 + 200 * np.clip(t / 0.06, 0, 1)
    tone = np.sin(2 * np.pi * np.cumsum(f0) / SR) * np.exp(-t * 45)
    env = np.where(t < TW_PEAK_AT, np.sin(0.5 * np.pi * t / TW_PEAK_AT) ** 2,
                   np.exp(-(t - TW_PEAK_AT) / 0.045))
    fade = np.minimum(1, (n - np.arange(n)) / (0.015 * SR))
    s = (air * env + 0.35 * tone * np.minimum(1, t / 0.004)) * fade
    return s / (np.sqrt((s[:int(0.10 * SR)] ** 2).mean()) + 1e-9)       # RMS первых 100мс = 1


def read_wav(p):
    w = wave.open(p)
    d = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    ch = w.getnchannels()
    return d.reshape(-1, ch) if ch > 1 else d.reshape(-1, 1)


def main():
    tmp = f"{BUILD}/assets/_voice43.wav"
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
    for t in IMPACTS:
        add(IM, t, peak * 0.070)
    TK = soft_tick()
    for t in TICKS:
        add(TK, t, peak * 0.030)
    TWS = text_whoosh()
    vrms = np.sqrt((voice[np.abs(voice[:, 0]) > 0.02] ** 2).mean())   # RMS голоса по речи
    tw_times = text_whoosh_times()
    for t in tw_times:
        add(TWS, t + 0.05 - TW_PEAK_AT, vrms * 10 ** (TW_REL_DB / 20))
    print("вжух появления текста:", len(tw_times), "шт.:", " ".join(f"{t:.2f}" for t in tw_times))

    music_path = f"{ROOT}/audios/song1.mp3"
    if os.path.exists(music_path):
        mw = f"{BUILD}/assets/_music43.wav"
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
    out = f"{BUILD}/assets/_mix43.wav"
    w = wave.open(out, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()

    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", VID, "-i", out,
                    "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7",
                    "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", OUT], check=True)
    # однопроходный loudnorm недобирает ~0.9 dB (delivery-specs §5) — доводка по замеру
    # готового файла: +0.9 dB -> -14.2 LUFS / -1.6 dBTP (вариант +0.7 dB дал -14.4)
    tmp = OUT.replace(".mp4", ".gain.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", OUT, "-c:v", "copy",
                    "-af", f"volume={POST_GAIN_DB}dB,alimiter=limit=0.75:level=disabled",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp], check=True)
    os.replace(tmp, OUT)
    print("готово:", OUT)


if __name__ == "__main__":
    main()

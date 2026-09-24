"""Ролик 29 («Санкт-Петербургский парадокс»), альтернативные хуки.

Тело с кадра 155 не перерисовывается: готовые кадры читаются из
`assets/_video_st_petersburg.mp4`. Заново собираются только хуковой блок,
цельная звуковая дорожка и обязательная уникализация версии.

    python3 hook29.py h1
    python3 hook29.py h2
"""
import json
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render29 as R
import storyboard29 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/29"
BODY = f"{BUILD}/assets/_video_st_petersburg.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

# Старое вступление занимало планы 0.00–5.14. Первый готовый кадр следующего
# плана (`coinLaunch`) — 155, время 5.1667с. По огибающей первый звук «итак»
# начинается в 5.1715с: рез сохраняет слово целиком и 4.8мс тишины перед ним.
CUT_T = 5.140
CUT_F = math.ceil(CUT_T * FPS - 1e-6)
CUT = CUT_F / FPS
TARGET_LUFS = -14.20

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        frames=155,
        out=f"{VIDEO_DIR}/st_petersburg_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="на бумаге выигрыш в этой игре будет бесконечен, а на самом деле "
              "вы получите максимум пару сотен рублей",
        shots=[
            (0.000, 1.700, "A1", {}),
            (1.700, 155 / 30, "infiniteWin", {}),
        ],
        caps=[
            (0.150, 0.925, [("на бумаге", R_, 50)], "A"),
            (0.925, 1.700, [("выигрыш в этой игре", R_, 44)], "A"),
            (1.700, 2.558, [("будет ", R_, 46), ("бесконечен", S_, 58)], "G"),
            (2.701, 3.272, [("а на самом деле", R_, 46)], "G"),
            (3.272, 4.353, [("вы получите максимум", R_, 44)], "G"),
            (4.353, 155 / 30, [("пару сотен рублей", R_, 48)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # В исходнике 144 кадра; ещё четыре кадра держит графика, не лицо.
        # Это даёт 0.138с до первого звука тела вместо почти нулевой паузы.
        frames=148,
        out=f"{VIDEO_DIR}/st_petersburg_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        post_v="setpts=PTS/1.02,fps=30",
        post_a="atempo=1.02",
        title="это один из самых известных парадоксов теории вероятности "
              "с бесконечным математическим ожиданием",
        shots=[
            (0.000, 1.025, "A1", {}),
            (1.025, 148 / 30, "infiniteWin", {}),
        ],
        caps=[
            (0.098, 1.025, [("это один из самых", R_, 46)], "A"),
            (1.025, 2.013, [("известных парадоксов", R_, 48)], "G"),
            (2.013, 2.981, [("теории вероятности", R_, 48)], "G"),
            (2.981, 3.626, [("с ", R_, 46), ("бесконечным", S_, 54)], "G"),
            (3.626, 148 / 30, [("математическим ожиданием", R_, 44)], "G"),
        ],
    ),
}


def card_rgb(path, t, kind="A1"):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    card = np.asarray(R.source_card(fr, kind))
    return tuple(int(v) for v in card.reshape(-1, 3).mean(axis=0))


def build_video(key, cfg, tmp):
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["src"])
    stocks = R.StockReader()
    enc = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "15",
        "-preset", "medium", "-pix_fmt", "yuv420p", "-color_range", "tv",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:"
                  "matrix_coefficients=1:video_full_range_flag=0", tmp,
    ], stdin=subprocess.PIPE)

    sheet, last_fr, held = [], None, 0
    for fno in range(nf_hook):
        t = fno / FPS
        ok, fr = cap.read()
        if ok:
            last_fr = fr
        else:
            fr = last_fr
            held += 1
        t0, _t1, kind, _prm = R.shot_at(t)
        if not ok and kind in R.FACE_KINDS:
            raise RuntimeError(f"A-roll кончился на лице, кадр {fno}")
        canvas = R.background(kind, fr, t, t - t0, stocks)
        gl = R.graphics_layer(kind, t - t0)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        rgb = canvas.convert("RGB")
        enc.stdin.write(np.asarray(rgb).tobytes())
        if fno % 5 == 0:
            sheet.append((t, np.asarray(rgb)))
    cap.release()
    stocks.release()

    body = cv2.VideoCapture(BODY)
    body.set(cv2.CAP_PROP_POS_FRAMES, CUT_F)
    n_body = 0
    while True:
        ok, frame = body.read()
        if not ok:
            break
        enc.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
        n_body += 1
    body.release()
    enc.stdin.close()
    enc.wait()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg video exited {enc.returncode}")
    print(f"видео: хук {nf_hook} кадров ({held} добито графикой) + тело {n_body}")
    return nf_hook, n_body, sheet


def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook29.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    if start:
        cmd += ["-ss", f"{start:.6f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
    subprocess.run(cmd, check=True)
    data = read_wav(tmp)
    os.remove(tmp)
    return data


def speech_rms(x):
    env = np.abs(x).mean(axis=1)
    speech = env > env.max() * .12
    return float(np.sqrt((x[speech] ** 2).mean()))


def apply_post_v(key, cfg):
    src = f"{BUILD}/assets/_video_st_petersburg_{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_st_petersburg_{key}_post.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
        "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-bsf:v", "h264_metadata=colour_primaries=1:transfer_characteristics=1:"
                  "matrix_coefficients=1:video_full_range_flag=0", dst,
    ], check=True)
    print("правка кадра:", vf)
    return dst


def measure_i(path):
    p = subprocess.run([
        "ffmpeg", "-hide_banner", "-nostats", "-i", path,
        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
        "-f", "null", "-",
    ], capture_output=True, text=True)
    block = p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1]
    data = json.loads(block)
    return float(data["input_i"]), float(data["input_tp"])


def mux(vid, mix_path, cfg):
    def run(post_db):
        af = (f"{cfg['post_a']}," if cfg.get("post_a") else "") + \
             "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        if abs(post_db) > .02:
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.76:level=disabled"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", vid, "-i", mix_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-af", af, "-ar", str(SR), "-c:a", "aac",
            "-b:a", "192k", "-shortest", "-movflags", "+faststart", cfg["out"],
        ], check=True)
        loud, tp = measure_i(cfg["out"])
        print(f"loudness: {loud:.2f} LUFS, TP {tp:.2f}, post {post_db:+.2f} dB")
        return loud, tp

    post = 0.0
    for _ in range(4):
        loud, tp = run(post)
        if abs(loud - TARGET_LUFS) < .15 and tp <= -1.0:
            return
        post += TARGET_LUFS - loud


def build_audio(key, cfg, nf_hook, vid):
    hook = wav_of(cfg["src"])
    body = wav_of(SRC, CUT)
    gain = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook *= gain
    print(f"коррекция RMS хука ×{gain:.3f}")

    n_hook = int(round(nf_hook / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    fade = int(.008 * SR)
    hook[-fade:] *= np.linspace(1, 0, fade)[:, None]
    body = body.copy()
    body[:fade] *= np.linspace(0, 1, fade)[:, None]
    voice = np.vstack([hook, body]).astype(np.float32)

    n = len(voice)
    peak = np.abs(voice).max()
    bed = np.zeros((n, 2), np.float32)

    def add(sig, t, gain_):
        i = int(round(t * SR))
        m = min(len(sig), n - i)
        if i >= 0 and m > 0:
            bed[i:i + m, 0] += sig[:m] * gain_
            bed[i:i + m, 1] += sig[:m] * gain_

    hook_dur = nf_hook / FPS
    dt = hook_dur - CUT
    wh, im, tk = low_whoosh(), impact(), tick()
    for t0, _t1, _kind, _prm in cfg["shots"][1:]:
        add(wh, t0, peak * .040)
    add(wh, hook_dur, peak * .040)
    for t0, _t1, _kind, _prm in SB.SHOTS:
        if t0 > CUT + .01:
            add(wh, t0 + dt, peak * .040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * .070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * .085, peak * .030)
    for t0, _t1, kind, _prm in cfg["shots"]:
        if kind not in R.FACE_KINDS:
            for k in range(3):
                add(tk, t0 + .12 + k * .085, peak * .030)

    music_wav = f"{BUILD}/assets/_music_{os.path.basename(cfg['music']).split('.')[0]}.wav"
    if not os.path.exists(music_wav):
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", cfg["music"], "-ac", "2", "-ar", str(SR),
            "-c:a", "pcm_s16le", music_wav,
        ], check=True)
    music = read_wav(music_wav)
    if music.shape[1] == 1:
        music = np.repeat(music, 2, axis=1)
    if len(music) >= n:
        track = music[:n].copy()
    else:
        xf = int(.25 * SR)
        core, tail = music[:-xf], music[-xf:]
        ramp = np.linspace(0, 1, xf)[:, None]
        loop = core.copy()
        loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
        track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n].copy()
    voice_rms = np.sqrt((voice ** 2).mean()) + 1e-9
    music_rms = np.sqrt((track ** 2).mean()) + 1e-9
    track *= (voice_rms / music_rms) * (10 ** (-19 / 20))
    fi, fo = int(.8 * SR), int(2.0 * SR)
    track[:fi] *= np.linspace(0, 1, fi)[:, None]
    track[-fo:] *= np.linspace(1, 0, fo)[:, None]
    bed += track.astype(np.float32)

    mix = voice + bed
    if np.abs(mix).max() > .99:
        mix *= .99 / np.abs(mix).max()
    mix_path = f"{BUILD}/assets/_mix_st_petersburg_{key}.wav"
    with wave.open(mix_path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes((mix * 32767).astype(np.int16).tobytes())
    mux(vid, mix_path, cfg)
    return gain


def save_sheet(key, sheet):
    cols = 6
    rows = math.ceil(len(sheet) / cols)
    out = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    draw = ImageDraw.Draw(out)
    for i, (t, arr) in enumerate(sheet):
        tile = Image.fromarray(arr).resize((180, 320), Image.LANCZOS)
        out.paste(tile, ((i % cols) * 180, (i // cols) * 320))
        draw.text(((i % cols) * 180 + 7, (i // cols) * 320 + 7), f"{t:.1f}", fill="white")
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    path = f"{BUILD}/test/hook29_{key}_sheet.jpg"
    out.save(path, quality=92)
    print("контакт-лист:", path)


def write_plan(key, cfg, n_body, gain):
    duration = (cfg["frames"] + n_body) / FPS
    if cfg.get("post_a"):
        duration /= 1.02
    lines = [
        f"# Монтажный лист — ролик 29, {key}", "",
        f"Файл: `videos/29/{os.path.basename(cfg['out'])}` · 1080×1920 · 30 fps · {duration:.2f}с",
        f"Новый хук: «{cfg['title']}»", "",
        f"Тело: готовые кадры {CUT_F}+ из `assets/_video_st_petersburg.mp4`; не пересобиралось.",
        f"Звук тела начинается с {CUT:.4f}с исходника — перед полным словом «итак».", "",
        "| тайминг | рецепт | что на экране | текст |", "|---|---|---|---|",
    ]
    for t0, t1, kind, _ in cfg["shots"]:
        texts = ["".join(x[0] for x in runs).strip() for a, _b, runs, _s in cfg["caps"] if t0 <= a < t1]
        desc = "лицо, штатный A-roll" if kind in R.FACE_KINDS else "сетка: растущая лестница монет и ∞"
        lines.append(f"| {t0:.2f}–{t1:.2f} | {'R1' if kind in R.FACE_KINDS else 'R4+R5a'} | {desc} | {' / '.join(texts)} |")
    lines += [
        "", "## Версия", "",
        f"- Музыка: `{os.path.basename(cfg['music'])}`.",
        f"- Деформация: `{cfg['post_v']}`.",
        f"- Коррекция RMS голоса хука: ×{gain:.3f}.",
        "- Старое вступление и его субтитры удалены целиком.",
        "- Стоковые вставки и кириллические «О»/«Р» в теле сохранены без изменений.",
    ]
    path = f"{VIDEO_DIR}/montage-plan-hook{1 if key == 'h1' else 2}.md"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(key):
    cfg = HOOKS[key]
    tmp = f"{BUILD}/assets/_video_st_petersburg_{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    print("RGB хук", card_rgb(cfg["src"], .8), "· основной", card_rgb(SRC, .8))
    gain = build_audio(key, cfg, nf_hook, apply_post_v(key, cfg))
    save_sheet(key, sheet)
    write_plan(key, cfg, n_body, gain)
    print("готово:", cfg["out"])


if __name__ == "__main__":
    main(sys.argv[1])

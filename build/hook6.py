"""Ролик 6, альтернативные хуки.

Тело ролика (всё после 3.900с) не пересобирается — берётся готовыми кадрами из
assets/_video_v6.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры, та же графика «палка с двумя разломами».

    python3 hook6.py h1
    python3 hook6.py h2
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render6 as R
from sfx import low_whoosh, impact, tick, read_wav
import storyboard6 as SB6

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/6"
BODY = f"{BUILD}/assets/_video_v6.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
MUSIC = f"{AUDIO_DIR}/song1.mp3"   # трек ролика по умолчанию; у каждой версии свой
SRC = SB6.SRC
SR = 48000

CUT = 3.900                      # где кончается старый хук в исходном ролике
CUT_F = int(round(CUT * FPS))    # 117
BODY_DUR = SB6.DUR - CUT

S = "r"

# A-roll хука собирается штатным трактом: FRAMINGS["A1"] + hflip + GRADE, без поправок под дубль.
# Крупность, свет и ориентация дублей — забота съёмки, монтаж их не подгоняет (START-HERE.md).
#
# Исключение — крупность: дубли сняты с большей дистанции (лицо 367 и 382px против 431px),
# поэтому кроп у них свой, подобранный по детектору лица. Цвет НЕ трогаем: дубли уже
# отгрейжены на съёмочной стороне, lut выключен.
HOOKS = {
    "h1": dict(
        src=f"{VIDEO_DIR}/hook1.mov",
        crop="hflip,crop=828:1312:93:404",
        lut=None,
        dur=100 / FPS,                       # 3.333с, речь кончается на 3.20
        out=f"{VIDEO_DIR}/stick_triangle_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # рилс 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=lanczos",
        title="«эту задачу дают на собеседовании»",
        shots=[
            (0.000, 1.600, "A1", {}),
            (1.600, 100 / FPS, "intuition", {}),
        ],
        caps=[
            (0.000, 0.940, [("эту задачу", S, 54)], "A"),
            (0.940, 1.920, [("дают на ", S, 46), ("собеседование", "s", 60)], "A"),
            (1.920, 2.400, [("в лучшей", S, 50)], "T"),
            (2.400, 100 / FPS, [("компании ", S, 46), ("мира", "s", 66)], "T"),
        ],
    ),
    "h2": dict(
        src=f"{VIDEO_DIR}/hook2.mov",
        crop="hflip,crop=862:1366:98:389",
        lut=None,
        dur=87 / FPS,                        # 2.900с, речь кончается на 2.82
        out=f"{VIDEO_DIR}/stick_triangle_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # рилс 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: без него на выходе 30.6 fps
        post_a="atempo=1.02",
        title="«все, кого я спросил, решили неправильно»",
        shots=[
            (0.000, 1.300, "A1", {}),
            (1.300, 87 / FPS, "intuition", {}),
        ],
        caps=[
            (0.000, 0.840, [("все, кого я", S, 52)], "A"),
            (0.840, 1.300, [("спросил", "s", 66)], "A"),
            (1.300, 1.780, [("решили эту", S, 50)], "T"),
            (1.780, 87 / FPS, [("задачу ", S, 46), ("неправильно", "s", 62)], "T"),
        ],
    ),
}

GRADE = R.GRADE
CX, CY, CW, CH = CARD_A


# ----------------------------------------------------------------- A-roll хука
def prep_hook_aroll(key, cfg):
    """Тот же тракт, что у основного A-roll: hflip + FRAMINGS + GRADE.
    cfg['crop'] / cfg['lut'] — аварийные переопределения, по умолчанию выключены."""
    out = f"{BUILD}/assets/aroll_v6{key}_A1.mp4"
    if not os.path.exists(out):
        if cfg["crop"]:
            vf = f"{cfg['crop']},scale={CW}:{CH}:flags=lanczos,{GRADE}"
        else:
            w, h, x, y = SB6.FRAMINGS["A1"]
            vf = f"hflip,crop={w}:{h}:{x}:{y},scale={CW}:{CH}:flags=lanczos,{GRADE}"
        if cfg["lut"]:
            vf += f",{cfg['lut']}"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", cfg["src"], "-vf", vf, "-an", "-c:v", "libx264",
                        "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p",
                        out], check=True)
        print("A-roll хука:", out)
    return out


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_v6.mp4."""
    aroll = prep_hook_aroll(key, cfg)

    # переключаем раскадровку в render6 на хуковую и пересобираем раскладку блоков
    R.SHOTS, R.CAPS = cfg["shots"], cfg["caps"]
    R.BLOCKS = R._build_blocks()
    R.BLOCK_LAY = R._block_layout()

    nf_hook = int(round(cfg["dur"] * FPS))
    maskA = rounded_mask(CW, CH, R_A)
    cap = cv2.VideoCapture(aroll)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet = []
    si = 0
    for f in range(nf_hook):
        t = f / FPS
        cap.grab()
        while si < len(R.SHOTS) - 1 and t >= R.SHOTS[si][1]:
            si += 1
        t0, t1, kind, prm = R.SHOTS[si]
        lt = t - t0

        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        if kind == "A1":
            okr, img = cap.retrieve()
            if not okr:
                img = np.zeros((CH, CW, 3), np.uint8)
            canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                         (CX, CY), maskA)
        else:
            canvas.paste(grid_canvas(CW, CH, phase=(f // 3) * 0.06), (CX, CY), maskA)

        gl, titems = R.shot_layers(kind, prm, lt, t1 - t0, t)
        for lay in gl:
            canvas.alpha_composite(lay)
        items = titems + R.caption_items(t, kind)
        if items:
            canvas.alpha_composite(text_layer((W, H), items))

        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 6 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()

    # тело: готовые кадры, начиная с 3.900с
    body = cv2.VideoCapture(BODY)
    body.set(cv2.CAP_PROP_POS_FRAMES, CUT_F)
    n_body = 0
    while True:
        ok, img = body.read()
        if not ok:
            break
        ff.stdin.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes())
        n_body += 1
    body.release()

    ff.stdin.close(); ff.wait()
    print(f"видео: хук {nf_hook} кадров + тело {n_body} кадров -> {tmp}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
    subprocess.run(cmd, check=True)
    d = read_wav(tmp)
    os.remove(tmp)
    return d


def speech_rms(x):
    m = np.abs(x).mean(axis=1)
    thr = m.max() * 0.12
    sel = m > thr
    return float(np.sqrt((x[sel] ** 2).mean()))


def build_audio(key, cfg, nf_hook, vid, out):
    hook = wav_of(cfg["src"])
    body = wav_of(SRC, CUT)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = nf_hook * SR // FPS
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]

    voice = np.vstack([hook, body]).astype(np.float32)
    n = voice.shape[0]
    peak = np.abs(voice).max()
    bed = np.zeros((n, 2), np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        m = min(len(sig), n - i)
        if i >= 0 and m > 0:
            bed[i:i + m, 0] += sig[:m] * gain
            bed[i:i + m, 1] += sig[:m] * gain

    dt = cfg["dur"] - CUT                      # сдвиг тела на новой таймлинии
    wh, im, tk = low_whoosh(), impact(), tick()
    starts = [s[0] for s in cfg["shots"][1:]] + [s[0] + dt for s in SB6.SHOTS if s[0] >= CUT]
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    for t in SB6.NUM_REVEALS:
        add(im, t + dt, peak * 0.070)
    for t in SB6.CASCADES:
        for k in range(3):
            add(tk, t + dt + k * 0.085, peak * 0.030)

    music = cfg.get("music", MUSIC)
    mw = f"{BUILD}/assets/_music_{os.path.basename(music).split('.')[0]}.wav"
    if not os.path.exists(mw):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", music,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
    print("музыка:", os.path.basename(music))
    mus = read_wav(mw)
    if mus.shape[1] == 1:
        mus = np.repeat(mus, 2, axis=1)
    xf = int(0.25 * SR)
    core, tail = mus[:len(mus) - xf], mus[len(mus) - xf:]
    ramp = np.linspace(0, 1, xf)[:, None]
    loop = core.copy()
    loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
    track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n]
    vr = np.sqrt((voice ** 2).mean()) + 1e-9
    mr = np.sqrt((track ** 2).mean()) + 1e-9
    track *= (vr / mr) * (10 ** (-19 / 20))
    fi, fo = int(0.8 * SR), int(2.0 * SR)
    track[:fi] *= np.linspace(0, 1, fi)[:, None]
    track[-fo:] *= np.linspace(1, 0, fo)[:, None]
    bed += track.astype(np.float32)

    mix = voice + bed
    m = np.abs(mix).max()
    if m > 0.99:
        mix *= 0.99 / m
    mix_path = f"{BUILD}/assets/_mix_v6{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()

    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Лёгкая деформация кадра/темпа, чтобы Instagram не счёл версии одинаковыми.
    Работает по немому _video_v6<key>.mp4, кадры хука и тела не перерисовываются."""
    src = f"{BUILD}/assets/_video_v6{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_v6{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                    "-pix_fmt", "yuv420p", dst], check=True)
    print("правка кадра:", vf)
    return dst


def measure_i(path):
    import json
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    m = json.loads(s[s.rindex("{"):s.rindex("}") + 1])
    return float(m["input_i"]), float(m["input_tp"])


def mux(vid, mix_path, out, target=-14.2, extra_af=None):
    """loudnorm в динамическом режиме на этом материале недобирает ~0.9 dB,
    поэтому подгоняем предусиление за пару итераций (эталон ролика 6: -14.2 LUFS)."""
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        if abs(post_db) > 0.02:
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.80:level=disabled"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", vid, "-i", mix_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-af", af,
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out,
        ], check=True)
        i, tp = measure_i(out)
        print("  loudnorm: post %+.2f dB -> I=%.2f LUFS, TP=%.2f dBFS" % (post_db, i, tp))
        return i, tp

    post_db, best = 0.0, None
    for _ in range(4):
        i, tp = run(post_db)
        if tp <= -1.2 and (best is None or abs(i - target) < abs(best[1] - target)):
            best = (post_db, i)
        if abs(i - target) < 0.15 and tp <= -1.2:
            best = None
            break
        post_db += target - i
    if best is not None:                       # последняя итерация вышла за TP — вернуть лучшую
        run(best[0])
    print("готово:", out)


def main(key):
    cfg = HOOKS[key]
    tmp = f"{BUILD}/assets/_video_v6{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        im = Image.fromarray(arr).resize((180, 320))
        cs.paste(im, ((i % cols) * 180, (i // cols) * 320))
    csp = f"{BUILD}/test/hook_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_v6{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        # кадры уже собраны — пересобрать только звук, правку кадра и мукс
        nf = int(round(HOOKS[k]["dur"] * FPS))
        build_audio(k, HOOKS[k], nf, apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

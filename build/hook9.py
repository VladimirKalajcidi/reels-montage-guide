"""Ролик 9 («формула Эйлера»), альтернативные хуки.

Тело ролика (всё с 3.300с) не пересобирается — берётся готовыми кадрами из
assets/_video_euler.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры, та же вставка (учёная у доски, Mixkit 4619), что стояла
в старом хуке — это визуальная интрига начала, менять её незачем.

    python3 hook9.py h1
    python3 hook9.py h2

Точка реза 3.300с — конец хуковой фразы («…в истории математики», source.srt) и
ровно граница плана 2 (стоковая вставка). Планы 1–2 старого начала и их субтитры
выброшены целиком. Кадры 99…106 тела — те самые 0.24с паузы-вдоха перед «в ней»,
они уже есть в готовом видео и переносятся как есть: пауза ролика не переписывается.

Формат хуков (как у роликов 7 и 8): дубли сняты в HDR (10 бит, bt2020nc / HLG),
основной исходник — обычный SDR bt709. Без конверсии HLG декодируется как bt709
и картинка уезжает в светлый плоский «туман» (замер: 104/101/94 против 93/78/68
у основного дубля). Поэтому сначала честный tonemap HDR->SDR, потом штатный тракт
A-roll (source_card: hflip + кроп под карточку A + грейд) без единой поправки цвета.
npl подобран по числам, не на глаз — см. TONEMAP.
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
import render9 as R
import storyboard9 as SB
from sfx9 import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/9"
BODY = f"{BUILD}/assets/_video_euler.mp4"
STOCK_DIR = f"{VIDEO_DIR}/stock"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT = 3.300                        # конец хуковой фразы = граница плана 2
CUT_F = 99                         # 3.300*30 = 99.0 — точная граница кадра
TARGET_LUFS = -15.0                # как в уже сданном euler_edit.mp4 (замер: -15.04)

S = "r"

# Зеркальность проверена по кадру: логотип на толстовке читается задом наперёд
# и в основном дубле, и в обоих хуках — штатный hflip в source_card подходит всем трём.
#
# npl подобран по средним RGB кадра (лицо, t=1.0с) против основного дубля 93/78/68:
#   npl=130 -> 93/87/79   npl=150 -> 89/83/75   npl=160 -> 87/81/73   npl=200 -> 80/74/68
# Взят 160: расхождение −6/+3/+6 — минимальный разброс по всем трём каналам.
TONEMAP = ("zscale=t=linear:npl=160,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")

STOCK_HOOK = {"clip": "4619", "ss": 2.0}     # та же вставка, что в старом хуке

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        frames=146,                       # весь дубль; речь кончается на 4.820
        speech_end=4.820,
        out=f"{VIDEO_DIR}/euler_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=lanczos",
        post_a=None,
        title="«в математике есть формула, которая кажется слишком идеальной»",
        shots=[
            (0.000, 1.720, "A1",    {}),
            (1.720, 3.400, "stock", STOCK_HOOK),
            (3.400, 146 / 30, "A1", {}),
        ],
        caps=[
            (0.000, 0.840,  [("в математике", S, 50)]),
            (0.840, 1.720,  [("есть формула", S, 50)]),
            (1.820, 2.280,  [("которая кажется", S, 48)]),
            (2.280, 3.400,  [("слишком идеальной", S, 46)]),
            (3.420, 3.780,  [("чтобы быть", S, 48)]),
            (3.780, 146 / 30, [("просто ", S, 46), ("случайностью", "s", 66)]),
        ],
        cascades=[],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        frames=134,                       # весь дубль; речь кончается на 4.420
        speech_end=4.420,
        out=f"{VIDEO_DIR}/euler_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",  # fps=30 обязателен: без него на выходе 30.6 fps
        post_a="atempo=1.02",
        title="«эту формулу считают самой красивой в истории математики, и это не просто так»",
        shots=[
            (0.000, 1.660, "A1",    {}),
            (1.660, 3.400, "stock", STOCK_HOOK),
            (3.400, 134 / 30, "A1", {}),
        ],
        caps=[
            (0.000, 0.960,  [("эту формулу", S, 50)]),
            (0.960, 1.660,  [("считают самой", S, 48)]),
            (1.660, 2.320,  [("красивой", "s", 74)]),
            (2.320, 3.400,  [("в истории математики", S, 46)]),
            (3.400, 3.660,  [("и это", S, 48)]),
            (3.660, 134 / 30, [("не просто так", S, 50)]),
        ],
        cascades=[],
    ),
}

CX, CY, CW, CH = CARD_A


# ----------------------------------------------------------------- A-roll хука
def sdr_source(key, cfg):
    """HDR-дубль -> SDR bt709. Звук берём из оригинала, здесь только картинка."""
    out = f"{BUILD}/assets/_hook9_{key}_sdr.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", cfg["src"], "-vf", TONEMAP, "-an",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                        "-pix_fmt", "yuv420p", out], check=True)
        print("хук приведён к SDR bt709:", os.path.basename(out))
    return out


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_euler.mp4.

    Раскадровка в render9 подменяется на хуковую: SHOTS/CAPS не импортированы
    по значению внутрь функций, они читаются из глобалей модуля — поэтому
    достаточно подменить их и пересобрать раскладку блоков."""
    aroll = sdr_source(key, cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(aroll)
    stock_cap, stock_key = None, None

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet = []
    last_img = None
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last_img = fr
        else:
            fr = last_img
        t0, t1, kind, prm = R.shot_at(t)
        lt = t - t0

        def read_stock():
            nonlocal stock_cap, stock_key
            k = (t0, prm["clip"])
            if k != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = k
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = R.background(kind, prm, fr, t, read_stock)
        gl = R.graphics_layer(kind, lt)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)

        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 6 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()
    if stock_cap is not None:
        stock_cap.release()

    # тело: готовые кадры, начиная с CUT_F
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

    ff.stdin.close()
    ff.wait()
    print(f"видео: хук {nf_hook} кадров + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook9.wav"
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
    sel = m > m.max() * 0.12
    return float(np.sqrt((x[sel] ** 2).mean()))


def build_audio(key, cfg, nf_hook, vid, out):
    """Дорожка пересобирается целиком: голос хука + голос тела с точки реза,
    постель со сдвигом (длина_хука − CUT), музыка с нуля, потом loudnorm."""
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
    # склейка в тишине: гасим последние 8мс хука и первые 8мс тела, чтобы не было щелчка
    r = int(0.008 * SR)
    hook[-r:] *= np.linspace(1, 0, r)[:, None]
    body = body.copy()
    body[:r] *= np.linspace(0, 1, r)[:, None]

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

    dt = nf_hook / FPS - CUT                 # сдвиг тела на новой таймлинии
    wh, im, tk = low_whoosh(), impact(), tick()
    starts = [s[0] for s in cfg["shots"][1:]] + \
             [s[0] + dt for s in SB.SHOTS if s[0] >= CUT]
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t in cfg["cascades"]:
        for k in range(3):
            add(tk, t + k * 0.085, peak * 0.030)

    music = cfg["music"]
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
    mix_path = f"{BUILD}/assets/_mix_euler{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Лёгкая деформация кадра/темпа, чтобы площадка не сочла версии дублями.
    Работает по уже собранному немому _video_euler<key>.mp4, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_euler{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_euler{key}_post.mp4"
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


def mux(vid, mix_path, out, target=TARGET_LUFS, extra_af=None):
    """Догоняем громкость до уровня уже сданного ролика."""
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
        if tp <= -1.0 and (best is None or abs(i - target) < abs(best[1] - target)):
            best = (post_db, i)
        if abs(i - target) < 0.15 and tp <= -1.0:
            best = None
            break
        post_db += target - i
    if best is not None:
        run(best[0])
    print("готово:", out)


def main(key):
    cfg = HOOKS[key]
    tmp = f"{BUILD}/assets/_video_euler{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook9_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_euler{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"], apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

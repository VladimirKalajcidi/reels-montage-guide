"""Ролик 13 («парадокс Банаха — Тарского»), альтернативные хуки.

Тело ролика (всё с 8.433с) не пересобирается — берётся готовыми кадрами из
assets/_video_banach.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры, та же графика (шар -> куски -> два шара -> равные
линейки), что стояла в старом хуке — это визуальная интрига начала, менять её незачем.

    python3 hook13.py h1
    python3 hook13.py h2

Точка реза 8.433с. Хуковая фраза оригинала кончается словом «исходный» на 8.420
(source.srt), ближайшая граница плана — 8.42 (конец плана 5, «equal»). В кадрах это
252.6, поэтому режем по первому кадру следующего плана: кадр 253 = 8.4333с. Планы 1–5
старого начала и их субтитры выброшены целиком, ни одного кадра хука в теле не остаётся.

Формат хуков: оба дубля SDR bt709, как и основной исходник — тонмаппинг не нужен.
Расходятся два формальных свойства, оба приводятся (START-HERE, «приведение формата
делается всегда»):
  * разрешение 720x1280 против 1080x1920 — апскейл lanczos до холста тракта;
  * color_range=pc (yuvj420p) против tv — приведение к limited.
Замер: приведение range меняет картинку на ~2 единицы по каналу (109.7/97.5/92.0 ->
107.6/96.6/89.7), то есть декодер и так читал теги верно. Настоящее расхождение по
свету — съёмочное, оно НЕ правится, см. шапку монтажного листа.

Зеркальность проверена по кадру: надпись на толстовке читается задом наперёд и в
основном дубле, и в обоих хуках — штатный hflip в source_card подходит всем трём.
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
import render13 as R
import storyboard13 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/13"
BODY = f"{BUILD}/assets/_video_banach.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 253                        # первый кадр плана 6; 253/30 = 8.4333с
CUT = CUT_F / FPS
TARGET_LUFS = -14.1                # как в уже сданном banach_edit.mp4 (замер: -14.1)

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        frames=233,                       # дубль 234 кадра; речь кончается на 7.720
        speech_end=7.720,
        out=f"{VIDEO_DIR}/banach_hook1.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=lanczos",
        post_a=None,
        title="«вы скорее всего в это не поверите, но математически доказано, "
              "что из одного шара можно получить два шара такого же размера»",
        shots=[
            (0.000, 1.700, "A1", {}),
            (1.700, 3.200, "ball", {}),
            (3.200, 4.600, "split", {}),
            (4.600, 6.180, "two", {}),
            (6.180, 233 / 30, "equal", {}),
        ],
        caps=[
            (0.000, 0.660, [("вы скорее всего", S, 48)], "A"),
            (0.660, 1.700, [("в это ", S, 46), ("не поверите", "s", 62)], "A"),
            (1.700, 3.200, [("но математически ", S, 44), ("доказано", "s", 74)], "G"),
            (3.200, 3.920, [("что из одного шара", S, 44)], "G"),
            (3.920, 4.600, [("можно", S, 50)], "G"),
            (4.600, 5.300, [("получить", S, 52)], "G"),
            (5.300, 6.180, [("два шара", S, 50)], "G"),
            (6.180, 7.060, [("такого же ", S, 46), ("размера", "s", 60)], "G"),
            (7.060, 233 / 30, [("как и изначально", S, 46)], "G"),
        ],
        # каскады хука: разлёт секторов в split, ревилов синего числа в хуке нет
        cascades=[3.45],
        reveals=[],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        frames=227,                       # дубль 228 кадров; речь кончается на 7.560
        speech_end=7.560,
        out=f"{VIDEO_DIR}/banach_hook2.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",  # fps=30 обязателен: без него на выходе 30.6 fps
        post_a="atempo=1.02",
        title="«я сам в это в начале не поверил, но оказалось, что математически "
              "доказали, что шар можно разбить на два шара такого же размера»",
        shots=[
            (0.000, 1.680, "A1", {}),
            (1.680, 3.760, "ball", {}),
            (3.760, 4.980, "split", {}),
            (4.980, 6.400, "two", {}),
            (6.400, 227 / 30, "equal", {}),
        ],
        caps=[
            (0.000, 0.640, [("я сам в это", S, 48)], "A"),
            (0.640, 1.680, [("в начале ", S, 46), ("не поверил", "s", 62)], "A"),
            (1.680, 2.340, [("но оказалось", S, 48)], "G"),
            (2.340, 3.760, [("что математически ", S, 42), ("доказали", "s", 68)], "G"),
            (3.760, 4.400, [("что шар можно", S, 46)], "G"),
            (4.400, 4.980, [("разбить", "s", 66)], "G"),
            (4.980, 5.840, [("на два шара", S, 50)], "G"),
            (5.840, 6.400, [("такого же", S, 48)], "G"),
            (6.400, 7.060, [("размера", "s", 62)], "G"),
            (7.060, 227 / 30, [("как и изначально", S, 46)], "G"),
        ],
        cascades=[4.01],
        reveals=[],
    ),
}


# ----------------------------------------------------------------- A-roll хука
def conformed_source(key, cfg):
    """Приведение формата: 720x1280 -> холст тракта, full range -> limited.

    Это НЕ грейд. Оба свойства однозначно читаются из тегов файла, поэтому
    приводятся всегда (START-HERE, «граница простая»). Экспозиция, баланс белого
    и крупность головы не трогаются.
    """
    out = f"{VIDEO_DIR}/prepared/hook{cfg['version'] - 1}_aroll.mp4"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", cfg["src"],
            "-vf", "scale=1080:1920:flags=lanczos,"
                   "scale=in_range=full:out_range=limited,format=yuv420p",
            "-an", "-c:v", "libx264", "-crf", "14", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
            "-color_trc", "bt709", "-color_primaries", "bt709", out], check=True)
        print("хук приведён к формату тракта:", os.path.basename(out))
    return out


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_banach.mp4.

    SHOTS/CAPS читаются функциями render13 из глобалей модуля — поэтому достаточно
    подменить их и пересобрать раскладку блоков."""
    aroll = conformed_source(key, cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(aroll)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet, last_img = [], None
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last_img = fr
        else:
            fr = last_img
        t0, t1, kind, prm = R.shot_at(t)

        canvas = R.background(kind, prm, fr, t, None)
        gl = R.graphics_layer(kind, t - t0)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)

        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 8 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()

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
    tmp = f"{BUILD}/assets/_tmp_hook13.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
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
    # Склейка хук/тело — такой же рез, как все остальные, и свой whoosh обязана
    # получить. План 6 тела начинается на 8.42, то есть ЧУТЬ РАНЬШЕ точки реза
    # 8.4333, и по фильтру s[0] >= CUT в список не попадал — стык оставался немым.
    starts = [s[0] for s in cfg["shots"][1:]] + [nf_hook / FPS] + \
             [s[0] + dt for s in SB.SHOTS if s[0] >= CUT]
    for t0 in starts:
        add(wh, t0, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in cfg["reveals"]:
        add(im, t, peak * 0.070)
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
    mix_path = f"{BUILD}/assets/_mix_banach{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Лёгкая деформация кадра/темпа, чтобы площадка не сочла версии дублями.
    Работает по уже собранному немому _video_banach<key>.mp4, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_banach{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_banach{key}_post.mp4"
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
    """Догоняем громкость до уровня уже сданного ролика.

    alimiter=...:level=disabled обязателен: автонормализация у alimiter включена
    по умолчанию и разгоняет громкость мимо цели, выбивая истинный пик в ноль.
    """
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        if abs(post_db) > 0.02:
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.80:level=disabled"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", vid, "-i", mix_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-af", af, "-ar", str(SR),
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
    tmp = f"{BUILD}/assets/_video_banach{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)
    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook13_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "reaudio":
        # пересобрать только звук по уже собранному немому видео
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    elif len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_banach{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

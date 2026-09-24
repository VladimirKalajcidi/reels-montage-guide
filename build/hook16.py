"""Ролик 16 («почему А4 — это 1 : √2»), альтернативные хуки.

Тело ролика (всё с кадра 134) не пересобирается — берётся готовыми кадрами из
assets/_video_a4.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры и графика листа `sheet` — та же, что стояла в старом
хуке (`storyboard16.py`, план 1.86–4.44).

    python3 hook16.py h1
    python3 hook16.py h2

Точка реза: кадр 134 (4.46667с) — граница плана `sheet` (1.86–4.44), то есть конца
старого хука «формат бумаги А4 выбрали не случайно, он математически очень удобен».
Планы 1–2 старого начала и их субтитры выброшены целиком. Оба новых дубля говорят
ровно про это же — «а вы знали, что пропорции листа...», — поэтому оба садятся
на одну и ту же точку.

J-cut не нужен. В теле слово «возьмите» начинается по энергии на 4.560с, то есть
после реза остаётся 0.093с тишины — атака не задета. Хук режется на 0.04с позже
конца своей речи, суммарная пауза между хуком и телом выходит 0.13–0.14с
(норма delivery-specs §6 — 0.10–0.20с).

Формат хуков. Дубли сняты 720x1280 в полном диапазоне (`pix_fmt=yuvj420p`,
`color_range=pc`), основной исходник — 1080x1920 `tv`. И размер, и range — объективные
свойства файла, они приводятся всегда (START-HERE, «приведение формата»), после чего
хук идёт через штатный тракт A-roll без единой поправки под дубль. Грейда в тракте
нет ни у тела, ни у хука. Крупность головы и экспозиция у хуков заметно расходятся
с основным дублем — это НЕ правится, расхождение измерено и названо в сдаче.

Границы речи в дублях взяты по уровню, а не по Whisper: у обоих хуков Whisper
относит первое «А» на 0.4с раньше реального начала звука (1.34 против 1.76 у hook1,
0.92 против 1.44 у hook2).
"""
import json
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render16 as R
import storyboard16 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/16"
BODY = f"{BUILD}/assets/_video_a4.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 134                         # первый кадр тела = граница плана `sheet`
CUT = CUT_F / FPS                   # 4.46667с
TARGET_LUFS = -14.0                 # как в уже сданном a4_edit.mp4 (замер)

# Зеркальность проверена по кадру: надпись на футболке читается задом наперёд
# и в основном дубле, и в обоих хуках — штатный hflip в source_card подходит всем трём.
CONFORM = "scale=1080:1920:flags=lanczos:in_range=pc:out_range=tv,format=yuv420p"

HOOK_GRID = {"sheet", "trio", "sides"}
S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        ss=1.620,          # реальное начало звука 1.76, берём 0.14с разгона
        frames=316,        # 10.5333с; речь кончается на 12.113 файла = 10.493 блока
        out=f"{VIDEO_DIR}/a4_hook1.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«а вы знали, что пропорции листа у всех форматов А3, А4, А5 они одинаковые, "
              "и почему они именно такие какие они есть? на самом деле это математически "
              "правильное соотношение сторон»",
        shots=[
            (0.000, 1.820, "A1", {}),
            (1.820, 4.920, "trio", {}),
            (4.920, 7.040, "A2", {}),
            (7.040, 8.960, "sheet", {}),
            (8.960, 316 / 30, "A1", {}),
        ],
        caps=[
            (0.140, 0.740, [("а вы знали", S, 50)], "A"),
            (0.940, 1.820, [("что пропорции листа", S, 46)], "A"),
            (1.820, 2.660, [("у всех форматов", S, 46)], "G"),
            # 2.66–3.86 «а3 а4 а5» — эти же имена стоят подписями на графике,
            # поэтому субтитра на них нет (brand-kit §5)
            (3.860, 4.920, [("они ", S, 46), ("одинаковые", "s", 60)], "G"),
            (5.180, 5.780, [("и почему они", S, 48)], "A"),
            (5.780, 6.280, [("именно такие", S, 48)], "A"),
            (6.280, 7.040, [("какие они есть", S, 46)], "A"),
            (7.340, 8.020, [("на самом деле", S, 48)], "G"),
            (8.020, 8.960, [("это ", S, 46), ("математически", "s", 54)], "G"),
            (8.960, 10.340, [("правильное соотношение сторон", S, 40)], "A"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        ss=1.300,          # реальное начало звука 1.44
        frames=209,        # 6.9667с; речь кончается на 8.220 файла = 6.920 блока
        out=f"{VIDEO_DIR}/a4_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«а вы знали, что формат листа А4 не просто такой, какой он есть. "
              "это очень математически выгодное соотношение»",
        shots=[
            (0.000, 1.760, "A1", {}),
            (1.760, 3.300, "sheet", {}),
            (3.300, 4.980, "A2", {}),
            # не второй `sheet`: он бы просто перерисовался с нуля и стоял.
            # `sides` берёт обе стороны в мерные скобки — это и есть «соотношение»
            (4.980, 209 / 30, "sides", {}),
        ],
        caps=[
            (0.140, 0.780, [("а вы знали", S, 50)], "A"),
            (1.020, 1.760, [("что формат листа", S, 46)], "A"),
            (1.760, 2.320, [("а4", S, 72)], "G"),
            (2.320, 3.300, [("не просто такой", S, 46)], "G"),
            (3.400, 4.160, [("какой он есть", S, 48)], "A"),
            (4.520, 4.980, [("это очень", S, 50)], "A"),
            (4.980, 6.100, [("математически ", S, 42), ("выгодное", "s", 60)], "G"),
            (6.100, 6.820, [("соотношение", "s", 62)], "G"),
        ],
    ),
}


# ----------------------------------------------------------------- A-roll хука
def conform_source(key, cfg):
    """Приведение формата хука к формату основного исходника. Звук не трогаем.

    -ss стоит ПОСЛЕ -i: нужен точный кадр, а не ближайший ключевой.
    """
    out = f"{BUILD}/assets/_hook16_{key}_conf.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", cfg["src"], "-ss", f"{cfg['ss']:.4f}",
                        "-vf", CONFORM, "-an", "-color_range", "tv",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                        "-pix_fmt", "yuv420p", out], check=True)
        print("хук приведён к 1080x1920 tv:", os.path.basename(out))
    return out


def card_rgb(path, t, kind="A1"):
    """Средний RGB внутри карточки A после штатного тракта A-roll."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    card = np.array(R.source_card(fr, kind))
    return tuple(int(v) for v in card.reshape(-1, 3).mean(axis=0))


def face_share(path, t):
    """Ширина лица в долях кадра — чтобы назвать расхождение крупности числом."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    cas = cv2.CascadeClassifier(cv2.data.haarcascades +
                                "haarcascade_frontalface_default.xml")
    d = cas.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.15, 6,
                             minSize=(int(fr.shape[1] * 0.15),) * 2)
    if len(d) == 0:
        return None
    x, y, w, h = max(d, key=lambda r: r[2])
    return round(100 * w / fr.shape[1], 1), round(100 * (y + 0.4 * h) / fr.shape[0], 1)


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_a4.mp4."""
    aroll = conform_source(key, cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(aroll)

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-pix_fmt", "yuv420p", tmp],
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
        gl = R.graphics_layer(kind, (t - t0) * prm.get("lts", 1.0) + prm.get("lt0", 0.0))
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
def wav_of(path, start=0.0, key=""):
    tmp = f"{BUILD}/assets/_tmp_hook16{key}.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    if start:
        cmd += ["-ss", f"{start:.4f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
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
    постель со сдвигом, музыка с нуля, потом loudnorm."""
    hook = wav_of(cfg["src"], cfg["ss"], key)
    body = wav_of(SRC, CUT, key)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round(nf_hook / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    r = int(0.008 * SR)                                  # мягкий стык, без щелчка
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
    for t0 in [s[0] for s in cfg["shots"][1:]]:
        add(wh, t0, peak * 0.040)
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 >= CUT:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0, _t1, kind, _p in cfg["shots"]:       # каскады графики самого хука
        if kind in HOOK_GRID:
            for k in range(3):
                add(tk, t0 + 0.12 + k * 0.085, peak * 0.030)

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
    mix_path = f"{BUILD}/assets/_mix_a4{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_a4{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_a4{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "12", "-preset", "slow",
                    "-pix_fmt", "yuv420p", dst], check=True)
    print("правка кадра:", vf)
    return dst


def measure_i(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=7:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    m = json.loads(s[s.rindex("{"):s.rindex("}") + 1])
    return float(m["input_i"]), float(m["input_tp"])


def mux(vid, mix_path, out, target=TARGET_LUFS, extra_af=None):
    """Догоняем громкость до уровня уже сданного ролика, следя за истинным пиком."""
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + \
             "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
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
    tmp = f"{BUILD}/assets/_video_a4{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    # расхождение дублей — измеряем и называем, а не чиним кропом
    conf = conform_source(key, cfg)
    print("средний RGB в карточке A: хук", card_rgb(conf, 1.0),
          "· основной дубль", card_rgb(SRC, 17.1))
    print("лицо (% ширины кадра, линия глаз % высоты): хук", face_share(conf, 1.0),
          "· основной дубль", face_share(SRC, 17.1))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook16_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_a4{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

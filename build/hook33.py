"""Ролик 33 («100 заключённых и 100 коробок»), альтернативные хуки.

Тело ролика (всё с кадра 85) не пересобирается — берётся готовыми кадрами из
`assets/_video_boxes.mp4` и склеивается за новым хуком. Заново рисуется только
блок хука. render33.py этот скрипт не трогает: графика хука — тот же самый
план `people` из `render33.GFX`, со своим локальным временем.

    python3 hook33.py h1
    python3 hook33.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «это одна из самых безумных задач по вероятности»
(0.000–2.449). Оно занимает план `A1` 0.000–1.746 **и хвостом заходит**
в план `people` 1.746–3.873, потому что решётка из ста фигурок начинает
собираться ещё под словами «по вероятности». Тело же начинается внутри того
же плана: «итак есть 100 заключённых» — 2.842.

Поэтому «резать по границе плана» здесь буквально нельзя, и оба соседних
варианта плохи:
* рез по 1.746 оставил бы в версии повисший хвост старой фразы
  «по вероятности» — без начала предложения;
* рез по 3.873 выбросил бы вместе с хвостом и «итак есть 100 заключённых»,
  то есть постановку задачи: ни один из новых дублей сотню не вводит.

Рез поставлен по речевой границе **2.833** (кадр 85) — в тишине 2.449–2.842,
за 9мс до атаки слова «Итак» (огибающая: 2.80 −73 дБ, 2.83 −42, 2.84 −28).
От старого хука не остаётся ни кадра лица, ни одного его субтитра: блок
«по вероятности» срезан на 2.449, следующий стартует только на 2.842.

Чтобы шов не читался, блок хука **кончается тем же планом `people`**, что
идёт в теле: последние M кадров блока — та же решётка, отрисованная той же
функцией. Каскад фигурок кончается на lt≈0.77, а в теле план подхватывается
на lt=1.087, то есть по обе стороны шва решётка уже собрана и неподвижна —
кадры совпадают. Белое «100» всплывает на lt=1.60, то есть уже в теле, и
в блоке хука чисел нет.

Фон-сетка на кадрах `people` считается не от времени версии, а обратным
отсчётом от шва (`t_grid = CUT_A - (N - f)/FPS`): иначе на шве прыгнула бы
фаза дрейфа сетки.

Whoosh на шве НЕ ставится: визуального реза там нет, план продолжается.
Единственный рез блока хука — `A1` -> `people` на 1.967.

--- передача эстафеты -----------------------------------------------------
Тело начинается самостоятельным предложением «итак, есть 100 заключённых
и 100 закрытых коробок»: своё подлежащее, ни одного местоимения, отсылок
к выброшенному тексту нет. Оба новых дубля кончаются словом «задача /
задачка ... по теории вероятности» и ведут ровно туда же, куда вело старое
начало.

--- рез внутри блока хука -------------------------------------------------
Оба дубля — сплошная речь без единой паузы (silencedetect не нашёл ничего),
поэтому рез 1.967 поставлен на ближайшую границу слов:
    hook1  «вероятности» / «совсем» — 1.953, промах 14мс;
    hook2  «теории» / «вероятностей» — 1.956, промах 11мс.
Оба промаха меньше 20мс, `speech_sync` их не считает попаданием в середину слова.

--- формат хуков ----------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709, 30 fps. Приведение цветового пространства
не требуется, HDR ни в одном файле нет.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 372  линия глаз 684  ширина лица 273
    hook1     cx 377  линия глаз 685  ширина лица 269
    hook2     cx 381  линия глаз 695  ширина лица 280
Кропом это не правится: оба дубля идут через те же `FRAMINGS`, что и основной
исходник, числа названы в сдаче.

--- пауза хук -> тело -------------------------------------------------------
Оба дубля кончаются речью у самого конца файла (хвостовой тишины нет:
hook1 3.0006с при длине 3.0006, hook2 2.7177 при длине 2.7177). Пауза
набирается кадрами блока хука: hook1 94 кадра -> 0.141с, hook2 86 кадров ->
0.158с. Оба в диапазоне 0.10–0.20с из delivery-specs §6. Кадры сверх A-roll
добиваются графическим планом, лицо не морозится.
"""
import json
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import *
import render33 as R
import storyboard33 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/33"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 85                                     # первый кадр тела
CUT_V = CUT_F / FPS                            # 2.8333с — картинка тела
CUT_A = CUT_V                                  # 2.8333с — голос тела (тишина, до атаки «Итак»)
PEOPLE_T0 = 1.746                              # начало плана people в теле

TARGET_LUFS = -13.9                            # замер сданного boxes_edit.mp4

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_frames=90,                   # 3.0006с, речь до самого конца файла
        a1_frames=59,                    # рез 1.9667 — граница «вероятности»/«совсем»
        frames=94,                       # 3.1333с — пауза до тела 0.141с
        out=f"{VIDEO_DIR}/boxes_hook1.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«решение этой задачи по теории вероятности совсем не очевидно»",
        face_cx=377, face_eye=685, face_d=269,
        caps=[
            (0.000, 1.007, [("решение этой ", R_, 42), ("задачи", R_, 52)], "A"),
            (1.007, 1.953, [("по теории ", R_, 42), ("вероятности", S_, 52)], "A"),

            (1.967, 3.000, [("совсем не ", R_, 42), ("очевидно", S_, 54)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_frames=81,                   # 2.7177с, речь до самого конца файла
        a1_frames=59,                    # рез 1.9667 — граница «теории»/«вероятностей»
        frames=86,                       # 2.8667с — пауза до тела 0.158с
        out=f"{VIDEO_DIR}/boxes_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«это очень интересная задачка по теории вероятностей»",
        face_cx=381, face_eye=695, face_d=280,
        # 1.956-2.718 «вероятностей» — одно слово, на фразу из 2-4 слов не бьётся;
        # план несёт графика (решётка из ста фигурок), субтитра на нём нет
        caps=[
            (0.000, 0.474, [("это ", R_, 42), ("очень", R_, 52)], "A"),
            (0.474, 1.524, [("интересная ", R_, 42), ("задачка", S_, 54)], "A"),

            (1.524, 1.956, [("по ", R_, 42), ("теории", R_, 52)], "A"),
        ],
    ),
}


def shots_of(cfg):
    a1 = cfg["a1_frames"] / FPS
    return [(0.000, a1, "A1", {}),
            (a1, cfg["frames"] / FPS, "people", {})]


def card_rgb(path, t, kind="A1"):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    card = np.array(R.source_card(fr, kind))
    return tuple(int(v) for v in card.reshape(-1, 3).mean(axis=0))


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    shots = shots_of(cfg)
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, s=shots: next((x for x in s if x[0] <= t < x[1]), s[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    a1n = cfg["a1_frames"]
    cap = cv2.VideoCapture(cfg["src"])

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet, last_img, held = [], None, 0
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last_img = fr
        else:
            fr = last_img          # только под графическим планом, лицо не морозим
            held += 1
        t0, t1, kind, prm = R.shot_at(t)
        if not ok and kind in R.FACE_KINDS:
            raise RuntimeError(f"A-roll кончился на кадре {f} под планом с лицом {kind}")

        # сетка-подложка идёт обратным отсчётом от шва — иначе на шве
        # прыгнет фаза её дрейфа
        t_grid = t if kind in R.FACE_KINDS else CUT_A - (nf_hook - f) / FPS
        canvas = R.background(kind, prm, fr, t_grid)
        gl = R.graphics_layer(kind, t - t0)
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 4 == 0:
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
    print(f"видео: хук {nf_hook} кадров (A-roll {a1n}, графика {nf_hook - a1n}, "
          f"из них {held} сверх дубля) + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook33.wav"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path]
    if start:
        cmd += ["-ss", f"{start:.5f}"]
    cmd += ["-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
    subprocess.run(cmd, check=True)
    d = read_wav(tmp)
    os.remove(tmp)
    return d


def speech_rms(x):
    m = np.abs(x).mean(axis=1)
    sel = m > m.max() * 0.12
    return float(np.sqrt((x[sel] ** 2).mean()))


def build_audio(key, cfg, nf_hook, n_body, vid, out):
    shots = shots_of(cfg)
    hook = wav_of(cfg["src"])
    body = wav_of(SRC, CUT_A)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round(nf_hook / FPS * SR))
    n_bod = int(round(n_body / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
    if len(body) < n_bod:
        body = np.vstack([body, np.zeros((n_bod - len(body), 2), np.float32)])
    else:
        body = body[:n_bod]
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

    seam = nf_hook / FPS
    dt = seam - CUT_V              # SFX тела считаются от КАРТИНКИ
    wh, im, tk = low_whoosh(), impact(), tick()
    for t0 in [s[0] for s in shots[1:]]:            # единственный рез блока хука
        add(wh, t0, peak * 0.040)
    # на шве whoosh не ставится: визуального реза там нет, план people продолжается
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > CUT_V + 0.01:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT_V:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT_V:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0, _t1, kind, _p in shots:                 # каскад решётки в блоке хука
        if kind not in R.FACE_KINDS:
            for k in range(3):
                add(tk, t0 + 0.12 + k * 0.085, peak * 0.030)

    music = cfg["music"]
    mw = f"{BUILD}/assets/_music_{os.path.basename(music).split('.')[0]}.wav"
    if not os.path.exists(mw):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", music,
                        "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
    mus = read_wav(mw)
    if mus.shape[1] == 1:
        mus = np.repeat(mus, 2, axis=1)
    if len(mus) >= n:
        track = mus[:n].copy()
        print("музыка: %s, петли нет" % os.path.basename(music))
    else:
        xf = int(0.25 * SR)
        core, tail = mus[:len(mus) - xf], mus[len(mus) - xf:]
        ramp = np.linspace(0, 1, xf)[:, None]
        loop = core.copy()
        loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
        track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n]
        print("музыка: %s, петля %.2fс x%.1f" %
              (os.path.basename(music), len(loop) / SR, n / len(loop)))
    vr = np.sqrt((voice ** 2).mean()) + 1e-9
    mr = np.sqrt((track ** 2).mean()) + 1e-9
    track = track * (vr / mr) * (10 ** (-19 / 20))
    fi, fo = int(0.8 * SR), int(2.0 * SR)
    track[:fi] *= np.linspace(0, 1, fi)[:, None]
    track[-fo:] *= np.linspace(1, 0, fo)[:, None]
    bed += track.astype(np.float32)

    mix = voice + bed
    m = np.abs(mix).max()
    if m > 0.99:
        mix *= 0.99 / m
    mix_path = f"{BUILD}/assets/_mix_{SB.TAG}{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    src = f"{BUILD}/assets/_video_{SB.TAG}{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_{SB.TAG}{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15", "-preset", "medium",
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
    tmp = f"{BUILD}/assets/_video_{SB.TAG}{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 1.0),
          "· основной дубль", card_rgb(SRC, 1.0))

    build_audio(key, cfg, nf_hook, n_body, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook33_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    else:
        main(k)

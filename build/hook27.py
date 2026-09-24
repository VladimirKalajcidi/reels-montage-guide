"""Ролик 27 («формула шнуровки»), альтернативные хуки.

Тело ролика (всё с кадра 122) не пересобирается — берётся готовыми кадрами
из assets/_video_lace.mp4 и склеивается за новым хуком. Заново рисуется
только хук.

    python3 hook27.py h1
    python3 hook27.py h2

Точка реза: кадр 122 (4.067с) — граница планов `stock` -> `plane`, то есть
конец старого начала «эта формула считает площадь любого многоугольника на
клетке. Сначала нужно». Планы 1-3 старого хука (A1, `shape`, сток-вставка
с тетрадью) и их субтитры выброшены целиком.

`CUT_F = ceil(4.050 * 30 - 1e-6)` = 122, а не `round` = 122 (здесь оба метода
совпадают: 4.050с ровно на середине между кадрами 121 и 122 — 4.05*30=121.5,
и `ceil` берёт следующий целый кадр 122, тот же, что дал бы обычный `round`
при округлении вверх). Кадр 121 (t=4.0333) — ещё последний кадр
ВЫБРАСЫВАЕМОГО плана `stock`; кадр 122 (t=4.0667) — первый кадр плана
`plane`. Проверка машинная, в qa27hook.old_hook_gone.

Передача эстафеты. Тело начинается с «ввести координатную плоскость и
записать координаты вершин ЭТОГО многоугольника» — местоимение «этого»
нуждается в антецеденте. Оба новых хука его дают явно: hook1 заканчивается
на «...любого многоугольника, нарисованного на клетчатой бумаге?», hook2 —
на «...посчитать площадь любого многоугольника, нарисованного на клетчатой
бумаге.» Многоугольник как объект уже назван словом в конце каждого хука —
подставлять ничего не пришлось.

Формат обоих дублей совпадает с основным исходником по всем тегам: 720x1280,
yuv420p, color_range=tv, bt709/bt709/bt709, 30 fps. Приведение цветового
пространства не требуется.

Реальное число кадров в контейнерах (`ffprobe -count_frames`) расходится
с метаданными `nb_frames` — 168 у hook1 (не 198) и 190 у hook2 (не 206).
Использовано реальное число: коду нужно ровно столько кадров, сколько
дубль фактически отдаёт.

Крупность дублей (детектор, после hflip, медиана по всему дублю):
    основной  cx 347  линия глаз 677  ширина лица 313
    hook1     cx 347  линия глаз 670  ширина лица 347  (+11% шире — ближе к камере)
    hook2     cx 343  линия глаз 685  ширина лица 329  (+5% шире)
Кропом это НЕ правится: центр и линия глаз внутри разброса детектора
(±6-9px), но ширина лица заметно больше на обоих дублях — камера стояла
ближе. Числа названы в сдаче, решение — за съёмочной стороной.

Графика хука. Оба хука договаривают ту же вводную фразу, что звучала в
старом начале («эта формула считает площадь любого многоугольника... на
клетке/бумаге»), поэтому визуальная интрига — тот же план `shape`
(многоугольник обводится и заливается на сетке, без осей) и та же
стоковая вставка `stock_50113` (рука чертит фигуру в тетради в клетку),
что стояли в старом начале и стоят в теле. Ни одного нового плана
не нарисовано — только новая раскладка резов под новый текст.

Уникализация (START-HERE, «Уникализация версий под площадку»):
    v1 lace_edit   song3, без деформации — эталон, файл не трогается
    v2 lace_hook1  song1, кроп x0.98
    v3 lace_hook2  song2, ускорение x1.02 (картинка setpts + звук atempo, fps=30)
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
import render27 as R
import storyboard27 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/27"
BODY = f"{BUILD}/assets/_video_lace.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_T = 4.050                                  # граница планов stock -> plane
CUT_F = math.ceil(CUT_T * FPS - 1e-6)          # 122 — первый кадр тела
CUT = CUT_F / FPS                              # 4.0667с
TARGET_LUFS = -14.2                            # замер сданного lace_edit.mp4

# планов с полной длительностью анимации на масштабирование времени не нужно:
# `shape` укладывается в отведённые интервалы обоих хуков без ускорения
GFX_FULL = {}

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_ss=0.0,
        frames=168,                             # реальных кадров дубля (ffprobe -count_frames)
        out=f"{VIDEO_DIR}/lace_hook1.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«а вы знали, что существует формула, которая считает площадь "
              "любого многоугольника, нарисованного на клетчатой бумаге?»",
        face_cx=347, face_eye=670, face_d=347,
        # структура: лицо -> многоугольник обводится на сетке -> тетрадь в клетку
        shots=[
            (0.00, 1.78, "A1", {}),
            (1.78, 3.88, "shape", {}),
            (3.88, 5.60, "stock", {"clip": "50113", "ss": 0.0}),
        ],
        caps=[
            (0.00, 0.58, [("а вы знали", R_, 50)], "A"),
            (0.58, 1.78, [("что существует ", R_, 46), ("формула", S_, 58)], "A"),

            (1.78, 2.86, [("которая считает", R_, 48)], "G"),
            (2.86, 3.88, [("площадь ", S_, 58), ("любого многоугольника", R_, 44)], "G"),

            (3.88, 5.60, [("нарисованного на клетчатой ", R_, 42),
                          ("бумаге", S_, 52)], "B"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_ss=0.0,
        frames=190,                             # реальных кадров дубля (ffprobe -count_frames)
        out=f"{VIDEO_DIR}/lace_hook2.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 3: ускорение 1.02 картинкой и звуком разом; fps=30 обязателен,
        # иначе на выходе 30.6 fps (нестандартный фреймрейт, площадка пережмёт).
        post_v="setpts=PTS/1.02,fps=30",
        post_a="atempo=1.02",
        title="«не нужно запоминать никакие формулы кроме этой, чтобы "
              "посчитать площадь любого многоугольника, нарисованного "
              "на клетчатой бумаге»",
        face_cx=343, face_eye=685, face_d=329,
        shots=[
            (0.00, 1.90, "A1", {}),
            (1.90, 3.20, "A2", {}),
            (3.20, 4.64, "shape", {}),
            (4.64, 190 / 30, "stock", {"clip": "50113", "ss": 0.0}),
        ],
        caps=[
            (0.00, 0.42, [("не нужно", R_, 50)], "A"),
            (0.42, 1.90, [("запоминать никакие ", R_, 44), ("формулы", S_, 54)], "A"),

            (1.90, 2.62, [("кроме этой", R_, 48)], "A"),
            (2.62, 3.20, [("чтобы посчитать", R_, 50)], "A"),

            (3.20, 4.64, [("площадь ", S_, 58), ("любого многоугольника", R_, 44)], "G"),

            (4.64, 190 / 30, [("нарисованного на клетчатой ", R_, 42),
                              ("бумаге", S_, 52)], "B"),
        ],
    ),
}


def lts_for(cfg):
    """Масштаб времени анимации на каждый графический план хука.

    max(1.0, ...) — анимацию можно только ускорить, чтобы она успела доиграть
    до реза. Замедлять нельзя: тот же план в теле идёт своим темпом.
    """
    out = {}
    for t0, t1, kind, _p in cfg["shots"]:
        if kind in GFX_FULL:
            full, hold = GFX_FULL[kind]
            out[kind] = max(1.0, full / max(0.4, (t1 - t0) - hold))
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


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_lace.mp4.

    A-roll читается из .mov последовательно, без промежуточного кодека:
    форматы дубля и основного исходника совпадают, приводить нечего.
    """
    lts = lts_for(cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["src"])
    for _ in range(int(round(cfg["src_ss"] * FPS))):      # отбрасываем голову
        cap.read()
    stock = R.StockReader()

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

        canvas = R.background(kind, prm, fr, t,
                              lambda: stock.frame(prm["clip"], prm.get("ss", 0), t - t0))
        gl = R.graphics_layer(kind, (t - t0) * lts.get(kind, 1.0))
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
    stock.release()

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
    print(f"видео: хук {nf_hook} кадров (из них {held} добито графикой) "
          f"+ тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    """-ss после -i: точный отброс, чтобы звук встал кадр в кадр с картинкой."""
    tmp = f"{BUILD}/assets/_tmp_hook27.wav"
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


def build_audio(key, cfg, nf_hook, vid, out):
    """Дорожка пересобирается целиком: голос хука + голос тела с точки реза,
    постель со сдвигом, музыка с нуля, потом loudnorm."""
    hook = wav_of(cfg["src"], cfg["src_ss"])
    body = wav_of(SRC, CUT)

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
    for t0 in [s[0] for s in cfg["shots"][1:]]:      # резы внутри хука
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)             # шов «хук -> тело»
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > CUT + 0.01:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= CUT:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= CUT:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0, _t1, kind, _p in cfg["shots"]:       # каскады графики самого хука
        if kind in SB.GRID_KINDS:
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
    mix_path = f"{BUILD}/assets/_mix_lace{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_lace{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_lace{key}_post.mp4"
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
    tmp = f"{BUILD}/assets/_video_lace{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    # расхождение дублей — измеряем и называем, а не чиним кропом
    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 1.0),
          "· основной дубль", card_rgb(SRC, 1.0))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook27_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_lace{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

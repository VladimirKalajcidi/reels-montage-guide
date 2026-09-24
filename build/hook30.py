"""Ролик 30 («парадокс дружбы»), альтернативные хуки.

Тело ролика (всё с кадра 124) не пересобирается — берётся готовыми кадрами из
`assets/_video_friendship.mp4` и склеивается за новым хуком. Заново рисуется
только блок хука.

    python3 hook30.py h1
    python3 hook30.py h2

--- точка реза ------------------------------------------------------------
Старое начало — «у ваших друзей в среднем больше друзей чем у вас и это
математический факт» (0.12–4.11) — занимало два плана: `A1` 0.00–1.53,
`netYou` 1.53–4.11. Граница плана 4.111 (`netYou` -> `A1` «а причина этому
очень простая») и есть точка реза: `CUT_F = ceil(4.111 * 30) = 124`.
Планы 1–2 и их пять субтитров выброшены целиком.

Граница 4.111 у whisper совпадает с огибающей день в день: `ffmpeg
silencedetect` на source.mov даёт тишину ровно 4.110952–4.376871 — слово
«факт.» кончается, где кончается и звук. Поправка (`head_trim_f`), в отличие
от ролика 28, не понадобилась.

--- передача эстафеты -----------------------------------------------------
Тело начинается самостоятельным предложением с собственным подлежащим:
«а причина этому очень простая». «Этому» отсылает к самому явлению (у друзей
в среднем больше друзей), а оно заново проговорено в обоих новых хуках —
дефицита антецедента нет.

hook1 кончается на «а математик», hook2 — на «на самом деле вам не кажется».
Оба ведут ровно туда же, куда вело старое начало: дальше зритель ждёт
объяснения, и получает «а причина этому очень простая».

--- формат хуков ----------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709, 30 fps. Приведение цветового пространства
не требуется.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 402  линия глаз 688  ширина лица 288
    hook1     cx 423  линия глаз 690  ширина лица 296  (+21px, +2.9%)
    hook2     cx 402  линия глаз 699  ширина лица 287  (совпадает по x)
Кропом это НЕ правится: оба дубля идут через те же FRAMINGS, что и основной
исходник, числа названы в сдаче.

--- графика хука ----------------------------------------------------------
В старом начале стоял план `netYou` — небольшая эго-сеть, вы в центре
и несколько друзей вокруг. Это визуальная интрига ролика, сохранена в обоих
новых хуках без изменений (ни одного нового плана не нарисовано).

--- пауза хук -> тело -------------------------------------------------------
Оба дубля обрываются речью почти у самого конца файла (собственного хвостового
запаса тишины нет или он меньше кадра), поэтому пауза «хук -> тело» почти
целиком берётся из готовой паузы тела: слово «А» (начало «а причина...»)
звучит в теле только с 4.377, то есть через 0.244с после точки реза 4.133.
Это чуть больше диапазона 0.10-0.20с из delivery-specs §6 — число измерено
и не подгонялось искусственной обрезкой чужого хвоста; короче не сделать,
не подрезав последнее слово хука.
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
import render30 as R
import storyboard30 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/30"
BODY = f"{BUILD}/assets/_video_friendship.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_T = 4.111                                  # граница планов netYou -> A1
CUT_F = math.ceil(CUT_T * FPS - 1e-6)          # 124 — граница плана в кадрах
CUT = CUT_F / FPS                              # 4.1333с

TARGET_LUFS = -14.46                           # замер сданного friendship_edit.mp4

GFX_FULL = {}          # netYou укладывается в ~0.72с — быстрее любого хука, ускорять не нужно

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # дубль реально декодируется как 120 кадров (4.000с) при 30fps —
        # метаданные контейнера (128) не совпадают с числом декодируемых
        # кадров, см. шапку align30hooks.py; речь звучит до 4.010, разница
        # меньше кадра.
        src_ss=0.0,
        frames=120,
        head_trim_f=0,
        out=f"{VIDEO_DIR}/friendship_hook1.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«у ваших друзей больше друзей, чем у вас, и это не я так "
              "говорю, а математик»",
        face_cx=423, face_eye=690, face_d=296,
        shots=[
            (0.000, 1.181, "A1", {}),
            (1.181, 120 / 30, "netYou", {}),
        ],
        caps=[
            (0.000, 1.181, [("у ваших ", R_, 44), ("друзей больше", R_, 50)], "A"),

            (1.181, 2.198, [("друзей чем ", R_, 44), ("у вас", R_, 48)], "G"),
            (2.321, 2.809, [("и это ", R_, 42), ("не я", R_, 46)], "G"),
            (2.809, 3.216, [("так ", R_, 44), ("говорю", R_, 50)], "G"),
            (3.338, 120 / 30, [("а ", R_, 42), ("математик", S_, 56)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # дубль декодируется как 157 кадров (5.233с), речь кончается на 5.045.
        # Берём 152 (5.067с) — 0.022с хвоста, не всю тишину до конца файла,
        # иначе пауза хук->тело вышла бы почти вдвое длиннее (см. шапку).
        src_ss=0.0,
        frames=152,
        head_trim_f=0,
        out=f"{VIDEO_DIR}/friendship_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«многим кажется, что у ваших друзей друзей больше, чем у вас, "
              "но на самом деле вам не кажется»",
        face_cx=402, face_eye=699, face_d=287,
        shots=[
            (0.000, 1.113, "A1", {}),
            (1.113, 152 / 30, "netYou", {}),
        ],
        caps=[
            (0.183, 1.113, [("многим ", R_, 44), ("кажется что", R_, 50)], "A"),

            (1.113, 2.128, [("у ваших ", R_, 44), ("друзей", R_, 50)], "G"),
            (2.128, 2.931, [("друзей ", R_, 46), ("больше", R_, 50)], "G"),
            (2.995, 3.544, [("чем у ", R_, 44), ("вас", R_, 48)], "G"),
            (3.671, 4.284, [("но на ", R_, 44), ("самом", R_, 48)], "G"),
            (4.284, 152 / 30, [("деле вам не ", R_, 42), ("кажется", S_, 52)], "G"),
        ],
    ),
}


def cut_f_of(cfg):
    return CUT_F + cfg.get("head_trim_f", 0)


def cut_of(cfg):
    return cut_f_of(cfg) / FPS


def lts_for(cfg):
    out = {}
    for t0, t1, kind, _p in cfg["shots"]:
        if kind in GFX_FULL:
            full, hold = GFX_FULL[kind]
            out[kind] = max(1.0, full / max(0.4, (t1 - t0) - hold))
    return out


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
    lts = lts_for(cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["src"])
    for _ in range(int(round(cfg["src_ss"] * FPS))):
        cap.read()

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

        canvas = R.background(kind, prm, fr, t)
        gl = R.graphics_layer(kind, (t - t0) * lts.get(kind, 1.0))
        if gl is not None:
            canvas.alpha_composite(gl)
        cl = R.caption_layer(t)
        if cl is not None:
            canvas.alpha_composite(cl)
        rgb = canvas.convert("RGB")
        ff.stdin.write(rgb.tobytes())
        if f % 5 == 0:
            sheet.append((t, np.array(rgb)))
    cap.release()

    body = cv2.VideoCapture(BODY)
    body.set(cv2.CAP_PROP_POS_FRAMES, cut_f_of(cfg))
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
    tmp = f"{BUILD}/assets/_tmp_hook30.wav"
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
    cut = cut_of(cfg)
    hook = wav_of(cfg["src"], cfg["src_ss"])
    body = wav_of(SRC, cut)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round(nf_hook / FPS * SR))
    if len(hook) < n_hook:
        hook = np.vstack([hook, np.zeros((n_hook - len(hook), 2), np.float32)])
    else:
        hook = hook[:n_hook]
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

    dt = nf_hook / FPS - cut
    wh, im, tk = low_whoosh(), impact(), tick()
    for t0 in [s[0] for s in cfg["shots"][1:]]:
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)             # шов «хук -> тело»
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > cut + 0.01:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= cut:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= cut:
            for k in range(3):
                add(tk, t + dt + k * 0.085, peak * 0.030)
    for t0, _t1, kind, _p in cfg["shots"]:
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
    mix_path = f"{BUILD}/assets/_mix_friendship{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    src = f"{BUILD}/assets/_video_friendship{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_friendship{key}_post.mp4"
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
            af += f",volume={post_db:+.2f}dB,alimiter=limit=0.76:level=disabled"
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
    tmp = f"{BUILD}/assets/_video_friendship{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 1.0),
          "· основной дубль", card_rgb(SRC, 1.0))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook30_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_friendship{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

"""Ролик 32 («100 заключённых и шляпы»), альтернативные хуки.

Тело ролика (всё с кадра 165) не пересобирается — берётся готовыми кадрами из
`assets/_video_hats.mp4` и склеивается за новым хуком. Заново рисуется только
блок хука.

    python3 hook32.py h1
    python3 hook32.py h2

--- точка реза ------------------------------------------------------------
Оба новых дубля перекрывают не одну первую фразу, а две: и «представьте 100
заключённых, стоящих в ряд», и «каждому надевают случайно чёрную или белую
шляпу». Значит выбрасывается всё до конца плана `hats` — планы 1-4
(`A1`, `row`, `A1`, `hats`, 0.000-5.499) и их шесть субтитров.

Граница плана 5.499 → `CUT_F = ceil(5.499 * 30) = 165`, то есть 5.500с.
Whisper и огибающая здесь совпадают: `ffmpeg silencedetect` на source.mov даёт
тишину ровно 5.499252-5.900272 — слово «шляпу» кончается там же, где кончается
звук, и рез 5.500 попадает в тишину, а не в атаку слова.

--- передача эстафеты -----------------------------------------------------
Тело начинается с «каждый видит шляпы людей перед собой». «Каждый» и «шляпы»
опираются на заключённых и на шляпы — и то и другое заново проговорено в обоих
новых хуках, дефицита антецедента нет.

Пауза «хук → тело» — 0.400с. Это не растянутый шов: ровно столько же длится
пауза между этими же двумя предложениями в самом исходнике (5.499 → 5.900),
и она целиком приходит из тела. У этого дубля межфразовые паузы вообще
0.26-0.46с, так что 0.40 — его норма, а не брак монтажа.

--- формат хуков ----------------------------------------------------------
Оба дубля совпадают с основным исходником по всем тегам: 720x1280, yuv420p,
color_range=tv, bt709/bt709/bt709, 30 fps. Приведение цветового пространства
не требуется. Сцена та же (тёмная стена слева, светлая справа), штатный
`hflip` подходит обоим.

Крупность дублей (детектор, после hflip, среднее по дублю):
    основной  cx 390  линия глаз 648  ширина лица 283
    hook1     cx 377  линия глаз 686  ширина лица 285   (-13 / +38 / +2)
    hook2     cx 381  линия глаз 687  ширина лица 288   (-9 / +39 / +5)
Ширина лица сходится в пределах 2%, но голова в обоих хуках стоит на ~38px
ниже: при штатном кропе линия глаз садится на 46% высоты карточки вместо 42%.
Кропом это НЕ правится — оба дубля идут через те же FRAMINGS, что и основной
исходник, числа названы в сдаче.

--- графика хука ----------------------------------------------------------
В старом начале стояли планы `row` (шеренга + белое 100) и `hats` (шляпы
падают на головы). Это визуальная завязка ролика, она сохранена в обоих новых
хуках без единого нового рисунка. Добавлены только два служебных состояния
того же плана: `rowBuild` (шеренга собирается, числа ещё нет) и `rowNum`
(шеренга уже стоит, всплывает 100) — они нужны там, где между сборкой ряда
и словом «100» вклинивается план с лицом.

Хуки длиннее старого начала, поэтому анимации растянуты множителем
`gfx_scale` (<1 замедляет): иначе `hats` отыгрывает за 0.99с и стоит
неподвижно ещё две секунды.
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
import render32 as R
import storyboard32 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/32"
BODY = f"{BUILD}/assets/_video_{SB.TAG}.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_T = 5.499                                  # граница планов hats -> sees
CUT_F = math.ceil(CUT_T * FPS - 1e-6)          # 165 — граница плана в кадрах
CUT = CUT_F / FPS                              # 5.5000с

TARGET_LUFS = -14.1                            # замер сданного hats_edit.mp4

R_ = "r"
S_ = "s"


# --- служебные состояния плана `row` для хуков ------------------------------
def g_row_build(lay, lt):
    """Шеренга собирается слева направо. Числа нет — оно придёт позже,
    когда его произнесут."""
    z = R.Z()
    prog = R.stagger(lt, SB.N, t0=0.02, step=0.035, dur=0.20)
    for i in range(SB.N):
        R.person(z, i, a=prog[i], sc=0.55 + 0.45 * prog[i])
    lay.alpha_composite(z.out())


def g_row_num(lay, lt):
    """Шеренга уже стоит (её собрали на предыдущем графическом плане),
    всплывает белое 100 — резким попом сразу после произнесённого «100»."""
    z = R.Z()
    for i in range(SB.N):
        R.person(z, i)
    lay.alpha_composite(z.out())
    q = R.clamp01((lt - 0.10) / 0.10)
    if q > 0:
        R.draw_number(lay, str(SB.PRISONERS), SB.NUM_XY, SB.NUM_SIZE,
                      color=WHITE, glow=WHITE, glow_a=0.55,
                      scale=0.6 + 0.4 * ease_out(q))


R.GFX["rowBuild"] = g_row_build
R.GFX["rowNum"] = g_row_num


HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # контейнер обещает 284 кадра, декодируется 258 (8.6000с) — речь идёт
        # до самого конца файла, хвостовой тишины у дубля нет
        src_ss=0.0,
        frames=258,
        head_trim_f=0,
        out=f"{VIDEO_DIR}/hats_hook1.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«это очень интересная задачка на логику: представьте, что перед "
              "вами стоят 100 заключённых»",
        face_cx=377, face_eye=686, face_d=285,
        gfx_scale={"hats": 0.42},
        num_reveals=[4.147],
        shots=[
            (0.000, 1.519, "A1", {}),
            (1.519, 2.618, "rowBuild", {}),
            (2.618, 4.040, "A1", {}),
            (4.040, 5.662, "rowNum", {}),
            (5.662, 258 / 30, "hats", {}),
        ],
        caps=[
            (0.000, 0.434, [("это ", R_, 44), ("очень", R_, 50)], "A"),
            (0.434, 1.519, [("интересная ", R_, 42), ("задачка", S_, 54)], "A"),

            (1.519, 2.061, [("на ", R_, 42), ("логику", S_, 54)], "G"),

            (2.618, 3.276, [("представьте ", R_, 42), ("что", R_, 48)], "A"),
            (3.276, 4.040, [("перед вами ", R_, 42), ("стоят", R_, 50)], "A"),

            # 4.040-4.147 — «100»: число несёт график, субтитра нет
            (4.147, 5.484, [("заключённых стоящих ", R_, 40), ("в ряд", R_, 52)], "G"),

            (5.662, 6.933, [("каждому надевают ", R_, 42), ("случайно", S_, 52)], "G"),
            (6.933, 7.600, [("либо ", R_, 42), ("чёрную", R_, 52)], "G"),
            (7.600, 8.600, [("либо белую ", R_, 42), ("шляпу", R_, 54)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # контейнер обещает 204 кадра, декодируется 178 (5.9333с); речь идёт
        # до 5.930, хвоста тишины 0.003с
        src_ss=0.0,
        frames=178,
        head_trim_f=0,
        out=f"{VIDEO_DIR}/hats_hook2.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«представьте тюрьму, 100 заключённых, они стоят в ряд»",
        face_cx=381, face_eye=687, face_d=288,
        gfx_scale={"row": 1.25, "hats": 0.50},
        num_reveals=[1.649],
        shots=[
            (0.000, 1.153, "A1", {}),
            (1.153, 2.200, "row", {}),
            (2.200, 3.633, "A1", {}),
            (3.633, 178 / 30, "hats", {}),
        ],
        caps=[
            (0.000, 0.859, [("представьте ", R_, 44), ("тюрьму", S_, 54)], "A"),
            # 0.943-1.153 — «100»: число несёт график, субтитра нет

            (1.153, 2.200, [("заключённых ", R_, 42), ("они", R_, 48)], "G"),

            (2.200, 2.829, [("стоят ", R_, 42), ("в ряд", R_, 52)], "A"),
            (3.174, 3.633, [("и ", R_, 42), ("каждому", R_, 50)], "A"),

            (3.633, 4.858, [("надевают либо ", R_, 42), ("чёрную", R_, 52)], "G"),
            (4.858, 5.930, [("либо белую ", R_, 42), ("шляпу", R_, 54)], "G"),
        ],
    ),
}


def cut_f_of(cfg):
    return CUT_F + cfg.get("head_trim_f", 0)


def cut_of(cfg):
    return cut_f_of(cfg) / FPS


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
    scale = cfg.get("gfx_scale", {})

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
        gl = R.graphics_layer(kind, (t - t0) * scale.get(kind, 1.0))
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
    tmp = f"{BUILD}/assets/_tmp_hook32.wav"
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
    for t in cfg.get("num_reveals", []):             # импакт под 100 в хуке
        add(im, t, peak * 0.070)
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

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    csp = f"{VIDEO_DIR}/contact-sheet-hook{cfg['version'] - 1}.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_{SB.TAG}{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

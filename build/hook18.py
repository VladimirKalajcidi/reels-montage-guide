"""Ролик 18 («правило високосного года»), альтернативные хуки.

Тело ролика не пересобирается — берётся готовыми кадрами из assets/_video_leap.mp4
и склеивается за новым хуком одним проходом кодека. Заново рисуется только блок
хука: свой A-roll, свои субтитры и та же графика, что стояла в старом начале
(`div4` — шкала лет, где каждый четвёртый год получает день; плюс та же стоковая
вставка с классом).

    python3 hook18.py h1
    python3 hook18.py h2

Точки реза у хуков РАЗНЫЕ, потому что дубли пересказывают разный объём тела.
Владелец фразу для выброса не называл, поэтому она выведена из самих дублей:

* h1 «всем нам врали в школе, ведь всем же говорили, что год високосный, если он
  делится на 4. оказывается, это не совсем так» — заявляет всё то же, что старое
  начало (0.00–2.98), бит «всем обычно говорят … на четыре» (2.98–6.60) и бит
  «но это правило не полное» (6.60–8.08). Режем по **12.84** (граница планов
  drift -> A2): тело начинается с «дело в том, что год длится неровно».

  Почему не по 8.08. Оставить 6.60–8.08 нельзя: «оказывается, это не совсем так»
  и сразу «но это правило не полное» — заикание. Но и рез по 8.08 не годится:
  тело тогда начинается с «и если бы ИМ пользовались без исключений», а слова
  «правило» в этом дубле нет вообще — местоимению не на что опереться, и фраза
  повисает («чем пользовались?»). Правка владельца: выбросить всё это
  предложение целиком, вместе с «календарь со временем начал бы немного
  съезжать», то есть планы noexc и drift (8.08–12.84). Слово «правило» дальше
  звучит само на 24.02 («правило каждый четвёртый год слегка перебарщивает»),
  там контекст уже есть.

* h2 «большинство людей неправильно понимают, что такое високосный год. всем же
  в школе говорили, что год високосный, если он делится на 4» — доходит ровно до
  конца бита «на четыре» и не трогает «но это правило не полное». Режем по 6.60
  (граница планов div4 -> A1): тело начинается с «но это правило не полное»,
  что прямо продолжает хук.

Пауза между хуком и телом. В обоих дублях речь идёт до самого конца файла, а
первое слово тела стоит ровно в кадре реза, поэтому длина блока хука задана
кадрами с запасом: h1 — 200 кадров (хвост 0.19с), h2 — 204 (хвост 0.16с).
Хвост добит графическим планом `div4`, а не стоп-кадром лица: A-roll там уже
кончился, но сетку рисует движок.

Формат. Оба хука 720x1280 yuv420p tv bt709 — байт в байт те же теги, что у
основного исходника. Приведение формата не нужно вообще, хук идёт через штатный
тракт A-roll без единой поправки. Зеркальность: рука с петличкой входит в сырой
кадр справа во всех трёх файлах, штатный hflip подходит всем.

Расхождение дублей измерено и названо в сдаче, а не подчищено кропом:
                 лицо d   линия глаз   средний RGB карточки A
  основной дубль   302        640           124.6 / 112.0 / 105.9
  hook1            313        648           122.3 / 109.5 / 102.9
  hook2            316        636           123.8 / 110.2 / 103.8
Крупность выше на 3.6% и 4.5%, экспозиция сходится в пределах 3 единиц по каналу
(норма START-HERE — до ~10). Кропом это не правится.
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
import render18 as R
import storyboard18 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/18"
BODY = f"{BUILD}/assets/_video_leap.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

TARGET_LUFS = -14.1                 # замер уже сданного leap_edit.mp4

STOCK = {"clip": "6636", "ss": 0.5}      # та же вставка, что в старом начале

# (длительность полной анимации, запас до реза) — движение обязано доиграть
# до реза, а не оборваться на полудвижении.
GFX_FULL = {"div4": (2.25, 0.12)}

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        src_ss=0.0,                 # речь с 0.16с, тишины в голове нет
        frames=200,                 # 6.667с; речь до 6.48 -> хвост 0.19с
        # ceil, а не round: границы планов стоят на дробных кадрах (12.84с =
        # кадр 385.2), и round оставил бы в сборке последний кадр выброшенного
        # плана. Проверено глазами на стыке — так уже прилетал лишний кадр.
        cut_f=math.ceil(12.84 * 30 - 1e-6),  # 386 -> первый кадр плана A2
        # Звук тела берём на 0.087с раньше кадра реза и кладём в тишину хвоста
        # хука. Слово «дело» начинается на 12.825, то есть ДО кадра реза 12.8667:
        # взять звук ровно с реза значило бы срезать взрыв «д». Сдвиг компенсируем
        # укорочением хвоста хука ровно на ту же величину, поэтому A/V-синхрон
        # не едет ни на кадр.
        body_ss=12.78,
        out=f"{VIDEO_DIR}/leap_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«всем нам врали в школе, ведь всем же говорили, что год високосный, "
              "если он делится на 4. оказывается, это не совсем так»",
        face_d=313, face_y=648, rgb="122.3/109.5/102.9",
        shots=[
            (0.000, 44 / 30, "A1", {}),
            (44 / 30, 78 / 30, "stock", STOCK),
            (78 / 30, 128 / 30, "A2", {}),
            (128 / 30, 200 / 30, "div4", {}),
        ],
        caps=[
            (0.000, 0.820, [("всем нам ", S, 46), ("врали", "s", 58)], "A"),
            (0.820, 44 / 30, [("в школе", S, 52)], "A"),
            (44 / 30, 78 / 30, [("ведь всем же говорили", S, 44)], "B"),
            (78 / 30, 3.560, [("что год високосный", S, 46)], "A"),
            (3.560, 128 / 30, [("если он делится", S, 46)], "A"),
            (128 / 30, 5.200, [("на четыре", S, 56)], "G"),
            (5.260, 5.700, [("оказывается", "s", 58)], "G"),
            (5.780, 200 / 30, [("это не совсем так", S, 46)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        src_ss=0.0,                 # речь с 0.04с
        frames=204,                 # 6.800с; речь до 6.64 -> хвост 0.16с
        cut_f=math.ceil(6.60 * 30 - 1e-6),  # 198 -> первый кадр плана A1
        out=f"{VIDEO_DIR}/leap_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«большинство людей неправильно понимают, что такое високосный год. "
              "всем же в школе говорили, что год високосный, если он делится на 4»",
        face_d=316, face_y=636, rgb="123.8/110.2/103.8",
        shots=[
            (0.000, 57 / 30, "A1", {}),
            (57 / 30, 103 / 30, "A2", {}),
            (103 / 30, 137 / 30, "stock", STOCK),
            (137 / 30, 204 / 30, "div4", {}),
        ],
        caps=[
            (0.000, 0.820, [("большинство людей", S, 46)], "A"),
            (0.820, 57 / 30, [("неправильно ", "s", 54), ("понимают", S, 44)], "A"),
            (57 / 30, 103 / 30, [("что такое високосный год", S, 42)], "A"),
            (103 / 30, 4.040, [("всем же в школе", S, 46)], "B"),
            (4.040, 137 / 30, [("говорили", S, 52)], "B"),
            (137 / 30, 5.520, [("что год високосный", S, 46)], "G"),
            (5.520, 6.140, [("если он делится", S, 46)], "G"),
            (6.140, 204 / 30, [("на четыре", S, 56)], "G"),
        ],
    ),
}


def lts_for(cfg):
    """Масштаб времени анимации на каждый графический план хука."""
    out = {}
    for t0, t1, kind, _p in cfg["shots"]:
        if kind in GFX_FULL:
            full, hold = GFX_FULL[kind]
            out[kind] = full / max(0.4, (t1 - t0) - hold)
    return out


def card_rgb(path, t):
    """Средний RGB внутри карточки A после штатного тракта A-roll."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    card = np.array(R.source_card(fr, "A1"))
    return tuple(round(float(v), 1) for v in card.reshape(-1, 3).mean(axis=0))


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_leap.mp4."""
    lts = lts_for(cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(cfg["src"])
    if cfg["src_ss"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(cfg["src_ss"] * FPS)))
    stock_cap = None

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "15", "-preset", "slow", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    sheet, last_img = [], None
    for f in range(nf_hook):
        t = f / FPS
        ok, fr = cap.read()
        if ok:
            last_img = fr
        else:
            fr = last_img          # только под графическим планом, лицо не морозим
        t0, t1, kind, prm = R.shot_at(t)

        def read_stock():
            nonlocal stock_cap
            if stock_cap is None:
                stock_cap = cv2.VideoCapture(f"{R.STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = R.background(kind, prm, fr, t, read_stock)
        # lts — масштаб времени анимации: графика тела рассчитана на длину плана
        # в теле, на плане хука другой длины она иначе оборвётся или замрёт
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
    if stock_cap is not None:
        stock_cap.release()

    body = cv2.VideoCapture(BODY)
    body.set(cv2.CAP_PROP_POS_FRAMES, cfg["cut_f"])
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
    """-ss после -i: точный отброс, чтобы звук встал кадр в кадр с картинкой."""
    tmp = f"{BUILD}/assets/_tmp_hook18.wav"
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
    cut = cfg["cut_f"] / FPS
    body_ss = cfg.get("body_ss", cut)
    lead = cut - body_ss          # сколько звука тела взято ДО кадра реза
    hook = wav_of(cfg["src"], cfg["src_ss"])
    body = wav_of(SRC, body_ss)
    if lead:
        print("звук тела взят на %.3fс раньше реза, хвост хука укорочен на столько же"
              % lead)

    g = speech_rms(body) / max(1e-9, speech_rms(hook))
    hook = hook * g
    print("хук: коррекция громкости голоса x%.3f" % g)

    n_hook = int(round((nf_hook / FPS - lead) * SR))
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

    dt = nf_hook / FPS - cut                 # сдвиг тела на новой таймлинии
    wh, im, tk = low_whoosh(), impact(), tick()
    for t0 in [s[0] for s in cfg["shots"][1:]]:
        add(wh, t0, peak * 0.040)
    add(wh, nf_hook / FPS, peak * 0.040)     # склейка хук -> тело
    for t0, _t1, _k, _p in SB.SHOTS:
        if t0 > cut:
            add(wh, t0 + dt, peak * 0.040)
    for t in SB.NUM_REVEALS:
        if t >= cut:
            add(im, t + dt, peak * 0.070)
    for t in SB.CASCADES:
        if t >= cut:
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
    mix_path = f"{BUILD}/assets/_mix_leap{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_leap{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_leap{key}_post.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
                    "-vf", vf, "-an", "-c:v", "libx264", "-crf", "15", "-preset", "slow",
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
    tmp = f"{BUILD}/assets/_video_leap{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    # расхождение дублей — измеряем и называем, а не чиним кропом
    print("средний RGB в карточке A: хук", card_rgb(cfg["src"], 3.0),
          "· основной дубль", card_rgb(SRC, 10.0))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook18_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_leap{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

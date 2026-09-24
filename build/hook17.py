"""Ролик 17 («алгоритм Луна»), альтернативные хуки.

Тело ролика (всё с кадра 312) не пересобирается — берётся готовыми кадрами из
assets/_video_luhn.mp4 и склеивается за новым хуком. Заново рисуется только хук:
свой A-roll, свои субтитры и та же графика, что стояла в старом хуке
(`card16` — номер карты набирается группами, `scan` — по номеру идёт луч и берёт
одну цифру в рамку; плюс та же стоковая вставка с набором номера на ноутбуке).

    python3 hook17.py h1
    python3 hook17.py h2

Точка реза: кадр 312 (10.4000с) — граница планов `card16` -> `check`, то есть
конец фразы «номер банковской карты — это не просто какой-то случайный набор
16-ти цифр» (слово «цифр,» кончается на 10.26, план — на 10.40).

Почему режем именно здесь, а не по концу старого хука на 5.98. Оба новых дубля
заявляют ровно то же, что старая фраза тела на 6.12–10.26: «номер не случайный».
Оставить её — получить заикание в первые же секунды. Поэтому вместе со старым
хуком выброшена и она; тело начинается с «в нём зашита очень простая
математическая проверка». Планы 1–5 старого начала и их субтитры выброшены
целиком, ни одного кадра от них не осталось.

J-cut не нужен: «цифр,» кончается на 10.26, следующее слово «в» начинается на
10.48, рез по 10.40 попадает в паузу и ничего не срезает.

Формат хуков. Дубли сняты 720x1280 в полном диапазоне (`pix_fmt=yuvj420p`,
`color_range=pc`), основной исходник — 1080x1920 `tv`. И размер, и range —
объективные свойства файла, они приводятся всегда (START-HERE, «приведение
формата»), после чего хук идёт через штатный тракт A-roll без единой поправки
под дубль. Голова в хуках сидит заметно ниже, чем в основном дубле, — это НЕ
правится кропом, расхождение измерено и названо в сдаче.
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
import render17 as R
import storyboard17 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/17"
BODY = f"{BUILD}/assets/_video_luhn.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 312                         # первый кадр тела = граница плана card16 -> check
CUT = CUT_F / FPS                   # 10.4000с
TARGET_LUFS = -14.4                 # как в уже сданном luhn_edit.mp4 (замер)

# Зеркальность проверена по кадру: надпись на футболке читается задом наперёд
# и в основном дубле, и в обоих хуках — штатный hflip в source_card подходит всем трём.
# Приведение формата: 720x1280 pc -> 1080x1920 tv. Ничего кроме размера и range.
CONFORM = "scale=1080:1920:flags=lanczos:in_range=pc:out_range=tv,format=yuv420p"

STOCK = {"clip": "17618", "ss": 2.0}     # та же вставка, что в старом хуке

# (длительность полной анимации, запас до реза) — чтобы задать lts под длину
# плана хука: движение обязано доиграть до реза, а не оборваться на полудвижении.
# У `scan` запас 0.32с: рамка и крест — это пуант хука, и если они дорисовываются
# ровно в кадре реза, прочитать их зритель не успевает. Графика при этом всё равно
# заканчивается НА склейке (никуда не пропадает), просто финал успевает постоять.
GFX_FULL = {"card16": (2.15, 0.0), "scan": (1.60, 0.32)}

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # дубль начинается с 2.23с тишины — она срезается; речь 2.36–8.70.
        # 197 кадров: после последнего слова остаётся 0.10с, плюс 0.08с в голове
        # тела = пауза 0.18с (норма 0.10–0.20).
        src_ss=67 / 30,
        frames=197,
        out=f"{VIDEO_DIR}/luhn_hook1.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«а вы знали, что номера банковских карт всех людей обладают "
              "определённым свойством? это сделано не просто так»",
        face_d=480, face_y=783, rgb="112/97/90",
        shots=[
            (0.000, 1.387, "A1", {}),
            (1.387, 3.047, "stock", STOCK),
            (3.047, 4.927, "card16", {}),
            (4.927, 197 / 30, "scan", {}),
        ],
        caps=[
            (0.127, 1.387, [("а вы знали что", S, 46)], "A"),
            (1.387, 2.207, [("номера банковских карт", S, 44)], "B"),
            (2.207, 3.047, [("всех людей", S, 50)], "B"),
            (3.047, 4.187, [("обладают определённым", S, 44)], "G"),
            (4.187, 4.927, [("свойством", "s", 66)], "G"),
            (4.927, 5.587, [("это сделано", S, 48)], "G"),
            (5.587, 197 / 30, [("не просто ", S, 46), ("так", "s", 70)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # тишина в голове 0.87с срезается; речь 0.98–7.74.
        # 209 кадров: хвост 0.09с + 0.08с тела = пауза 0.17с.
        src_ss=26 / 30,
        frames=209,
        out=f"{VIDEO_DIR}/luhn_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«а вы знали, что номер вашей банковской карты — это не просто "
              "какие-то произвольные цифры? достаньте её прямо сейчас и проверьте»",
        face_d=471, face_y=787, rgb="111/98/92",
        shots=[
            (0.000, 1.313, "A1", {}),
            (1.313, 2.693, "stock", STOCK),
            (2.693, 4.753, "card16", {}),
            (4.753, 209 / 30, "scan", {}),
        ],
        caps=[
            (0.113, 1.313, [("а вы знали что", S, 46)], "A"),
            (1.313, 1.873, [("номер вашей", S, 50)], "B"),
            (1.873, 2.693, [("банковской карты", S, 48)], "B"),
            (2.693, 3.273, [("это не просто", S, 46)], "G"),
            (3.273, 4.293, [("какие-то произвольные", S, 44)], "G"),
            (4.293, 4.753, [("цифры", "s", 66)], "G"),
            (4.993, 5.773, [("достаньте её прямо", S, 44)], "G"),
            (5.773, 209 / 30, [("сейчас и ", S, 46), ("проверьте", "s", 60)], "G"),
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


# ----------------------------------------------------------------- A-roll хука
def conform_source(key, cfg):
    """Приведение формата хука к формату основного исходника. Звук не трогаем.

    -ss стоит после -i: точный посекундный отброс кадров, а не прыжок по опорным.
    """
    out = f"{BUILD}/assets/_hook17_{key}_conf.mp4"
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-i", cfg["src"], "-ss", f"{cfg['src_ss']:.5f}",
                        "-vf", CONFORM, "-an", "-color_range", "tv",
                        "-c:v", "libx264", "-crf", "12", "-preset", "medium",
                        "-pix_fmt", "yuv420p", out], check=True)
        print("хук приведён к 1080x1920 tv:", os.path.basename(out))
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
    return tuple(int(v) for v in card.reshape(-1, 3).mean(axis=0))


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    """Кадры хука рисуем, кадры тела берём готовыми из _video_luhn.mp4."""
    aroll = conform_source(key, cfg)
    lts = lts_for(cfg)

    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, shots=cfg["shots"]: next(
        (s for s in shots if s[0] <= t < s[1]), shots[-1])
    R.BLOCKS = R.build_blocks()

    nf_hook = cfg["frames"]
    cap = cv2.VideoCapture(aroll)
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
                stock_cap = cv2.VideoCapture(
                    f"{R.STOCK_DIR}/stock_{prm['clip']}.mp4")
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
    """-ss после -i: точный отброс, чтобы звук встал кадр в кадр с картинкой."""
    tmp = f"{BUILD}/assets/_tmp_hook17.wav"
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
    mix_path = f"{BUILD}/assets/_mix_luhn{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_luhn{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_luhn{key}_post.mp4"
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
    tmp = f"{BUILD}/assets/_video_luhn{key}.mp4"
    nf_hook, n_body, sheet = build_video(key, cfg, tmp)

    # расхождение дублей — измеряем и называем, а не чиним кропом
    print("средний RGB в карточке A: хук", card_rgb(conform_source(key, cfg), 1.0),
          "· основной дубль", card_rgb(SRC, 1.0))

    build_audio(key, cfg, nf_hook, apply_post_v(key, cfg), cfg["out"])

    cols = 6
    rows = int(np.ceil(len(sheet) / cols))
    cs = Image.new("RGB", (cols * 180, rows * 320), (10, 10, 10))
    for i, (t, arr) in enumerate(sheet):
        cs.paste(Image.fromarray(arr).resize((180, 320)), ((i % cols) * 180, (i // cols) * 320))
    os.makedirs(f"{BUILD}/test", exist_ok=True)
    csp = f"{BUILD}/test/hook17_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_luhn{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

"""Ролик 19 («QR-код и код Рида-Соломона»), альтернативные хуки.

Тело ролика (всё с кадра 227) не пересобирается — берётся готовыми кадрами из
assets/_video_qr.mp4 и склеивается за новым хуком. Заново рисуется только хук.

    python3 hook19.py h1
    python3 hook19.py h2

Точка реза: кадр 227 (7.5667с) — граница планов `A2` -> `qrbuild`, то есть
конец старой хуковой фразы «и вот почему это так работает» (слово «работает.»
кончается на 7.26, план — на 7.5667). Планы 1-5 старого начала и их субтитры
выброшены целиком, ни одного кадра от них не осталось.

Оба новых дубля кончаются на «и вот как это работает» — ровно та же передача
эстафеты, что в старом начале, поэтому фразу тела резать не пришлось: тело
начинается с «QR-код выглядит как случайный узор из чёрных квадратиков».

J-cut не нужен: предыдущая фраза реально смолкает на 7.138с, а первое слово тела
реально начинается на 7.590с (замерено по огибающей, не по whisper). Рез по
7.5667 попадает в паузу и ничего не срезает.

Формат хуков. Оба дубля совпадают с основным исходником по всем тегам:
720x1280, yuv420p, color_range=tv, bt709/bt709/bt709, 30 fps. Приведение формата
не требуется вообще — в отличие от ролика 17, где хуки приходили в полном
диапазоне. Поэтому кадры читаются прямо из .mov и идут в штатный тракт A-roll
без единого промежуточного кодека.

Крупность дублей (детектор, после hflip):
    основной  cx 377  линия глаз 650  размер лица 289
    hook1     cx 361  линия глаз 660  размер лица 284
    hook2     cx 381  линия глаз 660  размер лица 280
Расхождение до 16px по горизонтали, 10px по линии глаз и 3% по размеру лица.
Это НЕ правится кропом — числа названы в сдаче, решение за съёмочной стороной.

Графика хука. Гайд велит брать ту же графику, что стояла в старом хуке
(`qrlogo` — QR с логотипом, `qrok` — луч и галка). Здесь взяты `qrworn` и
`qrok2` — потрёпанный символ и луч с галкой по нему. Причина: оба новых дубля
говорят «порван или помят», а старая графика показывает символ, закрытый
логотипом. Логотип на словах «порван или помят» — ровно тот дефект, который
гайд запрещает («графика должна быть похожа на то, что произносится»).
Ни одного нового плана при этом не нарисовано: `qrworn` и `qrok2` уже есть
в ролике, они же стоят на вердикте в конце — начало и финал становятся рамкой.

Сток в хуках. Оба дубля говорят про супермаркет и кассу, поэтому под эти слова
стоят буквальные вставки: `49137` (женщина с тележкой выбирает продукты в
торговом зале) и `15914` (оплата картой на терминале у кассы).

Их не было в первой сборке версий по моей ошибке: парсер выдачи Mixkit брал id
из списка mp4-ссылок соседних блоков страницы, а не из слага самого клипа, и
сопоставлял их с чужими подписями. По такому «поиску» выходило, что супермаркета
на Mixkit нет. Реальные id лежат в слаге (`...-supermarket-49137/#video`) — по
запросу `supermarket` находится пять клипов, по `cashier` ещё пять.
Проверено заново и для основного ролика: `coffee` -> 236 и `puzzle` -> 42185
совпали с тем, что уже стояло, `qr-code`/`scanner`/`barcode` действительно пусты.

Весь сток приведён к 30 fps (`stock/raw_<id>.mp4` -> `stock/stock_<id>.mp4`):
клипы приходят 23.976/24/25 fps, а рендер читает их кадр в кадр, поэтому без
конформа они играли на 25% быстрее (delivery-specs §2).
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
import render19 as R
import storyboard19 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/19"
BODY = f"{BUILD}/assets/_video_qr.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_F = 227                         # первый кадр тела = граница плана A2 -> qrbuild
CUT = CUT_F / FPS                   # 7.5667с
TARGET_LUFS = -13.8                 # как в уже сданном qr_edit.mp4 (замер)

# (длительность полной анимации, запас до реза). Масштаб времени применяется
# только чтобы УСКОРИТЬ анимацию под короткий план хука; замедлять её ниже
# темпа тела нельзя — иначе тот же план в хуке и в теле живёт по-разному.
GFX_FULL = {"qrbuild": (0.70, 0.00), "qrworn": (0.80, 0.00), "qrok2": (1.45, 0.25)}

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # дубль звучит с первого кадра, тишины в голове нет — src_ss=0.
        # A-roll 341 кадр (11.3667с), речь в аудио идёт до 11.402с.
        # 346 кадров: хвост 0.131с + 0.023с в голове тела = пауза 0.155с
        # (норма 0.10-0.20). Последние 5 кадров добиваются графическим планом
        # qrok2, лицо не морозится.
        src_ss=0.0,
        frames=346,
        out=f"{VIDEO_DIR}/qr_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«каждый человек 100% хоть раз замечал, что когда вы покупаете "
              "продукты в супермаркете и идёте на кассу самообслуживания, "
              "даже если QR-код порван или помят, вы всё равно его легко "
              "отсканируете. и вот как это работает»",
        face_cx=361, face_eye=660, face_d=284,
        shots=[
            (0 / 30,   50 / 30,  "A1", {}),
            (50 / 30,  106 / 30, "stock", {"clip": "49137", "ss": 0.6}),
            (106 / 30, 140 / 30, "A2", {}),
            (140 / 30, 204 / 30, "stock", {"clip": "15914", "ss": 2.2}),
            (204 / 30, 256 / 30, "qrworn", {}),
            (256 / 30, 304 / 30, "A1", {}),
            (304 / 30, 346 / 30, "qrok2", {}),
        ],
        caps=[
            (0.000, 0.720, [("каждый человек", S, 50)], "A"),
            (0.720, 50 / 30, [("100% хоть раз", S, 52)], "A"),

            (50 / 30, 2.340, [("замечал что", S, 48)], "B"),
            (2.340, 2.820, [("например", "s", 60)], "B"),
            (2.820, 106 / 30, [("когда вы покупаете", S, 44)], "B"),

            (106 / 30, 4.000, [("продукты", S, 52)], "A"),
            (4.000, 140 / 30, [("в супермаркете", S, 48)], "A"),

            (140 / 30, 5.240, [("и идёте", S, 50)], "B"),
            (5.240, 204 / 30, [("на кассу ", S, 42), ("самообслуживания", "s", 46)], "B"),

            (6.820, 7.140, [("даже если", S, 50)], "G"),
            (7.140, 7.900, [("qr-код ", S, 46), ("порван", "s", 60)], "G"),
            (7.900, 256 / 30, [("или помят", S, 52)], "G"),

            (256 / 30, 9.120, [("вы всё равно", S, 48)], "A"),
            (9.120, 304 / 30, [("легко ", S, 44), ("отсканируете", "s", 54)], "A"),

            (10.420, 10.840, [("и вот как", S, 50)], "G"),
            (10.840, 346 / 30, [("это ", S, 46), ("работает", "s", 62)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # тишины в голове 0.02с, срезать нечего. A-roll 282 кадра (9.400с),
        # речь до 9.335с. 284 кадра: хвост 0.132с + 0.023с тела = пауза 0.155с.
        # Последние 2 кадра добиваются планом qrok2.
        src_ss=0.0,
        frames=284,
        out=f"{VIDEO_DIR}/qr_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«вы точно это замечали при покупке продуктов в супермаркете. "
              "если QR-код у вас немного замят или порван, вы всё равно его "
              "пробьёте на кассе. и вот как это работает»",
        face_cx=381, face_eye=660, face_d=280,
        shots=[
            (0 / 30,   56 / 30,  "A1", {}),
            (56 / 30,  112 / 30, "stock", {"clip": "49137", "ss": 4.4}),
            (112 / 30, 172 / 30, "A2", {}),
            (172 / 30, 216 / 30, "qrworn", {}),
            (216 / 30, 253 / 30, "stock", {"clip": "15914", "ss": 5.6}),
            (253 / 30, 284 / 30, "qrok2", {}),
        ],
        caps=[
            (0.000, 0.740, [("вы точно это", S, 50)], "A"),
            (0.740, 56 / 30, [("замечали при ", S, 44), ("покупке", "s", 58)], "A"),

            (56 / 30, 2.420, [("например", "s", 58)], "B"),
            (2.420, 112 / 30, [("продуктов в супермаркете", S, 40)], "B"),

            (3.940, 4.580, [("если qr-код", S, 50)], "A"),
            (4.580, 5.220, [("у вас немного", S, 46)], "A"),
            (5.220, 172 / 30, [("замят", "s", 66)], "A"),

            (172 / 30, 6.380, [("или ", S, 46), ("порван", "s", 64)], "G"),
            (6.480, 216 / 30, [("вы всё равно его", S, 46)], "G"),

            (216 / 30, 7.780, [("пробьёте", "s", 62)], "B"),
            (7.780, 253 / 30, [("на кассе", S, 52)], "B"),

            (253 / 30, 8.760, [("и вот как", S, 50)], "G"),
            (8.760, 284 / 30, [("это ", S, 46), ("работает", "s", 62)], "G"),
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
    """Кадры хука рисуем, кадры тела берём готовыми из _video_qr.mp4.

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

    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", tmp],
        stdin=subprocess.PIPE)

    stock_cap, stock_key = None, None
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

        def read_stock():
            nonlocal stock_cap, stock_key
            key = (t0, prm["clip"])
            if key != stock_key:
                if stock_cap is not None:
                    stock_cap.release()
                stock_cap = cv2.VideoCapture(f"{R.STOCK_DIR}/stock_{prm['clip']}.mp4")
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, prm.get("ss", 0) * 1000)
                stock_key = key
            ok_s, sfr = stock_cap.read()
            if not ok_s:
                stock_cap.set(cv2.CAP_PROP_POS_MSEC, 0)
                ok_s, sfr = stock_cap.read()
            return sfr

        canvas = R.background(kind, prm, fr, t, read_stock)
        # lts — масштаб времени анимации: графика тела рассчитана на длину плана
        # в теле, на более коротком плане хука она иначе оборвётся на полудвижении
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
    print(f"видео: хук {nf_hook} кадров (из них {held} добито графикой) "
          f"+ тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf_hook, n_body, sheet


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    """-ss после -i: точный отброс, чтобы звук встал кадр в кадр с картинкой."""
    tmp = f"{BUILD}/assets/_tmp_hook19.wav"
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
    mix_path = f"{BUILD}/assets/_mix_qr{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_qr{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_qr{key}_post.mp4"
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
    tmp = f"{BUILD}/assets/_video_qr{key}.mp4"
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
    csp = f"{BUILD}/test/hook19_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_qr{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

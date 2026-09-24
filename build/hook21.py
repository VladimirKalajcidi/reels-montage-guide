"""Ролик 21 («порог скидки»), альтернативные хуки.

Тело ролика (всё с кадра 152) не пересобирается — берётся готовыми кадрами из
assets/_video_porog.mp4 и склеивается за новым хуком. Заново рисуется только хук.

    python3 hook21.py h1
    python3 hook21.py h2

Точка реза: кадр 152 (5.0667с) — граница планов `A2` -> `calc`, то есть конец
фразы «это не просто какое-то случайное число» (слово «число» кончается на 5.060).
Планы 1-3 старого начала («когда вам говорят скидка при покупке» / «3000 ₽» /
«это не просто какое-то случайное число») и их субтитры выброшены целиком.

Кадр реза считается через ceil, а не round: 5.06с = кадр 151.8, и round(151.8)=152
здесь совпадает, но на дробной границе round отдал бы последний кадр
ВЫБРАСЫВАЕМОГО плана. Машинная проверка — qa21hook.old_hook_gone().

Почему рез именно здесь, а не на 3.16. Оба новых дубля целиком пересказывают
завязку («в магазине фраза "скидка от N рублей"») и заканчиваются вопросом
«откуда берётся это число / эта сумма». Тело с 5.0667 начинается словами
«это конкретный расчёт пользы магазина» — это прямой ответ на заданный вопрос.
Если резать по 3.16, тело начнёт с «это не просто какое-то случайное число»,
и ролик дважды подряд скажет, что число не случайное.

Связность местоимений (проверено чтением стыка вслух):
    h1: «...откуда берётся это ЧИСЛО, после которого идёт скидка.»
        -> «ЭТО конкретный расчёт пользы магазина.»          «это» -> «число» ✓
    h2: «...откуда ЭТА СУММА конкретно берётся?»
        -> «ЭТО конкретный расчёт пользы магазина.»          «это» -> «сумма» ✓
Ни одно местоимение тела не опирается на слово, которого в новом дубле нет.

Формат хуков. Оба дубля совпадают с основным исходником по всем тегам:
720x1280, yuv420p, color_range=tv, bt709/bt709/bt709, 30 fps. Приведение
цветового пространства не требуется — кадры читаются прямо из .mov и идут
в штатный тракт A-roll без промежуточного кодека.

Крупность дублей (детектор, после hflip, среднее по всему дублю):
    основной  cx 389.4  линия глаз 669.6  размер лица 267.6  (121 замер)
    hook1     cx 376.6  линия глаз 693.7  размер лица 273.9  (23 замера)
    hook2     cx 386.7  линия глаз 681.4  размер лица 273.6  (22 замера)
У hook1 линия глаз ниже основного дубля на 24px (в карточке A это ~31px из 1380),
у hook2 — на 12px. Размер лица расходится на 2.3%. Кропом это НЕ правится:
хук идёт через те же FRAMINGS, что тело, числа названы в сдаче.

Графика хука. В старом начале ролика 21 стоял план `thresh` — шкала чека
с отметкой порога и белым «3000 ₽». Он сохранён в обоих новых хуках: это
визуальная интрига ролика, и без него порог 3000 в теле появился бы ниоткуда
(в теле после реза число «3 тысячи» больше не звучит). Оба дубля при этом
говорят про сумму намеренно расплывчато — «скольки-то рублей», «определённой
суммы», — и графика ставит на это место конкретное число. Ни одного нового
плана не нарисовано.

Сток хука. Оба дубля начинаются со слова «магазин», поэтому вместо тележки
(49137, она стоит в теле на 17.84) взят новый клип 6102 — полки продуктового
с ценниками, проезд камеры. Скачан в videos/21/stock под этот ролик.
Полоса субтитров после STOCK_GAIN: h1 (ss 0.0) средняя 60.6 / p90 129,
h2 (ss 1.0) средняя 77.0 / p90 133 — обе с запасом внутри порога 110 / 140.
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
import render21 as R
import storyboard21 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/21"
BODY = f"{BUILD}/assets/_video_porog.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

import math

CUT_F = math.ceil(5.06 * FPS - 1e-6)   # 152 — первый кадр тела, план `calc`
CUT = CUT_F / FPS                      # 5.0667с
TARGET_LUFS = -14.0                    # как в уже сданном porog_edit.mp4 (замер)

# (длительность полной анимации, запас до реза). Масштаб времени применяется
# только чтобы УСКОРИТЬ анимацию под короткий план хука; замедлять её ниже
# темпа тела нельзя — иначе тот же план в хуке и в теле живёт по-разному.
# У `thresh` полная анимация 0.62с, а планы хука 1.67 / 1.40с — множитель 1.0,
# то есть графика идёт ровно тем же темпом, что в теле.
GFX_FULL = {"thresh": (0.62, 0.08)}

S = "r"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # дубль звучит с 0.076, речь кончается на 7.568 по огибающей.
        # A-roll декодируется ровно 228 кадров (7.600с).
        # 228 кадров: хвост 0.032с + 0.133с в голове тела = пауза 0.165с.
        # Лицо не морозится ни на кадр — A-roll покрывает весь блок.
        src_ss=0.0,
        frames=228,
        out=f"{VIDEO_DIR}/porog_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«я думаю все шопоголики рады видеть в магазине фразу "
              "„скидка от скольки-то рублей“, но вопрос — откуда берётся "
              "это число, после которого идёт скидка»",
        face_cx=376.6, face_eye=693.7, face_d=273.9,
        # лицо -> магазин -> шкала с порогом -> лицо. Структура старого начала
        # (лицо -> thresh -> лицо) сохранена, добавлен сток под слово «магазин».
        shots=[
            (0 / 30,   58 / 30,  "A1", {}),
            (58 / 30,  101 / 30, "stock", {"clip": "6102", "ss": 0.0}),
            (101 / 30, 151 / 30, "thresh", {}),
            (151 / 30, 228 / 30, "A2", {}),
        ],
        caps=[
            (0.000, 0.660, [("я думаю все", S, 48)], "A"),
            (0.660, 1.280, [("шопоголики", "s", 56)], "A"),
            (1.280, 58 / 30, [("рады видеть", S, 50)], "A"),

            (58 / 30, 2.420, [("в магазине", S, 48)], "B"),
            (2.420, 101 / 30, [("фразу ", S, 44), ("скидка", "s", 56)], "B"),

            # 3.37-4.68 «от скольки-то рублей» — субтитра нет: сумму на этом плане
            # пишет только графика, и пишет её конкретным числом «3000 ₽»
            (4.680, 151 / 30, [("но ", S, 44), ("вопрос", "s", 60)], "G"),

            (151 / 30, 5.820, [("откуда берётся", S, 48)], "A"),
            (5.820, 6.440, [("это ", S, 44), ("число", "s", 58)], "A"),
            (6.440, 6.900, [("после которого", S, 46)], "A"),
            (6.900, 228 / 30, [("идёт ", S, 44), ("скидка", "s", 56)], "A"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # дубль звучит с 0.061 и до самого конца файла: аудио 7.0155с при
        # 210 кадрах видео (7.000с). Обрезаются последние 15мс затухающего
        # свистящего «-ся» на уровне 5-8% от пика, поверх них ложится штатный
        # фейд 8мс — щелчка нет (проверка qa21hook.seam_click).
        # Пауза до первого слова тела = 0.133с (голова тела), в норме 0.10-0.20.
        src_ss=0.0,
        frames=210,
        out=f"{VIDEO_DIR}/porog_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«все же видели в магазинах фразу „при покупке от определённой "
              "суммы у вас идёт скидка“. так вот вопрос — откуда эта сумма "
              "конкретно берётся?»",
        face_cx=386.7, face_eye=681.4, face_d=273.6,
        shots=[
            (0 / 30,   47 / 30,  "A1", {}),
            (47 / 30,  103 / 30, "stock", {"clip": "6102", "ss": 1.0}),
            (103 / 30, 145 / 30, "thresh", {}),
            (145 / 30, 210 / 30, "A2", {}),
        ],
        caps=[
            (0.000, 0.640, [("все же видели", S, 48)], "A"),
            (0.640, 1.260, [("в ", S, 44), ("магазинах", "s", 56)], "A"),
            (1.260, 47 / 30, [("фразу", S, 54)], "A"),

            (47 / 30, 2.280, [("при покупке", S, 48)], "B"),
            (2.280, 103 / 30, [("от определённой ", S, 42), ("суммы", "s", 54)], "B"),

            # число на этом плане пишет графика («3000 ₽»), субтитр несёт слова —
            # дублирования нет: в речи сумма не названа вообще
            (103 / 30, 4.040, [("у вас идёт", S, 48)], "G"),
            (4.040, 145 / 30, [("скидка", "s", 58)], "G"),

            (145 / 30, 5.380, [("так вот ", S, 44), ("вопрос", "s", 58)], "A"),
            (5.500, 5.920, [("откуда эта", S, 48)], "A"),
            (5.920, 6.600, [("сумма конкретно", S, 46)], "A"),
            (6.600, 210 / 30, [("берётся", "s", 58)], "A"),
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
    """Кадры хука рисуем, кадры тела берём готовыми из _video_porog.mp4.

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

    stock = R.StockReader()
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
    tmp = f"{BUILD}/assets/_tmp_hook21.wav"
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
    mix_path = f"{BUILD}/assets/_mix_porog{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_porog{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_porog{key}_post.mp4"
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
    tmp = f"{BUILD}/assets/_video_porog{key}.mp4"
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
    csp = f"{BUILD}/test/hook21_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_porog{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

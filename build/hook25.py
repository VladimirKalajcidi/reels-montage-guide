"""Ролик 25 («теорема Морли»), альтернативные хуки.

Тело ролика (всё с кадра 166) не пересобирается — берётся готовыми кадрами из
assets/_video_morley.mp4 и склеивается за новым хуком. Заново рисуется только хук.

    python3 hook25.py h1
    python3 hook25.py h2

Точка реза: кадр 166 (5.533с) — граница планов `teaser` -> `A2`, то есть конец
старого начала «возьмите абсолютно любой треугольник, даже самый кривой. один
приём всегда превращает его в идеальный». Планы 1-3 старого хука и их пять
субтитров выброшены целиком.

`cut_f = ceil(5.52 * 30)` = 166, а не `round(165.6)` = 166 по счастливой
случайности: граница плана стоит на дробном кадре 165.6, и кадр 165 — это ещё
последний кадр ВЫБРАСЫВАЕМОГО плана `teaser`. Округление вниз оставило бы
в версии кадр старого хука. Проверка машинная, в qa25hook.old_hook_gone.

Передача эстафеты. Тело начинается новым предложением с собственным подлежащим:
«итак разделите каждый из трёх углов треугольника». Ни одного местоимения
и ни одной отсылки к словам старого хука в первых фразах тела нет — поэтому
обе версии режутся по одной точке.

Резать раньше, по границе 3.19 (`anyTri` -> `teaser`), нельзя: тогда тело
начиналось бы с «один приём всегда превращает ЕГО в идеальный», и «его»
осталось бы без антецедента — «треугольник» звучал только в выброшенном хуке.

hook1 кончается на «получить идеально равносторонний» — то же обещание,
что несло старое «превращает его в идеальный». Первый присланный дубль hook1
был обрезан на последнем слове и переснят; версия 2 пересобрана по новому файлу,
версия 3 (hook2) при этом не трогалась.
hook2 кончается на вопросе «...равносторонний?» — после него «итак разделите»
читается как ответ.

Формат хуков. Оба дубля совпадают с основным исходником по всем тегам:
720x1280, yuv420p, color_range=tv, bt709/bt709/bt709, 30 fps. Приведение
цветового пространства не требуется — кадры читаются прямо из .mov и идут
в штатный тракт A-roll без промежуточного кодека.

Крупность дублей (детектор, после hflip, медиана по всему дублю):
    основной  cx 392  линия глаз 642  ширина лица 286
    hook1     cx 382  линия глаз 652  ширина лица 293  (+7px, +2.4%)
    hook2     cx 402  линия глаз 648  ширина лица 291  (+5px, +1.7%)
Кропом это НЕ правится: оба дубля идут через те же FRAMINGS, что и основной
исходник, числа названы в сдаче.

Графика хука. В старом начале стояли планы `anyTri` (кривой треугольник тремя
штрихами) и `teaser` (весь приём и синий равносторонний результат) — это
визуальная интрига ролика, она сохранена в обоих новых хуках. Ни одного
нового плана не нарисовано.
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
import render25 as R
import storyboard25 as SB
from sfx import low_whoosh, impact, tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = "/Users/vladimirkalajcidi/reels_good/videos/25"
BODY = f"{BUILD}/assets/_video_morley.mp4"
AUDIO_DIR = "/Users/vladimirkalajcidi/reels_good/audios"
SRC = SB.SRC
SR = 48000

CUT_T = 5.52                                   # граница планов teaser -> A2
CUT_F = math.ceil(CUT_T * FPS - 1e-6)          # 166 — граница плана в кадрах
CUT = CUT_F / FPS                              # 5.533с


def cut_f_of(cfg):
    """Первый кадр тела в этой версии.

    Базовая точка — граница плана (кадр 166): всё, что раньше, принадлежит
    старому хуку и выбрасывается целиком. `head_trim_f` дополнительно срезает
    кадры с ГОЛОВЫ первого оставленного плана, чтобы укоротить паузу между
    хуком и первым словом тела. Речь при этом не режется: между границей плана
    (5.52с) и словом «итак» (6.18с) в дубле сплошная тишина, и срез идёт
    только по ней. Проверяется в qa25hook.head_trim.
    """
    return CUT_F + cfg.get("head_trim_f", 0)


def cut_of(cfg):
    return cut_f_of(cfg) / FPS
TARGET_LUFS = -14.0                            # замер сданного morley_edit.mp4

# (длительность полной анимации, запас до реза). Масштаб времени применяется
# только чтобы УСКОРИТЬ анимацию под короткий план хука; замедлять её ниже
# темпа тела нельзя — иначе тот же план в хуке и в теле живёт по-разному.
GFX_FULL = {"anyTri": (1.22, 0.10), "teaser": (1.64, 0.10)}

R_ = "r"
S_ = "s"

HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        # Первый присланный файл был обрезан на 5.877с прямо на последнем слове;
        # этот дубль полный: 188 кадров (6.258с), речь 0.26-6.08 по огибающей,
        # дальше 0.18с тишины. Берём 183 кадра (6.100с) — ровно до конца слова
        # плюс 0.02с, хвост тишины в версию не тащим.
        src_ss=0.0,
        frames=183,
        # 5 кадров (0.167с) срезаны с головы первого плана тела: пауза между
        # «равносторонний» и «итак» была 0.662с — собственный межфразовый разрыв
        # дубля, на стыке версий он читался длинновато. Стало 0.498с.
        # Срез идёт по тишине, ни одного слова не задето.
        head_trim_f=5,
        out=f"{VIDEO_DIR}/morley_hook1.mp4",
        music=f"{AUDIO_DIR}/song3.mp3",
        # версия 2: кроп 0.98 — обрезаем 2% по краям и возвращаем холст 1080x1920.
        # bilinear, а не lanczos: у lanczos отрицательные лепестки дают на границе
        # карточки светлый ореол в 1px, и автопроверка «текст за карточкой» краснеет.
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«этот красивый приём в геометрии помогает из любого, даже самого "
              "кривого треугольника получить идеально равносторонний»",
        face_cx=382, face_eye=652, face_d=293,
        # На полном дубле whisper слышит последнее слово как «равностороннее» —
        # так же, как в hook2. В кадре пишем «равносторонний»: слово согласуется
        # с «треугольника» в мужском роде, средний род тут не к чему привязать.
        asr_endings={"равносторонний": "равностороннее"},
        # структура старого начала сохранена: лицо -> графика -> лицо -> графика
        shots=[
            (0.00, 1.74, "A1", {}),
            (1.74, 2.92, "anyTri", {}),
            (2.92, 4.37, "A2", {}),
            (4.37, 183 / 30, "teaser", {}),
        ],
        caps=[
            (0.26, 1.12, [("этот красивый приём", R_, 46)], "A"),
            (1.12, 1.74, [("в геометрии", R_, 52)], "A"),

            (1.74, 2.92, [("помогает из любого", R_, 46)], "G"),

            (2.92, 3.27, [("даже самого", R_, 50)], "A"),
            (3.27, 4.37, [("кривого треугольника", R_, 46)], "A"),

            (4.37, 183 / 30, [("получить идеально ", R_, 42),
                              ("равносторонний", S_, 48)], "G"),
        ],
    ),
    "h2": dict(
        version=3,
        src=f"{VIDEO_DIR}/hook2.mov",
        # дубль 123 кадра (4.092с), речь 0.12-4.11 — тоже до конца файла.
        # 124 кадра: хвост слова цел, последний кадр добит графикой.
        src_ss=0.0,
        frames=124,
        out=f"{VIDEO_DIR}/morley_hook2.mp4",
        music=f"{AUDIO_DIR}/song1.mp3",
        # версия 3: ускорение 1.02 — картинка и звук вместе, синхрон не едет
        post_v="setpts=PTS/1.02,fps=30",   # fps=30 обязателен: иначе на выходе 30.6
        post_a="atempo=1.02",
        title="«а вы знаете, как получить из любого, даже самого кривого "
              "треугольника, равносторонний?»",
        face_cx=402, face_eye=648, face_d=291,
        # whisper слышит последнее слово как «равностороннее» (средний род),
        # то же самое и на полном дубле hook1. В кадре пишем «равносторонний»:
        # слово согласуется с «треугольника» в мужском роде, среднему роду тут
        # не к чему привязаться. Расхождение объявлено здесь, а не спрятано:
        # qa25hook считает такие случаи отдельным полем ending_overrides.
        asr_endings={"равносторонний": "равностороннее"},
        shots=[
            (0.00, 1.33, "A1", {}),
            (1.33, 2.68, "anyTri", {}),
            (2.68, 124 / 30, "teaser", {}),
        ],
        caps=[
            (0.12, 0.50, [("а вы знаете", R_, 50)], "A"),
            (0.50, 1.33, [("как получить из", R_, 48)], "A"),

            (1.33, 1.93, [("любого даже", R_, 50)], "G"),
            (1.93, 2.68, [("самого кривого", R_, 48)], "G"),

            # два слова подряд из речи: «...кривого треугольника, равносторонний?».
            # Разбить нечем — «равносторонний» осталось бы одиночным словом,
            # а это нарушение brand-kit §5 (2-4 слова во фразе).
            # Окончание — см. asr_endings ниже.
            (2.68, 124 / 30, [("треугольника ", R_, 44),
                              ("равносторонний", S_, 48)], "G"),
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
    """Кадры хука рисуем, кадры тела берём готовыми из _video_morley.mp4.

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
    """-ss после -i: точный отброс, чтобы звук встал кадр в кадр с картинкой."""
    tmp = f"{BUILD}/assets/_tmp_hook25.wav"
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
    for t0 in [s[0] for s in cfg["shots"][1:]]:      # резы внутри хука
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
    mix_path = f"{BUILD}/assets/_mix_morley{key}.wav"
    w = wave.open(mix_path, "wb")
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())
    w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    """Деформация версии по уже собранному немому видео, кадры не перерисовываются."""
    src = f"{BUILD}/assets/_video_morley{key}.mp4"
    vf = cfg.get("post_v")
    if not vf:
        return src
    dst = f"{BUILD}/assets/_video_morley{key}_post.mp4"
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
    tmp = f"{BUILD}/assets/_video_morley{key}.mp4"
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
    csp = f"{BUILD}/test/hook25_{key}_sheet.jpg"
    cs.save(csp, quality=92)
    print("контакт-лист хука:", csp)


if __name__ == "__main__":
    k = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == "remux":
        mux(apply_post_v(k, HOOKS[k]), f"{BUILD}/assets/_mix_morley{k}.wav",
            HOOKS[k]["out"], extra_af=HOOKS[k].get("post_a"))
    elif len(sys.argv) > 2 and sys.argv[2] == "audio":
        build_audio(k, HOOKS[k], HOOKS[k]["frames"],
                    apply_post_v(k, HOOKS[k]), HOOKS[k]["out"])
    else:
        main(k)

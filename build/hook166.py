"""Ролик 66 («обезьяна и Гамлет», слот 166), альтернативные хуки — блок 7. Копия hook305.py (шаблон под текущий тракт).

Тело ролика (всё с кадра 304) не пересобирается — берётся готовыми кадрами из `assets/_video_ape166.mp4` (сданная v1)
и склеивается за новым хуком. Заново рисуется только блок хука движком render166 (подмена SHOTS / CAPS / WORDS / DUR
и таймингов сцены inf из ape166). render166.py и storyboard166.py этот скрипт не трогает.

    python3 hook166.py h1

hook2.mov в videos/66/ нет — версия h2 не заведена (добавить в HOOKS, когда дубль появится: song3, ×1.02).

--- что выбрасывается -----------------------------------------------------------------------------------------------
hook1: «А вы знали, что обезьяна, бесконечно стучащая по клавиатуре, напечатает Гамлета? Вопрос только когда.» —
пересказ старого начала «Это самый необычный пример того, как работает бесконечность. Есть знаменитая теорема: если
обезьяна будет вечно нажимать случайные клавиши, она рано или поздно напечатает Гамлета» (0.29–9.82). Старое начало —
6 планов: A1, inf, A2, макака, клавиатура, машинка; граница плана после конца фразы — 9.95 (пауза 9.82–10.17).

--- шов ------------------------------------------------------------------------------------------------------------
Тело начинается планом A1 «Это звучит красиво, но давайте посчитаем» — «это» отсылает к утверждению обезьяны, которое
хук произносит сам. Шов — настоящий рез (хук кончается машинкой, тело — лицом), whoosh на нём как на любой склейке.
Голос тела «Это» атакует в 10.23 (10.22 = −44 дБ, 10.23 = −22). Пауза с границы плана 9.95 была бы ≥0.28с —
тело берётся с кадра 304 (10.1333) внутри тишины первого плана (урок videos_test/2): там молчащее лицо, субтитр «ЭТО»
всплывает только на 10.19. Голос тела — с кадра 303 (CUT_A = 10.10), на кадр раньше картинки (J-cut в 1 кадр).

--- пауза хук → тело (норма 0.10–0.20с) ---------------------------------------------------------------------------------
    hook1  «когда» кончается 5.28 (5.28 = −31 дБ, 5.30 = −36), блок 160 кадров (5.3333): голос хука до 5.30,
           тело с 10.10, «Это» в 10.23 → пауза 0.02 + 0.13 = 0.15с. Картинка хука кончается после конца речи (урок videos_test/1).

--- графика и сток блока хука ----------------------------------------------------------------------------------------
Та же, что стояла в старом хуке: знак ∞ (сцена inf из ape166) — рисуется с реза на «бесконечно», дальше точка;
пишущая машинка 50826 (ss 22, cx 0.75 — то же годное окно, что в v1; в теле её нет, повтора клипа нет).
Последний план — машинка до конца блока: лицом «Вопрос только когда» вышло бы 0.93с (< 1с).

--- формат и кадр дублей -------------------------------------------------------------------------------------------
hook1 совпадает с основным исходником по тегам: 2160×3840 HEVC, yuv420p, smpte170m/bt709/bt709 — HDR нет, приведение
не нужно. Тракт A-roll — тот же, что у основного: hflip (в дубле «dji» на петличке задом наперёд, work/hook1_mic.jpg),
FRAMINGS, GRADE из render166. Крупность и положение лица — в лог и в монтажный лист; кропом не правится.
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
from style import W, H, FPS
import render166 as R
import storyboard166 as SB
import ape166 as G
from sfx166 import low_whoosh, impact, soft_tick, read_wav

BUILD = os.path.dirname(os.path.abspath(__file__))
ROOT = "/Users/vladimirkalajcidi/reels_challenge"
VIDEO_DIR = f"{ROOT}/videos/66"
BODY = R.TMP                                    # assets/_video_ape166.mp4 — немое тело v1
AUDIO_DIR = f"{ROOT}/audios"
SRC = R.SRC
SR = 48000

CUT_F = 304                                     # первый кадр тела — внутри тишины плана A1 (см. «шов»)
CUT_V = CUT_F / FPS                             # 10.1333с — картинка тела
CUT_A = (CUT_F - 1) / FPS                       # 10.10с — голос тела, J-cut в 1 кадр
TARGET_LUFS = -14.2                             # замер сданного monkey_edit.mp4 (v1)

# импакты/тики тела — из списков ape166, посчитанных при импорте по таймингам ролика (до подмены G.T)
BODY_IMPACTS = list(G.IMPACT_T)
BODY_TICKS = list(G.TICK_T)
BODY_SHOTS = list(SB.SHOTS)

RD, RED = "r", "red"
HOOKS = {
    "h1": dict(
        version=2,
        src=f"{VIDEO_DIR}/hook1.mov",
        frames=160,                     # 5.3333с — пауза до тела 0.15с
        speech_end=5.28,
        tail_fade=None,                 # голос хука обрезается на 5.30 — после конца «когда» (−36 дБ)
        # резы: 41 (41 / 30 — провал 1.34–1.38 «обезьяна|бесконечно»), 72 (72 / 30 — провал 2.38–2.42 «стучащая|по»)
        shots=[(0, "A1"), (41, "inf"), (72, "stock")],
        stock={"clip": "50826", "ss": 22.0, "cx": 0.75},
        gfx=dict(i_draw=41 / 30, i_dot=41 / 30 + 0.55),
        fix={},
        # границы фраз на резах — ровно кадр/30 (уроки videos_test/4–6)
        caps=[
            (0.00, 0.81, [("а вы знали", RD, 62)], "A"),
            (0.81, 41 / 30, [("что обезьяна", RD, 62)], "A"),
            # «бесконечно» несёт знак ∞ — субтитр без него, явный тайминг; t_out до начала «по» (2.39) — иначе
            # явная фраза считает его занятым (урок 29), блок всё равно живёт до реза 72/30
            (41 / 30, 2.38, [("стучащая", RD, 62)], "G", [1.87]),
            (72 / 30, 3.19, [("по клавиатуре", RD, 62)], "B"),
            (3.19, 4.39, [("напечатает / гамлета", RD, 62)], "B"),
            (4.39, 160 / 30, [("вопрос только", RD, 62), ("когда", RED, 62)], "B"),
        ],
        out=f"{VIDEO_DIR}/monkey_hook1.mp4",
        music=f"{AUDIO_DIR}/song2.mp3",
        # версия 2: кроп ×0.98 — обрезаем 2% по краям и возвращаем холст 1080×1920
        post_v="crop=iw*0.98:ih*0.98,scale=1080:1920:flags=bilinear",
        post_a=None,
        title="«А вы знали, что обезьяна, бесконечно стучащая по клавиатуре, напечатает Гамлета? Вопрос только когда.»",
    ),
}


def shots_of(cfg):
    fr = cfg["shots"] + [(cfg["frames"], None)]
    return [(fr[i][0] / FPS, fr[i + 1][0] / FPS, fr[i][1], dict(cfg["stock"]) if fr[i][1] == "stock" else {})
            for i in range(len(cfg["shots"]))]


def hook_words(key, cfg):
    W_ = json.load(open(f"{VIDEO_DIR}/hook{key[1]}_words.json"))
    for k, (txt, t) in cfg["fix"].items():
        assert W_[k]["word"] == txt, (k, W_[k]["word"], txt)
        W_[k]["start"] = t
    for w0, w1 in zip(W_, W_[1:]):
        w0["end"] = min(w0["end"], w1["start"])
    return W_


def setup(key):
    """Подменяет в render166 / ape166 раскадровку на хуковую. Возвращает планы блока хука."""
    cfg = HOOKS[key]
    shots = shots_of(cfg)
    R.SHOTS = shots
    R.CAPS = cfg["caps"]
    R.DUR = cfg["frames"] / FPS
    R.WORDS = [w for w in hook_words(key, cfg) if any(ch.isalnum() for ch in w["word"])]
    R.BLOCKS = R._build_blocks()
    R.WORD_T = R._word_times_all()
    R.LAYOUT = {i: R._layout(i) for i in range(len(R.CAPS))}
    R._WL.clear()
    G.T.update(cfg["gfx"])
    # уроки videos_test/1, 4–6: фраза на резе, записанная ниже границы плана, уходит в прошлый план и не показывается
    for i, cap in enumerate(R.CAPS):
        si = R._shot_idx(cap[0])
        assert shots[si][0] <= cap[0] and cap[1] <= shots[si][1] + 1e-9, (i, cap[:2], shots[si][:2])
        assert R.BLOCKS[i][1] - cap[0] >= 1 / FPS, (i, cap[:2], R.BLOCKS[i])
        assert R._active(R.WORD_T[i][-1] + 0.05) == i, (i, cap[:2])
    print("фраз:", len(R.CAPS), "· мин. жизнь фразы %.3fс" % min(R.BLOCKS[i][1] - c[0] for i, c in enumerate(R.CAPS)))
    return shots


def prep_hook_aroll(key, cfg):
    out = {}
    for name in ("A1", "A2"):
        p = f"{BUILD}/assets/aroll_166_{key}_{name}.mp4"
        if not os.path.exists(p):
            w, h, x, y = R.FRAMINGS[name]
            vf = f"hflip,crop={w}:{h}:{x}:{y},scale={R.CW}:{R.CH}:flags=lanczos,{R.GRADE}"   # тракт render166.prep_aroll
            part = p.replace(".mp4", ".part.mp4")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", cfg["src"], "-vf", vf, "-an",
                            "-c:v", "libx264", "-crf", "14", "-preset", "medium", "-pix_fmt", "yuv420p", part], check=True)
            os.replace(part, p)
        out[name] = p
    return out


def face_stats(path):
    """Лицо в дубле (720×1280, с отражением — как идёт в рендер, hflip): среднее по 5 кадрам."""
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    xs, ws, ys = [], [], []
    for f in np.linspace(5, nf - 15, 5).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.resize(cv2.flip(fr, 1), (720, 1280), interpolation=cv2.INTER_AREA)
        d = casc.detectMultiScale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), 1.15, 6, minSize=(80, 80))
        if len(d):
            x, y, w, h = max(d, key=lambda r: r[2] * r[3])
            xs.append(x + w / 2); ws.append(w); ys.append(y + h / 2)
    cap.release()
    return round(float(np.mean(xs))), round(float(np.mean(ys))), round(float(np.mean(ws)))


def card_rgb(path, t):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000); ok, fr = cap.read(); cap.release()
    if not ok:
        return None
    fr = cv2.resize(fr, (720, 1280), interpolation=cv2.INTER_AREA)
    return tuple(int(v) for v in fr[200:1180, 100:620].reshape(-1, 3).mean(axis=0)[::-1])


# ----------------------------------------------------------------- видео
def build_video(key, cfg, tmp):
    shots = setup(key)
    ar = prep_hook_aroll(key, cfg)
    caps = {k: cv2.VideoCapture(p) for k, p in ar.items()}
    stock_files = R.prep_stock()                   # R.SHOTS уже хуковые — готовит только план машинки
    stock_caps = {i: cv2.VideoCapture(p) for i, p in stock_files.items()}
    stock_pos = {i: -1 for i in stock_files}
    nf = cfg["frames"]
    part = tmp.replace(".mp4", ".part.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", part],
        stdin=subprocess.PIPE)
    face_n = 0
    for f in range(nf):
        t = f / FPS
        oks = {k: c.grab() for k, c in caps.items()}
        shot = shots[R._shot_idx(t)]
        frame = None
        if shot[2] in R.FACE_KINDS:
            if not oks[shot[2]]:
                raise RuntimeError(f"A-roll хука кончился на кадре {f} под планом с лицом {shot[2]}")
            ok, frame = caps[shot[2]].retrieve()
            face_n += 1
        elif shot[2] == "stock":
            si = R._shot_idx(t)
            v = stock_caps[si]
            want = int(round((t - shot[0]) * FPS))
            while stock_pos[si] < want:
                if not v.grab():
                    break
                stock_pos[si] += 1
            ok, frame = v.retrieve()
            assert ok, f"сток хука кончился на кадре {f}"
        ff.stdin.write(R.compose(t, shot, frame).convert("RGB").tobytes())
    for c in list(caps.values()) + list(stock_caps.values()):
        c.release()
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
    assert ff.wait() == 0, "ffmpeg упал"
    os.replace(part, tmp)
    print(f"видео: хук {nf} кадров (лицо {face_n}, графика/сток {nf - face_n}) + тело {n_body} кадров -> {os.path.basename(tmp)}")
    return nf, n_body


# ----------------------------------------------------------------- звук
def wav_of(path, start=0.0):
    tmp = f"{BUILD}/assets/_tmp_hook166_{os.getpid()}.wav"   # h1 и h2 собираются параллельно — свой файл
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + (["-ss", f"{start:.5f}"] if start else []) + \
          ["-i", path, "-vn", "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", tmp]
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
    print("хук: уровень голоса подогнан к телу ×%.3f" % g)
    tf = cfg.get("tail_fade")
    if tf:
        i0, nf_ = int(tf[0] * SR), int(tf[1] * SR)
        m = max(0, min(nf_, len(hook) - i0))
        hook[i0:i0 + m] *= np.linspace(1, 0, m)[:, None]
        hook[i0 + m:] = 0.0
        print("хук: конец дубля погашен фейдом с %.3fс за %.3fс" % tf)
    n_hook = int(round((nf_hook - 1) / FPS * SR))      # голос тела — на кадр раньше картинки тела
    n_bod = int(round((n_body + 1) / FPS * SR))
    hook = np.vstack([hook, np.zeros((max(0, n_hook - len(hook)), 2), np.float32)])[:n_hook].copy()
    body = np.vstack([body, np.zeros((max(0, n_bod - len(body)), 2), np.float32)])[:n_bod].copy()
    r = int(0.008 * SR)
    hook[-r:] *= np.linspace(1, 0, r)[:, None]
    body[:r] *= np.linspace(0, 1, r)[:, None]
    voice = np.vstack([hook, body]).astype(np.float32)
    n = voice.shape[0]
    peak = np.abs(voice).max()
    bed = np.zeros((n, 2), np.float32)

    def add(sig, t, gain):
        i = int(t * SR)
        m = min(len(sig), n - i)
        if i >= 0 and m > 0:
            bed[i:i + m] += (sig[:m] * gain)[:, None]

    dt = nf_hook / FPS - CUT_V                          # SFX тела — от картинки
    WH, IM, TK = low_whoosh(), impact(), soft_tick()
    for s in shots[1:]:
        add(WH, s[0], peak * 0.040)                     # резы блока хука
    add(WH, nf_hook / FPS, peak * 0.040)                # шов — настоящий рез
    for s in BODY_SHOTS:
        if s[0] > CUT_V + 0.01:
            add(WH, s[0] + dt, peak * 0.040)
    for t in BODY_IMPACTS:
        if t >= CUT_V:
            add(IM, t + dt, peak * 0.070)
    for t in BODY_TICKS:
        if t >= CUT_V:
            add(TK, t + dt, peak * 0.030)
    for t in (cfg["gfx"]["i_draw"], cfg["gfx"]["i_dot"]):  # тики графики хука: ∞ рисуется, старт точки
        add(TK, t, peak * 0.030)

    mw = f"{BUILD}/assets/_music166_{os.path.basename(cfg['music']).split('.')[0]}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", cfg["music"],
                    "-ac", "2", "-ar", str(SR), "-c:a", "pcm_s16le", mw], check=True)
    mus = read_wav(mw)
    if mus.shape[1] == 1:
        mus = np.repeat(mus, 2, axis=1)
    if len(mus) >= n:
        track = mus[:n].copy()
        print("музыка: %s, петли нет" % os.path.basename(cfg["music"]))
    else:
        xf = int(0.25 * SR)
        core, tail = mus[:len(mus) - xf], mus[len(mus) - xf:]
        ramp = np.linspace(0, 1, xf)[:, None]
        loop = core.copy()
        loop[:xf] = loop[:xf] * ramp + tail * (1 - ramp)
        track = np.tile(loop, (int(np.ceil(n / len(loop))) + 1, 1))[:n]
        print("музыка: %s, петля %.2fс" % (os.path.basename(cfg["music"]), len(loop) / SR))
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
    mix_path = f"{BUILD}/assets/_mix166_{key}.wav"
    w = wave.open(mix_path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes()); w.close()
    mux(vid, mix_path, out, extra_af=cfg.get("post_a"))


def apply_post_v(key, cfg):
    src = f"{BUILD}/assets/_video_ape166{key}.mp4"
    dst = f"{BUILD}/assets/_video_ape166{key}_post.mp4"
    part = dst.replace(".mp4", ".part.mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src, "-vf", cfg["post_v"], "-an",
                    "-c:v", "libx264", "-crf", "15", "-preset", "medium", "-pix_fmt", "yuv420p", part], check=True)
    os.replace(part, dst)
    print("правка кадра:", cfg["post_v"])
    return dst


def measure(path):
    s = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    tail = s[s.rindex("Summary:"):]
    i = float(tail.split("I:")[1].split("LUFS")[0])
    tp = float(tail.split("Peak:")[1].split("dBFS")[0])
    return i, tp


def mux(vid, mix_path, out, target=TARGET_LUFS, extra_af=None):
    """loudnorm + доводка по замеру готового файла (delivery-specs §5); пик ≤ −1 dBTP обязателен."""
    def run(post_db):
        af = (f"{extra_af}," if extra_af else "") + "highpass=f=65,loudnorm=I=-14:TP=-1.5:LRA=7"
        af += f",volume={post_db:+.2f}dB,alimiter=limit=0.75:level=disabled"
        part = out.replace(".mp4", ".part.mp4")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", vid, "-i", mix_path,
                        "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-af", af, "-ar", str(SR),
                        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", part], check=True)
        os.replace(part, out)
        i, tp = measure(out)
        print("  звук: post %+.2f dB -> %.1f LUFS, %.1f dBTP" % (post_db, i, tp))
        return i, tp

    post_db, best = 1.7, None
    for _ in range(4):
        i, tp = run(post_db)
        if tp <= -1.0 and (best is None or abs(i - target) < abs(best[1] - target)):
            best = (post_db, i)
        if abs(i - target) <= 0.1 and tp <= -1.0:
            best = None
            break
        post_db += target - i
    if best is not None:
        run(best[0])
    print("готово:", out)


def main(key):
    cfg = HOOKS[key]
    tmp = f"{BUILD}/assets/_video_ape166{key}.mp4"
    nf_hook, n_body = build_video(key, cfg, tmp)
    print("лицо (cx, cy, ширина; 720×1280): хук", face_stats(cfg["src"]), "· основной дубль", face_stats(SRC))
    print("средний RGB в карточке A (до грейда): хук", card_rgb(cfg["src"], 1.0), "· основной дубль", card_rgb(SRC, 1.0))
    build_audio(key, cfg, nf_hook, n_body, apply_post_v(key, cfg), cfg["out"])


if __name__ == "__main__":
    main(sys.argv[1])

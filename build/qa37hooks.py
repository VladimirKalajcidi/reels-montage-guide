"""QA версий с другим хуком (ролик 37).

Проверяется то, что специфично для версий: от старого хука не осталось кадров,
тело совпадает кадр в кадр с версией 1, пауза на шве, отсутствие щелчка,
деформация версии измерена по объекту в кадре, fps после ускорения, длительности
видео и звука сходятся, громкость совпадает с оригиналом.
"""
import json
import os
import re
import subprocess
import sys
import wave

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import storyboard37 as SB
import render37 as R
import hook37 as HK
from style import FPS, CARD_A, CARD_B

VD = "/Users/vladimirkalajcidi/reels_good/videos/37"
ORIG = SB.OUT
TOTAL_FRAMES = 1675
BODY_FRAMES = TOTAL_FRAMES - HK.CUT_F


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate,nb_frames,width,height",
                        "-of", "json", path], capture_output=True, text=True)
    v = json.loads(r.stdout)["streams"][0]
    r2 = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                         "stream=codec_type,duration", "-of", "json", path],
                        capture_output=True, text=True)
    d = {s["codec_type"]: float(s.get("duration", 0)) for s in json.loads(r2.stdout)["streams"]}
    return dict(fps=v["r_frame_rate"], frames=int(v["nb_frames"]),
                w=int(v["width"]), h=int(v["height"]),
                v_dur=d.get("video", 0), a_dur=d.get("audio", 0))


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path,
                        "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    tail = r.stderr[-1400:]
    g = lambda pat: float(re.search(pat, tail).group(1))
    return (g(r"I:\s*(-?\d+\.\d+) LUFS"), g(r"Peak:\s*(-?\d+\.\d+) dBFS"),
            g(r"LRA:\s*(-?\d+\.\d+) LU"))


def _normalize(x, cfg):
    """Привести кадр версии 1 к геометрии версии: для кропа x0.98 применяем
    к оригиналу ту же обрезку и тот же ресайз, иначе сравнивались бы разные
    кадрирования, а не содержимое."""
    if not cfg.get("post_v", "").startswith("crop"):
        return x
    h, w = x.shape[:2]
    cw, ch = int(w * 0.98), int(h * 0.98)
    ox, oy = (w - cw) // 2, (h - ch) // 2
    return cv2.resize(x[oy:oy + ch, ox:ox + cw], (w, h), interpolation=cv2.INTER_LINEAR)


def body_identical(path, key, step=17):
    """Тело версии должно совпадать с версией 1 кадр в кадр.

    Для версии с деформацией кадра сравнение идёт после обратного приведения
    к общей геометрии, поэтому допуск — уровень шума кодека, а не 0.
    """
    cfg = HK.HOOKS[key]
    a = cv2.VideoCapture(ORIG)
    b = cv2.VideoCapture(path)
    off = cfg["frames"]
    rate = 1.02 if cfg.get("post_a") else 1.0
    worst, checked = 0.0, 0
    for f in range(HK.CUT_F, TOTAL_FRAMES, step):
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok1, x = a.read()
        j = int(round((off + (f - HK.CUT_F)) / rate))
        b.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok2, y = b.read()
        if not (ok1 and ok2):
            continue
        x = _normalize(x, cfg)
        if y.shape != x.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]))
        d = float(np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16))))
        worst = max(worst, d)
        checked += 1
    a.release()
    b.release()
    return checked, round(worst, 2)


def no_old_hook(path, key):
    """От старого хука не должно остаться ни кадра.

    `шов` — первый кадр тела версии против кадра CUT_F оригинала: обязаны
    совпасть. `начало` — кадр 10 версии против кадра 10 оригинала: обязаны
    РАЗОЙТИСЬ, там другой дубль. `хвост хука` — последний кадр блока хука
    против кадра CUT_F-1 оригинала (сетка с числом 2000): обязаны разойтись,
    иначе в версию затесался кадр старого начала.
    """
    cfg = HK.HOOKS[key]
    rate = 1.02 if cfg.get("post_a") else 1.0
    a = cv2.VideoCapture(ORIG)
    b = cv2.VideoCapture(path)
    out = {}
    pairs = (("шов (должны совпасть)", HK.CUT_F, int(round(cfg["frames"] / rate))),
             ("шов против кадра старого хука (должны разойтись)", HK.CUT_F - 1,
              int(round(cfg["frames"] / rate))),
             ("начало (должны разойтись)", 10, int(round(10 / rate))),
             ("хвост хука (должны разойтись)", HK.CUT_F - 1,
              int(round(cfg["frames"] / rate)) - 1))
    for name, fo, fv in pairs:
        a.set(cv2.CAP_PROP_POS_FRAMES, fo)
        ok1, x = a.read()
        b.set(cv2.CAP_PROP_POS_FRAMES, fv)
        ok2, y = b.read()
        if not (ok1 and ok2):
            continue
        x = _normalize(x, cfg)
        if y.shape != x.shape:
            y = cv2.resize(y, (x.shape[1], x.shape[0]))
        out[name] = round(float(np.mean(np.abs(x.astype(np.int16) - y.astype(np.int16)))), 2)
    a.release()
    b.release()
    return out


def old_hook_gone(path, key, thr=None):
    """Ни один кадр СТАРОГО НАЧАЛА не должен попасть в версию.

    Сравнивать блок хука со всеми кадрами 0..CUT_F-1 подряд бессмысленно:
    старое начало держало ту же стоковую вставку, что и новый блок, а клип
    со статичным классом на уменьшенном кадре почти не меняется от секунды
    к секунде. Такая метрика меряет «насколько сток похож сам на себя»
    (в ролике 36 давала min_diff 1.9 при полностью корректной склейке),
    а не «остался ли кадр старого хука».

    Поэтому проверяем раздельно:
      * с планами ЛИЦА старого начала — попиксельно: там другой дубль,
        разница обязана быть заметной;
      * по стоку — не пиксели, а окна клипа: отрезок [ss, ss+длина] нового
        блока не должен пересекаться со старым окном.
    Порог не константа: на статичной говорящей голове «другой дубль» и «тот
    же кадр» расходятся всего на несколько единиц, и число 8.0 из ролика 34
    здесь уже ничего не значит. Считаем порог от материала: берём разницу,
    которую даёт заведомо ТОТ ЖЕ кадр, прошедший через тракт версии (тело
    против версии 1), и требуем вдвое больше.
    """
    cfg = HK.HOOKS[key]
    if thr is None:
        thr = round(max(2.0, 2.0 * body_identical(path, key, step=31)[1]), 2)
    rate = 1.02 if cfg.get("post_a") else 1.0
    small = lambda img: cv2.resize(img, (96, 171)).astype(np.int16)
    a = cv2.VideoCapture(ORIG)
    olds = []
    for f in range(HK.CUT_F):
        if SB.shot_at(f / FPS)[2] not in SB.FACE_KINDS:
            continue
        a.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = a.read()
        if ok:
            olds.append((f, small(img)))
    a.release()
    b = cv2.VideoCapture(path)
    worst = None
    for f in range(int(round(cfg["frames"] / rate))):
        b.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = b.read()
        if not ok:
            break
        y = small(img)
        for fo, o in olds:
            d = float(np.mean(np.abs(y - o)))
            if worst is None or d < worst[0]:
                worst = (d, f, fo)
    b.release()
    d, fv, fo = worst
    return dict(old_face_frames=len(olds), min_diff=round(d, 2),
                closest_pair=f"кадр версии {fv} <-> кадр оригинала {fo}",
                threshold=thr, ok=bool(d > thr))


def stock_windows_disjoint(key):
    """Окно стокового клипа версии не должно пересекаться со старым окном.

    Пиксельно это не проверить (клип статичный), а по времени — однозначно.
    Пересечение означало бы, что в двух версиях подряд стоит один и тот же
    кадр стока: площадка честно сочтёт их дублями.
    """
    cfg = HK.HOOKS[key]
    old = [(prm["ss"], prm["ss"] + (t1 - t0))
           for t0, t1, k, prm in SB.SHOTS if k == "stock"]
    new = [(prm["ss"], prm["ss"] + (t1 - t0))
           for t0, t1, k, prm in HK.shots_of(cfg) if k == "stock"]
    bad = [(n, o) for n in new for o in old
           if n[0] < o[1] - 1e-9 and o[0] < n[1] - 1e-9]
    return dict(old=[(round(x, 2), round(y, 2)) for x, y in old],
                new=[(round(x, 2), round(y, 2)) for x, y in new],
                overlaps=len(bad), ok=not bad)


def seam_click(path, key):
    """Скачок между соседними сэмплами на шве не должен выделяться на фоне
    типичного скачка внутри дорожки. Склейка голоса — на кадре N-1 (J-cut)."""
    cfg = HK.HOOKS[key]
    wav = f"{HK.BUILD}/assets/_qa37seam.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                    "-vn", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", wav], check=True)
    w = wave.open(wav, "rb")
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    w.close()
    os.remove(wav)
    rate = 1.02 if cfg.get("post_a") else 1.0
    i = int((cfg["frames"] - 1) / FPS / rate * 48000)      # склейка голоса
    local = float(np.abs(np.diff(x[i - 3:i + 4])).max())
    typical = float(np.percentile(np.abs(np.diff(x)), 99.9))
    return round(local, 5), round(typical, 5)


def pause_at_seam(key):
    """Пауза «конец речи хука -> первое слово тела» в новой дорожке.

    Голос тела кладётся на кадр N-1, первое слово тела («итак») звучит
    в теле на 3.723, то есть на (3.723 - CUT_A) позже начала вставленного
    куска. По огибающей атака чуть раньше, на 3.705 — пауза от этого только
    короче на 18мс и из диапазона не выходит.
    """
    cfg = HK.HOOKS[key]
    hw = json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))
    last = max(w["end"] for w in hw)
    body = json.load(open(SB.WORDS))
    first = min(w["start"] for w in body if w["start"] >= HK.CUT_A)
    attack = (cfg["frames"] - 1) / FPS + (first - HK.CUT_A)
    rate = 1.02 if cfg.get("post_a") else 1.0
    return round((attack - last) / rate, 3), round(last, 3), round(attack, 3)


def deform(path, key):
    """Деформацию мерим по объекту в кадре: ширина карточки A на плане с лицом
    в теле версии против той же ширины в версии 1.

    Край карточки берём как ЦЕНТР перехода чёрное->кадр (полувысота контраста),
    а не как первый пиксель выше порога: bilinear-ресайз версии 2 размывает
    границу, порог срезал бы её внутрь с обеих сторон и занизил масштаб.
    Полувысота от размытия не едет — оно симметрично относительно неё.
    """
    cfg = HK.HOOKS[key]
    t_orig = 47.00                     # план A1 45.90-49.20, светлая футболка
    y_rel = 1500 / 1920                # строка внутри карточки A
    rate = 1.02 if cfg.get("post_a") else 1.0

    def card_width(p, f):
        c = cv2.VideoCapture(p)
        vals = []
        for k in range(9):
            c.set(cv2.CAP_PROP_POS_FRAMES, f + k)
            ok, img = c.read()
            if not ok:
                continue
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            h, w = img.shape[:2]
            y = int(round(y_rel * h))
            strip = g[y - 2:y + 3, :].mean(axis=0)
            hi = float(np.percentile(strip, 90))
            half = hi / 2.0
            if hi < 40:
                continue

            def cross(idx_range, rising):
                """Позиция полувысоты контраста, субпиксельно. Знаменатель
                берётся как есть: на спадающем фронте b-a отрицателен, и
                max(1e-6, b-a) превращал бы дробь в мусор."""
                for i in idx_range:
                    a, b = float(strip[i]), float(strip[i + 1])
                    if (rising and a < half <= b) or (not rising and a >= half > b):
                        d = b - a
                        if abs(d) < 1e-6:
                            return float(i)
                        return i + (half - a) / d
                return None

            l = cross(range(0, w - 1), True)
            r = cross(range(w - 2, 0, -1), False)
            if l is not None and r is not None:
                vals.append(r - l)
        c.release()
        return round(float(np.mean(vals)), 2) if vals else None

    f_orig = int(round(t_orig * FPS))
    f_ver = int(round((cfg["frames"] + (f_orig - HK.CUT_F)) / rate))
    a = card_width(ORIG, f_orig)
    b = card_width(path, f_ver)
    out = dict(card_px_v1=a, card_px_ver=b,
               scale=(round(b / a, 4) if a and b else None))
    if cfg.get("post_a"):                       # версия с ускорением
        pr = probe(path)
        want = (cfg["frames"] + BODY_FRAMES) / FPS / rate
        out.update(kind="ускорение x1.02", fps=pr["fps"],
                   dur=round(pr["v_dur"], 3), dur_expected=round(want, 3),
                   av_delta=round(abs(pr["v_dur"] - pr["a_dur"]), 3))
    else:
        # ffmpeg режет кроп до целых пикселей: 1080*0.98 = 1058.4 -> 1058,
        # поэтому реальный коэффициент 1080/1058, а не 1/0.98
        out.update(kind="кроп x0.98",
                   scale_expected=round(1080 / int(1080 * 0.98), 4))
    return out


def caption_words_match(key):
    """Слова субтитров хука обязаны быть в расшифровке этого дубля."""
    cfg = HK.HOOKS[key]
    norm = lambda w: re.sub(r"[^а-яa-z0-9]", "", w.lower().replace("ё", "е"))
    said = [norm(w["word"]) for w in
            json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))]
    said = [w for w in said if w]
    typed = [norm(w) for _t0, _t1, runs, _s in cfg["caps"]
             for txt, _k, _sz in runs for w in txt.split()]
    typed = [w for w in typed if w]
    i, missing = 0, []
    for w in typed:
        j = i
        while j < len(said) and said[j] != w:
            j += 1
        if j < len(said):
            i = j + 1
        else:
            missing.append(w)
    return len(missing), missing


def caption_checks(key):
    cfg = HK.HOOKS[key]
    shots = HK.shots_of(cfg)
    hw = json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))
    bad = []
    for t0, t1, runs, _s in cfg["caps"]:
        text = "".join(r[0] for r in runs).strip()
        if any(ch in text for ch in ".,!?;:«»"):
            bad.append(("пунктуация", t0, text))
        if text != text.lower():
            bad.append(("капс", t0, text))
        n = len(text.split())
        if not 2 <= n <= 4:
            bad.append((f"{n} слов", t0, text))
        if re.search(r"\d", text):
            bad.append(("цифра в субтитре", t0, text))
        if not any(abs(w["start"] - t0) < 0.26 for w in hw):
            bad.append(("мимо речи", t0, text))
        # проверяем по кадрам: субтитр появляется на первом кадре >= t0
        # и живёт до последнего кадра < t1
        f0 = int(np.ceil(t0 * FPS - 1e-9))
        f1 = int(np.ceil(t1 * FPS - 1e-9)) - 1
        s0 = next((s for s in shots if s[0] <= f0 / FPS < s[1]), None)
        s1 = next((s for s in shots if s[0] <= f1 / FPS < s[1]), None)
        if s0 is not None and s1 is not None and s0 is not s1:
            bad.append(("субтитр через рез", t0, text))
    return len(bad), bad


def cuts_inside_word(key):
    cfg = HK.HOOKS[key]
    hw = json.load(open(f"{VD}/words_hook{cfg['version'] - 1}.json"))
    bad = []
    for t0, _t1, _k, _p in HK.shots_of(cfg)[1:]:
        for w in hw:
            if w["start"] + 0.02 < t0 < w["end"] - 0.02:
                bad.append((round(t0, 3), w["word"]))
    return len(bad), bad


def aroll_not_frozen(key):
    """Лицо не морозится: кадров A-roll в дубле должно хватать на все планы
    с лицом. Хвост блока добивается стоковым планом, а не стоп-кадром лица."""
    cfg = HK.HOOKS[key]
    shots = HK.shots_of(cfg)
    at = lambda t: next((x for x in shots if x[0] <= t < x[1]), shots[-1])
    face_max = max((f for f in range(cfg["frames"])
                    if at(f / FPS)[2] in SB.FACE_KINDS), default=-1)
    return dict(last_face_frame=face_max, src_frames=cfg["src_frames"],
                tail_kind=at((cfg["frames"] - 1) / FPS)[2],
                ok=face_max < cfg["src_frames"])


def rhythm(key, path):
    cfg = HK.HOOKS[key]
    hook_shots = [(a, b, k) for a, b, k, _ in HK.shots_of(cfg)]
    # шов совпадает с настоящим резом (карточка B -> карточка A), планы
    # блока хука и тела не склеиваются
    body_shots = [(max(a, HK.CUT_V), b, k) for a, b, k, _ in SB.SHOTS if b > HK.CUT_V]
    dt = cfg["frames"] / FPS - HK.CUT_V
    body_shots = [(a + dt, b + dt, k) for a, b, k in body_shots]
    merged = hook_shots + body_shots
    rate = 1.02 if cfg.get("post_a") else 1.0
    all_len = [(b - a) / rate for a, b, _ in merged]
    face = sum(b - a for a, b, k in merged if k in SB.FACE_KINDS) / rate
    dur = probe(path)["v_dur"]
    s = sorted(all_len)
    return dict(shots=len(s), avg=round(sum(s) / len(s), 2), median=round(s[len(s) // 2], 2),
                shortest=round(s[0], 2), longest=round(s[-1], 2),
                first_cut=round(hook_shots[0][1] / rate, 2),
                face_pct=round(100 * face / dur, 1), dur=round(dur, 2))


def pixel_checks(key):
    """Те же машинные проверки, что у версии 1, но по кадрам блока хука:
    текст не выходит за карточку своего плана, строки блока не налезают."""
    from style import font
    cfg = HK.HOOKS[key]
    shots = HK.shots_of(cfg)
    R.CAPS = cfg["caps"]
    R.shot_at = lambda t, s=shots: next((x for x in s if x[0] <= t < x[1]), s[-1])
    R.BLOCKS = R.build_blocks()
    out = dict(text_out_of_card=0, gfx_text_overlap=0, gfx_out_of_zone=0)
    samples = []
    for f in range(0, cfg["frames"], 1):
        t = f / FPS
        t0, _t1, kind, _p = R.shot_at(t)
        gl = R.graphics_layer(kind, t - t0)
        cl = R.caption_layer(t)
        if cl is not None:
            a = np.array(cl.split()[3])
            ys, xs = np.nonzero(a > 120)
            cx, cy, cw, chh = CARD_B if kind == "stock" else CARD_A
            if len(ys) and (xs.min() < cx + 30 or xs.max() > cx + cw - 30
                            or ys.min() < cy + 30 or ys.max() > cy + chh - 30):
                out["text_out_of_card"] += 1
                samples.append(("текст за карточкой", round(t, 2), kind))
        if gl is not None:
            ga = np.array(gl.split()[3])
            ys, xs = np.nonzero(ga > 40)
            y0, y1 = SB.GFX_ZONE
            x0, x1 = SB.GFX_X
            if len(ys) and (ys.min() < y0 or ys.max() > y1
                            or xs.min() < x0 or xs.max() > x1):
                out["gfx_out_of_zone"] += 1
            if cl is not None and int(np.count_nonzero((ga > 40) & (np.array(cl.split()[3]) > 40))):
                out["gfx_text_overlap"] += 1
    kmap = {"r": "sans", "i": "sans_it", "s": "serif_it"}
    boxes = {}
    for i, (_t0, _t1, runs, _s) in enumerate(cfg["caps"]):
        top = min(font(kmap[k], sz).getbbox(txt)[1] for txt, k, sz in runs)
        bot = max(font(kmap[k], sz).getbbox(txt)[3] for txt, k, sz in runs)
        boxes[i] = (top, bot)
    by_block = {}
    for i in range(len(cfg["caps"])):
        bid, pos, _n, _e, yoff = R.BLOCKS[i]
        by_block.setdefault(bid, []).append((pos, i, yoff))
    pairs = 0
    for _bid, rows in by_block.items():
        rows.sort()
        for a, b in zip(rows, rows[1:]):
            ha = (boxes[a[1]][1] - boxes[a[1]][0]) / 2
            hb = (boxes[b[1]][1] - boxes[b[1]][0]) / 2
            if a[2] + ha > b[2] - hb:
                pairs += 1
    out["line_overlap_pairs"] = pairs
    return out, samples[:6]


def face_rgb(key):
    """Средний RGB внутри карточки A: дубль хука против основного дубля."""
    cfg = HK.HOOKS[key]
    return dict(hook=HK.card_rgb(cfg["src"], 1.0), main=HK.card_rgb(SB.SRC, 1.0))


def v1_untouched():
    """Версия 1 остаётся эталоном: hook37.py в неё не пишет (пишет только в
    cards_hook*.mp4), файл должен быть старше обеих версий."""
    import hashlib
    t0 = os.path.getmtime(ORIG)
    later = {os.path.basename(c["out"]): round(os.path.getmtime(c["out"]) - t0, 1)
             for c in HK.HOOKS.values()}
    h = hashlib.sha256(open(ORIG, "rb").read()).hexdigest()[:16]
    return dict(sha256_16=h, size=os.path.getsize(ORIG),
                versions_newer_by_sec=later, ok=all(v > 0 for v in later.values()))


def tracks_unique():
    used = {"версия 1": os.path.basename("song1.mp3")}
    for k, c in HK.HOOKS.items():
        used[f"версия {c['version']} ({k})"] = os.path.basename(c["music"])
    return used, len(set(used.values())) == len(used)


if __name__ == "__main__":
    print("версия 1:", probe(ORIG), loudness(ORIG))
    print("треки:", tracks_unique())
    print("версия 1 не тронута:", v1_untouched())
    for key in ("h1", "h2"):
        cfg = HK.HOOKS[key]
        p = cfg["out"]
        print(f"\n===== {key} -> {os.path.basename(p)} (версия {cfg['version']}) =====")
        print("формат:", probe(p))
        i, tp, lra = loudness(p)
        print(f"громкость: I={i} LUFS  TP={tp} dBTP  LRA={lra} LU  "
              f"(версия 1: {loudness(ORIG)[0]} LUFS)")
        print("ритм:", rhythm(key, p))
        print("шов (средняя разница пикселей):", no_old_hook(p, key))
        print("ни одного кадра лица старого хука:", old_hook_gone(p, key))
        print("окна стока не пересекаются:", stock_windows_disjoint(key))
        ch, worst = body_identical(p, key)
        print(f"тело кадр в кадр: сверено {ch} кадров, худшая средняя разница {worst}")
        loc, typ = seam_click(p, key)
        print(f"щелчок на шве: локальный скачок {loc} против типичного {typ} (p99.9)")
        print("пауза хук->тело:", pause_at_seam(key), "(пауза, конец речи хука, атака тела)")
        print("деформация:", deform(p, key))
        print("A-roll не морожен:", aroll_not_frozen(key))
        print("средний RGB в карточке A:", face_rgb(key))
        print("субтитры мимо расшифровки:", caption_words_match(key))
        print("субтитры (стиль/тайминг):", caption_checks(key))
        print("резы внутри слова:", cuts_inside_word(key))
        print("пиксельные проверки блока хука:", pixel_checks(key))

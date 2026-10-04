"""Проверка стока ролика 56 (слот 157) на используемых отрезках: склейки внутри (попарная разница
кадров), средняя яркость кадра (≥~85, тёмные вставки автор просил не ставить) и ДОЛЯ ПОЧТИ БЕЛОГО
(>200) в полосе субтитров карточки B (y 440…560, x 154…708) — урок ролика 18: средняя по зоне
не ловит белую бумагу под текстом; брать ≈0–5%.
Режим scan (`python3 stockcheck157.py scan <id> <длина>`) — перебор ss с шагом 0.5с.
Подробности -> stockcheck157.log."""
import sys, cv2, numpy as np
D = "/Users/vladimirkalajcidi/reels_challenge/videos/56/stock"


def measure(clip, ss, dur, cx=0.5, cy=0.5, speed=1):
    c = cv2.VideoCapture(f"{D}/stock_{clip}.mp4")
    fps = c.get(cv2.CAP_PROP_FPS)
    c.set(cv2.CAP_PROP_POS_MSEC, ss * 1000)
    prev, diffs, white, mean = None, [], [], []
    for k in range(int((dur + 0.3) * fps)):
        # speed (урок ролика 46): в кадр идёт каждый speed-й кадр
        for _ in range(speed):
            ok, f = c.read()
        if not ok:
            break
        h, w = f.shape[:2]
        cw = min(w, int(h * 1.3839)); ch = min(h, int(w / 1.3839))
        x0 = int(round((w - cw) * cx))   # cx — сдвиг кропа по горизонтали (0 — левый край, 1 — правый)
        y0 = int(min(max(h * cy - ch / 2, 0), h - ch))
        f = f[y0:y0 + ch, x0:x0 + cw]
        f = cv2.resize(f, (858, 620))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(g - prev).mean())
        prev = g
        white.append((g[440:560, 154:708] > 200).mean() * 100); mean.append(g.mean())
    mx = max(diffs) if diffs else 0
    med = float(np.median(diffs)) if diffs else 0
    return dict(n=len(mean), mx=mx, med=med, cut=mx > max(25, 6 * med), static=med < 0.3,
                mean=float(np.mean(mean)) if mean else 0, white=float(max(white)) if white else 100,
                ok_len=len(mean) >= int(dur * fps) - 1)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "scan":
    clip, dur = sys.argv[2], float(sys.argv[3])
    cxs = [float(x) for x in sys.argv[4:]] or [0.5]
    tot = cv2.VideoCapture(f"{D}/stock_{clip}.mp4").get(cv2.CAP_PROP_FRAME_COUNT) / \
        cv2.VideoCapture(f"{D}/stock_{clip}.mp4").get(cv2.CAP_PROP_FPS)
    ss = 0.0
    while ss + dur + 0.3 <= tot:
      for cx in cxs:
        m = measure(clip, ss, dur, cx)
        print(f"{clip} ss={ss:4.1f} cx={cx:.2f} кадр {m['mean']:5.1f} белое {m['white']:4.1f}% diff {m['mx']:4.1f}/{m['med']:4.1f}"
              f"{' СКЛЕЙКА?' if m['cut'] else ''}")
      ss += 0.5 if tot < 30 else 2.0
elif __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fast":
    # урок ролика 34: быстрый параллельный отсев — сетка ss шаг 1с × cx 0/0.5/1, пул процессов
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for arg in sys.argv[2:]:
        clip, dur = arg.split(":"); dur = float(dur)
        c = cv2.VideoCapture(f"{D}/stock_{clip}.mp4")
        tot = c.get(cv2.CAP_PROP_FRAME_COUNT) / c.get(cv2.CAP_PROP_FPS)
        ss = 0.0
        while ss + dur + 0.3 <= tot:
            for cx in (0.0, 0.5, 1.0):
                jobs.append((clip, ss, dur, cx))
            ss += 1.0
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(measure, *zip(*jobs)))
    best = {}
    for (clip, ss, dur, cx), m in zip(jobs, res):
        ok = not m["cut"] and m["mean"] >= 85 and m["white"] <= 5
        if ok and (clip not in best or m["white"] < best[clip][1]["white"]):
            best[clip] = ((ss, cx), m)
    for arg in sys.argv[2:]:
        clip = arg.split(":")[0]
        allm = [m for (c_, *_), m in zip(jobs, res) if c_ == clip]
        if clip in best:
            (ss, cx), m = best[clip]
            print(f"{clip}: ok окон {sum(1 for (c_, *_), mm in zip(jobs, res) if c_ == clip and not mm['cut'] and mm['mean'] >= 85 and mm['white'] <= 5)}/{len(allm)}; лучшее ss={ss} cx={cx} яркость {m['mean']:.0f} белое {m['white']:.1f}%")
        else:
            print(f"{clip}: НЕТ годных окон; яркость {min(m['mean'] for m in allm):.0f}…{max(m['mean'] for m in allm):.0f}, белое ≥{min(m['white'] for m in allm):.1f}%")
elif __name__ == "__main__":
    from storyboard156 import SHOTS
    log = open(__file__.replace(".py", ".log"), "w")
    for t0, t1, kind, prm in SHOTS:
        if kind != "stock":
            continue
        m = measure(prm["clip"], prm["ss"], t1 - t0, prm.get("cx", 0.5), prm.get("cy", 0.5), prm.get("speed", 1))
        bad = m["static"] or m["cut"] or m["mean"] < 85 or m["white"] > 5 or not m["ok_len"]
        line = (f"{prm['clip']:>7} ss={prm['ss']:<4} кадров={m['n']} diff max={m['mx']:5.1f} med={m['med']:4.1f} "
                f"{'СКЛЕЙКА?' if m['cut'] else 'СТАТИКА' if m['static'] else 'ok'} | яркость кадра {m['mean']:5.1f}, почти белого в полосе текста "
                f"{m['white']:4.1f}% {'<-- ПРОВЕРИТЬ' if bad else ''}")
        log.write(line + "\n"); print(line)

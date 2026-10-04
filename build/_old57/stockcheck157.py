"""Проверка стока ролика 57 на используемых отрезках.

Проверки:
  - склейка внутри отрезка (max diff > max(25, 6×median))
  - яркость кадра ≥ 85 (тёмный клип?)
  - медиана diff ≥ 0.3 (СТАТИКА — freeze/слайдшоу)
  - белые пиксели в зоне субтитров < 5% (текст в кадре мешает субтитрам)
"""
import cv2, numpy as np
from storyboard157 import SHOTS

D = "/Users/vladimirkalajcidi/reels_challenge/videos/57/stock"
log = open(__file__.replace(".py", ".log"), "w")
ok_all = True

for t0, t1, kind, prm in SHOTS:
    if kind != "stock":
        continue
    c = cv2.VideoCapture(f"{D}/stock_{prm['clip']}.mp4")
    fps = c.get(cv2.CAP_PROP_FPS)
    c.set(cv2.CAP_PROP_POS_MSEC, prm["ss"] * 1000)
    prev, diffs, lum, mean, white_ratio = None, [], [], [], []
    for k in range(int((t1 - t0 + 0.3) * fps)):
        ok, f = c.read()
        if not ok:
            break
        h, w = f.shape[:2]
        cw = min(w, int(h * 1.3839)); ch = min(h, int(w / 1.3839))
        f = f[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
        f = cv2.resize(f, (858, 620))
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(g - prev).mean())
        prev = g
        # нижняя треть карточки B (зона субтитров: y 440-560 из 620)
        sub_band = g[440:560]
        lum.append(sub_band.mean())
        mean.append(g.mean())
        white_ratio.append((sub_band > 230).mean())
    c.release()

    mx = max(diffs) if diffs else 0
    med = float(np.median(diffs)) if diffs else 0
    mean_br = np.mean(mean) if mean else 0
    mean_lum = np.mean(lum) if lum else 0
    mean_wr = np.mean(white_ratio) if white_ratio else 0

    issues = []
    if mx > max(25, 6 * med):
        issues.append("СКЛЕЙКА?")
    if mean_br < 85:
        issues.append(f"ТЁМНЫЙ({mean_br:.0f})")
    if med < 0.3:
        issues.append(f"СТАТИКА(med={med:.2f})")
    if mean_wr > 0.05:
        issues.append(f"ТЕКСТ_В_КАДРЕ({mean_wr:.1%})")
    verdict = " ".join(issues) if issues else "ok"
    if issues:
        ok_all = False

    line = (f"{prm['clip']:>7} ss={prm['ss']:<4} кадров={len(lum)} "
            f"diff max={mx:5.1f} med={med:4.1f} | яркость {mean_br:5.1f} "
            f"sub {mean_lum:5.1f} white {mean_wr:.1%}  {verdict}")
    log.write(line + "\n")
    print(line)

summary = "ВСЕ OK" if ok_all else "ЕСТЬ ПРОБЛЕМЫ"
log.write(summary + "\n")
print(summary)

"""Параллельный замер окон стока ролика 57 (stockcheck157.measure): сетка ss × cx × cy по списку клипов -> scan157.log.
python3 scan157.py <id>:<длина> ... — печатает лучшее годное окно и окно с минимумом белого для каждого клипа."""
import sys, cv2
from concurrent.futures import ProcessPoolExecutor
import stockcheck157 as s


def run(j):
    return j, s.measure(*j)


if __name__ == "__main__":
    jobs = []
    for arg in sys.argv[1:]:
        c, d = arg.split(":"); d = float(d)
        v = cv2.VideoCapture(f"{s.D}/stock_{c}.mp4"); T = v.get(7) / v.get(5)
        ss = 0.0
        while ss + d + 0.3 <= T:
            jobs += [(c, ss, d, cx, cy) for cx in (0, 0.5, 1) for cy in (0.2, 0.5, 0.8)]
            ss += 1.5
    best, low, nok = {}, {}, {}
    with ProcessPoolExecutor() as ex, open(__file__.replace(".py", ".log"), "a") as log:
        for j, m in ex.map(run, jobs):
            ok = not m["cut"] and not m["static"] and m["mean"] >= 85 and m["white"] <= 5
            c = j[0]
            log.write(f"{j} ярк {m['mean']:.0f} белое {m['white']:.1f}% med {m['med']:.2f} {'ok' if ok else ''}\n")
            nok[c] = nok.get(c, 0) + ok
            if ok and (c not in best or m["white"] < best[c][1]["white"]):
                best[c] = (j, m)
            if c not in low or m["white"] < low[c][1]["white"]:
                low[c] = (j, m)
    for c in dict.fromkeys(a.split(":")[0] for a in sys.argv[1:]):
        j, m = best.get(c, low[c])
        print(f"{c}: годных {nok[c]}; {'лучшее' if c in best else 'НЕТ, мин. белое'} ss={j[1]} cx={j[3]} cy={j[4]} "
              f"ярк {m['mean']:.0f} белое {m['white']:.1f}% med {m['med']:.2f}")

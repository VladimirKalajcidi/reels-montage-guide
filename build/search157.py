"""Поиск стока для ролика 57 (слот 157): Pixabay + Pexels API. Превью -> scratch, подробности -> search157.log."""
import json, os, sys, urllib.request, urllib.parse
env = dict(l.strip().split("=", 1) for l in open(os.path.join(os.path.dirname(__file__), ".env")) if "=" in l)
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

import subprocess

def get(url, hdr=None):
    # у системного python нет корневых сертификатов — через curl
    h = sum((["-H", f"{k}: {v}"] for k, v in (hdr or {}).items()), [])
    return json.loads(subprocess.run(["curl", "-s", *h, url], capture_output=True, text=True).stdout)

def pixabay(q, n=16):
    u = "https://pixabay.com/api/videos/?" + urllib.parse.urlencode(dict(key=env["PIXABAY_API_KEY"], q=q, per_page=n, min_width=1280))
    return [("pb", h["id"], h["duration"], h["videos"]["medium"]["thumbnail"], h["tags"]) for h in get(u)["hits"]]

def pexels(q, n=8):
    u = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode(dict(query=q, per_page=n, orientation="landscape"))
    return [("px", v["id"], v["duration"], v["image"], v["url"].rstrip("/").split("/")[-1]) for v in get(u, {"Authorization": env["PEXELS_API_KEY"]})["videos"]]

log = open(__file__.replace(".py", ".log"), "a")
for q in sys.argv[2:]:
    for fn in (pixabay,) + ((pexels,) if "PEXELS_API_KEY" in env else ()):
        try:
            res = fn(q)
        except Exception as e:
            print(q, fn.__name__, "ERR", e); continue
        for src, i, d, th, tags in res:
            p = f"{OUT}/{src}_{i}.jpg"
            if not os.path.exists(p):
                try:
                    subprocess.run(["curl", "-s", "-o", p, th], check=True)
                except Exception:
                    continue
            log.write(f"{q}\t{src}\t{i}\t{d}s\t{tags}\n")
        print(q, fn.__name__, len(res))

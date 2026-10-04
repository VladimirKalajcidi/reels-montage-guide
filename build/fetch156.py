"""Скачивание стока ролика 56 (слот 156) в videos/56/stock/ + сверка длительности.
Pixabay API: `python3 fetch135.py <id> ...`; Mixkit (без API, запасной источник): `mk:<id>` —
assets.mixkit.co/videos/<id>/<id>-1080.mp4 (403 → -720), длительность сверяется с -720
(урок assets-manifest §3: -1080 бывает отдаёт чужой ролик)."""
import json, os, subprocess, sys
env = dict(l.strip().split("=", 1) for l in open(os.path.join(os.path.dirname(__file__), ".env")) if "=" in l)
DST = "/Users/vladimirkalajcidi/reels_challenge/videos/56/stock"
dur = lambda p: float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                                     capture_output=True, text=True).stdout or 0)
for arg in sys.argv[1:]:
    if arg.startswith("mk:"):
        vid = arg[3:]
        p = f"{DST}/stock_{vid}.mp4"
        base = f"https://assets.mixkit.co/videos/{vid}/{vid}"
        if not os.path.exists(p):
            for q in ("1080", "720"):
                subprocess.run(["curl", "-s", "-L", "-A", "Mozilla/5.0", "-o", p, f"{base}-{q}.mp4"])
                if dur(p) > 0:
                    break
        ref = f"/private/tmp/claude-501/-Users-vladimirkalajcidi-reels-challenge/9f9f2ab0-398e-4399-b090-55f2b4224bb6/scratchpad/v56/mk/_mk{vid}_360.mp4"
        subprocess.run(["curl", "-s", "-L", "-A", "Mozilla/5.0", "-o", ref, f"{base}-360.mp4"])
        d, dr = dur(p), dur(ref)
        wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                             "-of", "csv=p=0:s=x", p], capture_output=True, text=True).stdout.strip()
        print(f"mk {vid}: file {d:.2f}s {wh} | 360p {dr:.2f}s {'OK' if abs(d - dr) < 0.5 else 'РАСХОЖДЕНИЕ'}")
        continue
    vid = arg
    h = json.loads(subprocess.run(["curl", "-s", f"https://pixabay.com/api/videos/?key={env['PIXABAY_API_KEY']}&id={vid}"],
                                  capture_output=True, text=True).stdout)["hits"][0]
    v = h["videos"]["large"] if h["videos"]["large"]["url"] else h["videos"]["medium"]
    p = f"{DST}/stock_{vid}.mp4"
    if not os.path.exists(p):
        subprocess.run(["curl", "-s", "-L", "-o", p, v["url"]], check=True)
    d = dur(p)
    print(f"{vid}: api {h['duration']}s file {d:.2f}s {v['width']}x{v['height']} {'OK' if abs(d - h['duration']) < 1.5 else 'РАСХОЖДЕНИЕ'} | {h['tags']}")

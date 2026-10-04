"""Скачивание стока ролика 57 (слот 157) с Pixabay API в videos/57/stock/."""
import json, os, subprocess, sys
env = dict(l.strip().split("=", 1) for l in open(os.path.join(os.path.dirname(__file__), ".env")) if "=" in l)
DST = "/Users/vladimirkalajcidi/reels_challenge/videos/57/stock"
os.makedirs(DST, exist_ok=True)
for vid in (sys.argv[1:] or ["126558", "50365", "153460", "119", "107129", "3191", "215751", "20857", "69625"]):
    h = json.loads(subprocess.run(["curl", "-s", f"https://pixabay.com/api/videos/?key={env['PIXABAY_API_KEY']}&id={vid}"],
                                  capture_output=True, text=True).stdout)["hits"][0]
    v = h["videos"]["large"] if h["videos"]["large"]["url"] else h["videos"]["medium"]
    p = f"{DST}/stock_{vid}.mp4"
    if not os.path.exists(p):
        part = p + ".part"
        subprocess.run(["curl", "-s", "-L", "-o", part, v["url"]], check=True)
        os.replace(part, p)
    d = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                               "-of", "csv=p=0", p], capture_output=True, text=True).stdout)
    print(f"{vid}: api {h['duration']}s file {d:.2f}s {v['width']}x{v['height']} "
          f"{'OK' if abs(d - h['duration']) < 1.5 else 'РАСХОЖДЕНИЕ'} | {h['tags'][:60]}")

import cv2, numpy as np, sys

def face_metrics(path, n=30):
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    cap = cv2.VideoCapture(path)
    nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    for f in np.linspace(2, max(3, nf - 2), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, fr = cap.read()
        if not ok:
            continue
        fr = cv2.flip(fr, 1)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        det = casc.detectMultiScale(g, 1.15, 6, minSize=(80, 80))
        if len(det) == 0:
            continue
        x, y, w, h = max(det, key=lambda r: r[2]*r[3])
        roi = g[y:y+int(h*0.6), x:x+w]
        ee = eyes.detectMultiScale(roi, 1.12, 8, minSize=(18, 18))
        ey = y + int(np.mean([e[1]+e[3]/2 for e in ee])) if len(ee) >= 2 else y + int(h*0.42)
        rows.append((x + w//2, ey, w, fr.shape[1], fr.shape[0]))
    cap.release()
    if not rows:
        print("NO DETECTIONS"); return None
    a = np.array(rows, float)
    print(f"{path}: n={len(rows)} mean cx,eyeY,w,iw,ih:", np.round(a.mean(axis=0),1), "std:", np.round(a.std(axis=0),1))
    return a.mean(axis=0)

for p in sys.argv[1:]:
    face_metrics(p)

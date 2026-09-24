"""Машинные и смысловые проверки ролика 29."""
import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storyboard29 import (OUT, SHOTS, CAPS, DUR, GFX_ZONE, GFX_X,
                          FACE_KINDS, STOCK_KINDS, WORDS, shot_at)
from style import CARD_A, CARD_B, FPS, font
import render29

ALPHA = 40


def bright_outside():
    cap=cv2.VideoCapture(OUT); bad=[]
    for f in range(0,int(DUR*FPS),3):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f); ok,img=cap.read()
        if not ok: continue
        kind=shot_at(f/FPS)[2]
        x,y,w,h=CARD_B if kind in STOCK_KINDS else CARD_A
        mask=np.zeros(img.shape[:2],np.uint8); mask[y:y+h,x:x+w]=1
        g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if np.any((g>200)&(mask==0)): bad.append(round(f/FPS,2))
    cap.release(); return len(bad),bad[:8]


def line_overlaps():
    kmap={"r":"sans","i":"sans_it","s":"serif_it"}; boxes={}
    for i,(_,_,runs,_) in enumerate(CAPS):
        top=min(font(kmap[k],sz).getbbox(txt)[1] for txt,k,sz in runs)
        bot=max(font(kmap[k],sz).getbbox(txt)[3] for txt,k,sz in runs)
        boxes[i]=(top,bot)
    by={}
    for i in range(len(CAPS)):
        bid,pos,_,_,yoff=render29.BLOCKS[i]; by.setdefault(bid,[]).append((pos,i,yoff))
    bad=[]
    for bid,rows in by.items():
        rows.sort()
        for a,b in zip(rows,rows[1:]):
            ha=(boxes[a[1]][1]-boxes[a[1]][0])/2
            hb=(boxes[b[1]][1]-boxes[b[1]][0])/2
            if a[2]+ha>b[2]-hb: bad.append((bid,CAPS[a[1]][0],CAPS[b[1]][0]))
    return len(bad),bad[:8]


def gfx_text_overlap():
    bad=[]
    for f in range(0,int(DUR*FPS),2):
        t=f/FPS; t0,_,kind,_=shot_at(t); gl=render29.graphics_layer(kind,t-t0)
        cl=render29.caption_layer(t)
        if gl is None or cl is None: continue
        ga=np.array(gl.split()[3]); ca=np.array(cl.split()[3])
        n=int(np.count_nonzero((ga>ALPHA)&(ca>ALPHA)))
        if n: bad.append((round(t,2),kind,n))
    return len(bad),bad[:8]


def gfx_in_zone_and_not_clipped():
    outside=[]; clipped=[]
    for f in range(0,int(DUR*FPS),2):
        t=f/FPS; ts,_,kind,_=shot_at(t); gl=render29.graphics_layer(kind,t-ts)
        if gl is None: continue
        if kind in STOCK_KINDS:
            x0,y0,w,h=CARD_B; x1=x0+w; y1=y0+h
        else:
            y0,y1=GFX_ZONE; x0,x1=GFX_X
        a=np.array(gl.split()[3]); ys,xs=np.nonzero(a>ALPHA)
        if not len(ys): continue
        if ys.min()<y0 or ys.max()>y1 or xs.min()<x0 or xs.max()>x1:
            outside.append((round(t,2),kind,int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())))
        ys,xs=np.nonzero(a>120)
        if len(ys) and (ys.min()<y0+20 or ys.max()>y1-20 or xs.min()<x0+20 or xs.max()>x1-20):
            clipped.append((round(t,2),kind,int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())))
    return (len(outside),outside[:6]),(len(clipped),clipped[:6])


def text_in_card():
    bad=[]
    for f in range(0,int(DUR*FPS),2):
        t=f/FPS; kind=shot_at(t)[2]; cl=render29.caption_layer(t)
        if cl is None: continue
        x,y,w,h=CARD_B if kind in STOCK_KINDS else CARD_A
        a=np.array(cl.split()[3]); ys,xs=np.nonzero(a>120)
        if len(ys) and (xs.min()<x+30 or xs.max()>x+w-30 or ys.min()<y+30 or ys.max()>y+h-30):
            bad.append((round(t,2),int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max())))
    return len(bad),bad[:8]


def _words():
    return [(w["start"],w["end"],w["word"]) for w in json.load(open(WORDS))]


def speech_sync():
    bad=[]
    for t0,_,_,_ in SHOTS[1:]:
        for a,b,w in _words():
            if a+.02<t0<b-.02: bad.append((t0,w,a,b)); break
    return len(bad),bad


def caption_sync():
    bad=[]; words=_words()
    for t0,_,runs,_ in CAPS:
        if not any(abs(a-t0)<.26 for a,_,_ in words):
            bad.append((t0,"".join(x[0] for x in runs).strip()))
    return len(bad),bad[:8]


def caption_in_shot():
    bad=[]
    for t0,t1,runs,_ in CAPS:
        s0,s1,_,_=shot_at(t0)
        if t1>s1+1e-6: bad.append((t0,t1,s1,"".join(x[0] for x in runs)))
    return len(bad),bad[:8]


def caption_words_match():
    norm=lambda w:re.sub(r"[^а-яa-z0-9]","",w.lower().replace("ё","е"))
    said=[norm(w) for _,_,w in _words()]; said=[w for w in said if w]
    typed=[]
    for _,_,runs,_ in CAPS:
        for txt,_,_ in runs: typed += [norm(w) for w in txt.split()]
    typed=[w for w in typed if w]; i=0; missing=[]
    for w in typed:
        j=i
        while j<len(said) and said[j]!=w: j+=1
        if j<len(said): i=j+1
        else: missing.append(w)
    return len(missing),missing[:8]


def caption_style():
    bad=[]
    for t0,_,runs,_ in CAPS:
        text="".join(x[0] for x in runs).strip(); n=len(text.split())
        if text!=text.lower() or re.search(r'[.,!?;:«»]',text) or not 2<=n<=4:
            bad.append((t0,text,n))
    return len(bad),bad[:8]


def duplicate_numbers():
    bad=[]
    number_shots={"payout2","payout4","payout8","doubling","prob2","prob4",
                  "prob8","infiniteSum","stockSpb"}
    for t0,_,runs,_ in CAPS:
        if shot_at(t0)[2] in number_shots:
            bad.append((t0,shot_at(t0)[2],"".join(x[0] for x in runs)))
        if re.search(r'\d',"".join(x[0] for x in runs)):
            bad.append((t0,"digit-in-caption"))
    return len(bad),bad


def math_check():
    bad=[]; rows=[]
    for n in (1,2,3,4,8,16):
        prize=2**n; prob=1/(2**n); term=prize*prob
        rows.append((n,prize,prob,term))
        if abs(term-1)>1e-12: bad.append((n,term))
    return len(bad),bad,rows


def rhythm():
    lens=[b-a for a,b,_,_ in SHOTS]; face=sum(b-a for a,b,k,_ in SHOTS if k in FACE_KINDS)
    return dict(shots=len(SHOTS),avg=sum(lens)/len(lens),median=float(np.median(lens)),
                min=min(lens),max=max(lens),first_cut=SHOTS[0][1],face_pct=100*face/DUR)


def video_facts():
    cap=cv2.VideoCapture(OUT)
    out=dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             fps=cap.get(cv2.CAP_PROP_FPS),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release(); return out


def loudness():
    p=subprocess.run(["ffmpeg","-hide_banner","-nostats","-i",OUT,"-filter_complex",
                      "ebur128=peak=true","-f","null","-"],capture_output=True,text=True)
    txt=p.stderr; block=txt[txt.rfind("Summary:"):]
    grab=lambda pat: float(re.search(pat,block).group(1)) if re.search(pat,block) else None
    return dict(LUFS=grab(r"I:\s+(-?[0-9.]+) LUFS"),
                LRA=grab(r"LRA:\s+(-?[0-9.]+) LU"),
                TP=grab(r"Peak:\s+(-?[0-9.]+) dBFS"))


def ar_roll_rgb():
    cap=cv2.VideoCapture(OUT); vals=[]
    for t in (0.8,3.8,23.5,33.8,38.5,43.5,49.2):
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,im=cap.read()
        if ok:
            x,y,w,h=CARD_A; vals.append(im[y:y+h,x:x+w][:,:,::-1].mean((0,1)))
    cap.release(); return tuple(np.mean(vals,axis=0).round(1))


def run():
    zone,clip=gfx_in_zone_and_not_clipped()
    checks={
        "bright_outside":bright_outside(), "line_overlaps":line_overlaps(),
        "gfx_text_overlap":gfx_text_overlap(), "gfx_in_zone":zone,
        "gfx_clipped":clip, "text_in_card":text_in_card(),
        "speech_sync":speech_sync(), "caption_sync":caption_sync(),
        "caption_in_shot":caption_in_shot(), "caption_words_match":caption_words_match(),
        "caption_style":caption_style(), "duplicate_numbers":duplicate_numbers(),
        "math":math_check(), "rhythm":rhythm(), "video":video_facts(),
        "loudness":loudness(), "aroll_rgb":ar_roll_rgb(),
    }
    for k,v in checks.items(): print(f"{k:22}",v)
    return checks


if __name__=="__main__": run()

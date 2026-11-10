# extract first decodable frame (frame index k) of each head mp4 -> small png + npz
import cv2, glob, os, sys, numpy as np
heads, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True); nfr=[]
for p in sorted(glob.glob(f"{heads}/batch_*/*.mp4")):
    b, ep = p.split('/')[-2], os.path.basename(p)[:-4]
    cap = cv2.VideoCapture(p); fr=None; n=0
    while True:
        ok, f = cap.read()
        if not ok: break
        if n==k: fr=f
        n+=1
    nfr.append(n)
    if fr is None: print("fail", p); continue
    cv2.imwrite(f"{out}/{b}__{ep}.jpg", cv2.resize(fr,(640,360),interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("decodable frames in head chunks: min/med/max", min(nfr), int(np.median(nfr)), max(nfr))

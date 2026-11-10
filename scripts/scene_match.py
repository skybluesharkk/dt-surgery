# Chronologically sort episodes' first frames; estimate similarity transform between consecutive
# frames (SIFT + RANSAC). Large shift / low inliers => camera repositioned (scene boundary).
import cv2, json, glob, os, sys, numpy as np
meta = json.load(open(sys.argv[1])); ff = sys.argv[2]
meta.sort(key=lambda r: r["start"])
sift = cv2.SIFT_create(3000); bf = cv2.BFMatcher()
def feat(r):
    g = cv2.cvtColor(cv2.imread(f"{ff}/{r['batch']}__{r['ep']}.jpg"), cv2.COLOR_BGR2GRAY)
    return sift.detectAndCompute(cv2.createCLAHE(2.0,(8,8)).apply(g), None)
def match(a, b):
    (ka,da),(kb,db) = a,b
    if da is None or db is None: return 0, None
    m = [x for x,y in bf.knnMatch(da,db,k=2) if x.distance < 0.75*y.distance]
    if len(m) < 8: return 0, None
    pa = np.float32([ka[x.queryIdx].pt for x in m]); pb = np.float32([kb[x.trainIdx].pt for x in m])
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4)
    if M is None: return 0, None
    s = np.hypot(M[0,0], M[1,0]); ang = np.degrees(np.arctan2(M[1,0], M[0,0]))
    c = M @ np.array([320,180,1.0]) - np.array([320,180])   # shift of image centre (px @640x360)
    return int(inl.sum()), (float(np.hypot(*c)), float(s), float(ang))
F = [feat(r) for r in meta]
out = []
for i,r in enumerate(meta):
    n, t = match(F[i-1], F[i]) if i else (0, None)
    out.append(dict(batch=r["batch"], ep=r["ep"], start=r["start"], inl=n, T=t))
json.dump(out, open(sys.argv[3],"w"))

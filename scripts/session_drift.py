# Assign sessions by large time gaps (>20 min) and measure each episode's first frame against
# session anchors (similarity transform, SIFT+RANSAC). Writes data/sessions.json
import cv2, json, sys, numpy as np, collections
meta = sorted(json.load(open(sys.argv[1])), key=lambda r: r["start"]); ff = sys.argv[2]
sess=[0]
for a,b in zip(meta, meta[1:]): sess.append(sess[-1] + (b["start"]-(a["start"]+a["dur"]) > 25*60))
sift = cv2.SIFT_create(3000); bf = cv2.BFMatcher(); cl=cv2.createCLAHE(2.0,(8,8))
def feat(r): return sift.detectAndCompute(cl.apply(cv2.cvtColor(cv2.imread(f"{ff}/{r['batch']}__{r['ep']}.jpg"),cv2.COLOR_BGR2GRAY)),None)
def match(a,b):
    (ka,da),(kb,db)=a,b
    m=[x for x,y in bf.knnMatch(da,db,k=2) if x.distance<0.75*y.distance]
    if len(m)<8: return 0,None
    pa=np.float32([ka[x.queryIdx].pt for x in m]); pb=np.float32([kb[x.trainIdx].pt for x in m])
    M,inl=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=4)
    if M is None or inl.sum()<12: return int(0 if inl is None else inl.sum()),None
    c=M@np.array([320,180,1.])-np.array([320,180])
    return int(inl.sum()),(float(c[0]),float(c[1]),float(np.hypot(M[0,0],M[1,0])),float(np.degrees(np.arctan2(M[1,0],M[0,0]))))
F=[feat(r) for r in meta]; res=[]
for s in sorted(set(sess)):
    idx=[i for i in range(len(meta)) if sess[i]==s]
    # anchor = episode with most matches to 10 evenly spaced others (robust central frame)
    probe=idx[::max(1,len(idx)//10)]
    a=max(probe,key=lambda i: sum(match(F[i],F[j])[0] for j in probe if j!=i))
    for i in idx:
        n,T=match(F[a],F[i]) if i!=a else (999,(0,0,1,0))
        res.append(dict(session=s,batch=meta[i]["batch"],ep=meta[i]["ep"],start=meta[i]["start"],anchor=meta[a]["ep"],inl=n,T=T))
json.dump(res,open(sys.argv[3],"w"))
import datetime as dt
f=lambda t: dt.datetime.fromtimestamp(t,dt.timezone(dt.timedelta(hours=9))).strftime('%H:%M')
for s in sorted(set(sess)):
    R=[r for r in res if r["session"]==s]; ok=[r for r in R if r["T"]]
    sh=np.array([np.hypot(r["T"][0],r["T"][1]) for r in ok]); sc=np.array([r["T"][2] for r in ok]); an=np.array([r["T"][3] for r in ok])
    print(f"session {s}: {f(R[0]['start'])}-{f(R[-1]['start'])} n={len(R)} anchor={R[0]['anchor']} matched={len(ok)} | shift px@640 med {np.median(sh):.1f} p90 {np.percentile(sh,90):.1f} max {sh.max():.1f} | scale {sc.min():.2f}-{sc.max():.2f} | rot {an.min():.1f}..{an.max():.1f} deg")
    print("   batches:", dict(collections.Counter(r["batch"][-2:] for r in R)))
    bad=[r for r in R if not r["T"] or np.hypot(r["T"][0],r["T"][1])>40]
    print("   unmatched/large-shift:", [(f(r['start']),r['batch'][-2:],r['ep'][-6:],r['inl']) for r in bad][:20])

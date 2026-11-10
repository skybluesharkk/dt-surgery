import json,sys,numpy as np
D=json.load(open(sys.argv[1])); S={(r['batch']+'/'+r['ep']):r['session'] for r in json.load(open(sys.argv[2]))}
L=json.load(open(sys.argv[3])); dur={r['batch']+'/'+r['ep']:r['dur'] for r in L}
rows=[]
for k,ts in D.items():
    a=np.array(ts); m1=a[:,1]; m2=a[:,2]
    rows.append(dict(ep=k,s=S[k],p95=float(np.percentile(m1,95)),mx=float(m1.max()),p95_2=float(np.percentile(m2,95)),end=float(m1[-5:].mean())))
p=np.array([r['p95'] for r in rows]); p2=np.array([r['p95_2'] for r in rows])
print(f"N={len(rows)}  camera-shift lower bound (min-region, p95 over frames, px@1920): median {np.median(p):.1f}, 75th {np.percentile(p,75):.1f}, 90th {np.percentile(p,90):.1f}, max {p.max():.1f}")
print(f"   2nd-smallest region: median {np.median(p2):.1f}, 90th {np.percentile(p2,90):.1f}")
for th in (5,10,20,40): print(f"   episodes with p95 < {th:2d}px: {np.mean(p<th)*100:4.0f}%   (2nd region: {np.mean(p2<th)*100:3.0f}%)")
for s in range(4):
    q=[r['p95'] for r in rows if r['s']==s]; print(f"   session {s}: n={len(q)} median {np.median(q):.1f}px, max {max(q):.1f}px, <10px {np.mean(np.array(q)<10)*100:.0f}%")
rows.sort(key=lambda r:-r['p95']); print("largest:", [(r['ep'],round(r['p95'],1)) for r in rows[:10]])
json.dump(rows,open(sys.argv[4],"w"))

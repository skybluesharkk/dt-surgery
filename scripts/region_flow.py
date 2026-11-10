# median DIS optical flow (frame0 -> t) in a 3x3 grid of regions; plot dy/dx over time. px @1920
import cv2,sys,numpy as np,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sys.argv[1]; eps=sys.argv[2:]
fig,axs=plt.subplots(len(eps),2,figsize=(14,3.2*len(eps)),squeeze=False)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
for r,p in enumerate(eps):
    cap=cv2.VideoCapture(p); fps=cap.get(cv2.CAP_PROP_FPS); g0=None; T=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        g=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(640,360),interpolation=cv2.INTER_AREA)
        if g0 is None: g0=g
        fl=dis.calc(g0,g,None)
        T.append([[np.median(fl[j*120:(j+1)*120,i*213:(i+1)*213,c]) for c in (0,1)] for j in range(3) for i in range(3)])
    T=np.array(T)*3; t=np.arange(len(T))/fps
    for c,name in ((0,'dx'),(1,'dy')):
        for k in range(9): axs[r,c].plot(t,T[:,k,c],label=f"r{k//3}c{k%3}",lw=1)
        axs[r,c].set_title(f"{p.split('/')[-1][:-4]}  {name} median flow vs frame0 (px@1920)",fontsize=9); axs[r,c].set_ylim(-80,80); axs[r,c].grid(alpha=.3)
    axs[r,0].legend(fontsize=6,ncol=3)
    allk=np.abs(T).max(axis=2)  # per frame per region
    print(p.split('/')[-1], "frames",len(T)," min-over-regions |flow| p95: %.1fpx  median-over-regions p95: %.1fpx"%(np.percentile(allk.min(1),95),np.percentile(np.median(allk,1),95)))
plt.tight_layout(); plt.savefig(out,dpi=80)

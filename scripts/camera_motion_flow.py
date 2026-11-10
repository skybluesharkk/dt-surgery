# Per-episode camera-motion lower bound: DIS flow frame0->t, median per region (3x3 grid);
# camera motion moves every region, so min over regions of |median flow| bounds camera shift.
# Output per episode: per-frame min-region magnitude and the min-region vector (px @1920).
import cv2,sys,glob,os,json,numpy as np
from concurrent.futures import ProcessPoolExecutor
step=int(sys.argv[3])
def run(p):
    cv2.setNumThreads(1); dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    cap=cv2.VideoCapture(p); g0=None; i=0; ts=[]
    while cap.grab():
        if i%step==0:
            _,f=cap.retrieve(); g=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(640,360),interpolation=cv2.INTER_AREA)
            if g0 is None: g0=g
            fl=dis.calc(g0,g,None)
            v=np.array([[np.median(fl[j*120:(j+1)*120,k*213:(k+1)*213,c]) for c in (0,1)] for j in range(3) for k in range(3)])*3
            m=np.hypot(v[:,0],v[:,1]); o=np.argsort(m)
            ts.append([i,float(m[o[0]]),float(m[o[1]]),float(np.median(m)),int(o[0])])
        i+=1
    return p.split('/')[-2]+'/'+os.path.basename(p)[:-4], ts
paths=sorted(glob.glob(f"{sys.argv[1]}/batch_*/*.mp4"))
with ProcessPoolExecutor(14) as ex: res=dict(ex.map(run,paths))
json.dump(res,open(sys.argv[2],"w")); print(len(res))

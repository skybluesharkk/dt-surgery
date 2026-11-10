# save frames at given fractions of an episode into a grid with grid-lines for comparing alignment
import cv2, sys, numpy as np, json
p, out = sys.argv[1], sys.argv[2]; fr=[float(x) for x in sys.argv[3].split(',')]
cap=cv2.VideoCapture(p); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); tiles=[]
for f in fr:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f*(n-1))); ok,im=cap.read(); im=cv2.resize(im,(640,360))
    for x in range(0,640,80): cv2.line(im,(x,0),(x,359),(0,255,255),1)
    for y in range(0,360,60): cv2.line(im,(0,y),(639,y),(0,255,255),1)
    cv2.putText(im,f"t={f:.2f} f{int(f*(n-1))}",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),3); cv2.putText(im,f"t={f:.2f} f{int(f*(n-1))}",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    tiles.append(im)
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))

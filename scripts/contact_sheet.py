# grid of frames: usage contact_sheet.py out.jpg cols label_mode img1 img2 ...
import cv2, sys, numpy as np, os
out, cols = sys.argv[1], int(sys.argv[2]); paths = sys.argv[3:]
tiles=[]
for p in paths:
    im = cv2.resize(cv2.imread(p),(320,180)); lab=os.path.basename(p).replace('.jpg','').replace('batch_00','b').replace('20260830_','')
    cv2.putText(im, lab, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3); cv2.putText(im, lab, (4,16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
    tiles.append(im)
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[i:i+cols]) for i in range(0,len(tiles),cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

import face_alignment as FA
fa = FA.FaceAlignment(FA.LandmarksType._2D,device="cpu", flip_input=False)
import cv2
import matplotlib.pylab as plt
import sys
import os

f=sys.argv[1]
print(f)
img=cv2.imread(f)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
preds = fa.get_landmarks(img)
for mark in preds[0]:
    cv2.circle(img,(mark[0],mark[1]),2,(255,0,0),-1)
plt.imshow(img)
plt.show()

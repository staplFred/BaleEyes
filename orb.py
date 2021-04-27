import cv2
import numpy as np
import PossibleChar
import os

tpl = cv2.imread("./images/labels/template1.jpg", 0)
# tpl = cv2.imread("./images/labels/0265003.jpg", 0)
img = cv2.imread("./images/bales/bales.jpg", 0)

# (keypoints, scaling pyramid factor)
orb = cv2.ORB_create(2000, 1.5)

(kp1, des1) = orb.detectAndCompute(img,None)
(kpTpl, desTpl) = orb.detectAndCompute(tpl,None)

#brute force matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, desTpl, k=2)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
# matches = bf.match(des1, desTpl)

#sort matches based on distance.  Least distance is better
# matches = sorted(matches, key=lambda val: val.trainIdx)

# for m,n in matches:
#     # print(m.distance, m.trainIdx, m.queryIdx, m.imgIdx)
#     print(m.distance, n.distance)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        print(m)
        good.append([m])


img3 = cv2.drawMatchesKnn(img, kp1, tpl, kpTpl, matches, None, flags=2)

# for marker in matches:
#     img2 = cv2.drawMarker(img.copy(), tuple(int(i) for i in marker.pt), color=(0,255,0))
# cv2.imshow('Train', img_Train)

# cv2.imshow('Original', img)
# cv2.imshow('gray', gray)
cv2.imshow('img3', img3)

cv2.waitKey(0)
cv2.destroyAllWindows()


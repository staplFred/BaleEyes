import cv2
import numpy as np
import PossibleChar
import os

imgTemplate = cv2.imread("./images/labels/template1.jpg", 0)
img = cv2.imread("./images/bales/bales.jpg", 0)

# (keypoints, scaling pyramid factor)
orb = cv2.ORB_create(1000, 1.2)

(kp1, des1) = orb.detectAndCompute(img,None)
(kp2, des2) = orb.detectAndCompute(imgTemplate,None)

#brute force matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

#sort matches based on distance.  Least distance is better
matches = sorted(matches, key=lambda val: val.distance)

print(matches)

# for marker in matches:
#     img2 = cv2.drawMarker(img.copy(), tuple(int(i) for i in marker.pt), color=(0,255,0))
# cv2.imshow('Train', img_Train)

# cv2.imshow('Original', img)
# cv2.imshow('gray', gray)
# cv2.imshow('img2', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()


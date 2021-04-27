import cv2
import numpy as np

print(cv2.__version__)

# img = cv2.imread("./images/bales/bales.jpg")
img = cv2.imread("./images/bales/bales2.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# corners = dst[dst>0.01*dst.max()]

corners = cv2.goodFeaturesToTrack(gray,250,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow('dst', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


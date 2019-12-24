import cv2
import numpy as numpy
import os

print(cv2.__version__)

def gotClicked(e,x,y,flags,params):
    if e==cv2.EVENT_LBUTTONDOWN:
        print('Coords: ', x, y)

imgOriginal = cv2.imread("./images/bales/bales.jpg")               # open image

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', gotClicked)

if imgOriginal is None:                            # if image was not read successfully
    print("\nerror: image not read from file \n\n")  # print error message to std out
    os.system("pause")                                  # pause so user can see error message
    # end if

cv2.imshow('frame', imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()


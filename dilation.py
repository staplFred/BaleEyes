import cv2
import numpy as np
import PossibleChar
import os

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

GAUSSIAN_SMOOTH_FILTER_SIZE = (7, 7)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8
MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 0.75
MIN_PIXEL_AREA = 80

print(cv2.__version__)

###################################################################################################
def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape

    imgHSV = np.zeros((height, width, 3), np.uint8)

    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)

    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)

    return imgValue
# end function

###################################################################################################
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat
# end function

###################################################################################################
def checkIfPossibleChar(possibleChar):
            # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
            # note that we are not (yet) comparing the char to other chars to look for a group

    if (possibleChar.intBoundingRectArea > MIN_PIXEL_AREA and
        possibleChar.intBoundingRectWidth > MIN_PIXEL_WIDTH and possibleChar.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
        MIN_ASPECT_RATIO < possibleChar.fltAspectRatio and possibleChar.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False
    # end if
# end function

###################################################################################################
def findPossibleChars(img):
    listOfPossibleChars = []                # this will be the return value
    intCountOfPossibleChars = 0
    imgCopy = img.copy()
    contours, npaHierarchy = cv2.findContours(imgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours
    for i in range(0, len(contours)):                       # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[i])
        if (not checkIfPossibleChar(possibleChar)):
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # increment count of possible chars
            listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars
    return listOfPossibleChars
# end function


imgOriginal = cv2.imread("./images/bales/bales.jpg")               # open image
imgGrayscale = extractValue(imgOriginal)
imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)

height, width = imgGrayscale.shape
imgBlurred = np.zeros((height, width, 1), np.uint8)
imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)

kernel = np.ones((1,1), np.uint8)
img_erosion = cv2.erode(imgBlurred, kernel, iterations=1) 
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
imgThresh = cv2.adaptiveThreshold(img_dilation, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)

pcs = findPossibleChars(imgThresh)
for pc in pcs:
    # attrs = vars(pcs[i])
    # print(', '.join("%s: %s" % item for item in attrs.items()))
    x1 = pc.intBoundingRectX
    y1 = pc.intBoundingRectY
    x2 = pc.intBoundingRectX + pc.intBoundingRectWidth
    y2 = pc.intBoundingRectY + pc.intBoundingRectHeight
    cv2.rectangle(imgOriginal, (x1,y1), (x2,y2), SCALAR_GREEN)

# cv2.imshow('maxContrast', imgMaxContrastGrayscale)
# cv2.imshow('Blurred', imgBlurred)
cv2.imshow('Thresh', imgThresh)
cv2.imshow('Original', imgOriginal)
# cv2.imshow('Erosion', img_erosion) 
# cv2.imshow('Dilation', img_dilation) 

cv2.waitKey(0)
cv2.destroyAllWindows()


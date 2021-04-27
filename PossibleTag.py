import cv2
import numpy as np
import math

# these would be relative to the distance from the label
MIN_CONTOUR_AREA = 10000
MAX_CONTOUR_AREA = 20000
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 0.8

class PossibleTag:

    def __init__(self, _contour):
        self.contour = _contour
        self.boundingRect = cv2.boundingRect(self.contour)
        self.area = cv2.contourArea(self.contour)
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intX = intX
        self.intY = intY
        self.intWidth = intWidth
        self.intHeight = intHeight
        self.intBoundingRectArea = self.intWidth * self.intHeight
        self.intCenterX = (self.intX + self.intX + self.intWidth) / 2
        self.intCenterY = (self.intY + self.intY + self.intHeight) / 2
        self.fltDiagonalSize = math.sqrt((self.intWidth ** 2) + (self.intHeight ** 2))
        self.fltAspectRatio = float(self.intWidth) / float(self.intHeight)
        self.aspectRatio = self.intWidth / self.intHeight

    def meetsCriteria(self):
        if self.area < MIN_CONTOUR_AREA or self.area > MAX_CONTOUR_AREA:
            return False
        if self.aspectRatio < MIN_ASPECT_RATIO or self.aspectRatio > MAX_ASPECT_RATIO:
            return False
        return True

    def setROI(self, image):
        self.ROI = image[self.intY : self.intY + self.intHeight,
                           self.intX : self.intX + self.intWidth]

# end class



1

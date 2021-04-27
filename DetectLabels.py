import cv2
import numpy as np
import Main
import math
import random

import Preprocess
import DetectChars
import PossibleChar
import PossibleLabel

# module level variables ##########################################################################
PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5

def detectLabels(img):
    listOfPossibleLabels = []
    h, w, c = img.shape

    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(img)         # preprocess to get grayscale and threshold images

    if Main.showSteps == True: # show steps #######################################################
        # cv2.imshow("GrayScale", imgGrayscaleScene)
        # cv2.moveWindow("GrayScale", 25, 25)
        cv2.imshow("Thresh", imgThreshScene)
        cv2.moveWindow("Thresh", 50, 50)
        
    listOfPossibleCharsInScene = findPossibleCharsInScene(imgThreshScene, img)

    if Main.showSteps == True or Main.showStep1 == True: # show steps #######################################################
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 131 with MCLRNF1 image

        imgContours = np.zeros((h, w, 3), np.uint8)

        contours = []
        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)
        # end for

        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("Contours", imgContours)
        cv2.moveWindow("Contours", 0, 0)
    # end if # show steps #########################################################################

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Main.showSteps == True: # show steps #######################################################
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        imgContours = np.zeros((h, w, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)
            # end for

            cv2.drawContours(imgContours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

            cv2.drawContours(imgContours, contours, -1, Main.SCALAR_WHITE)
            cv2.imshow("Contours", imgContours)
            cv2.moveWindow("Contours", 0, 0)

        # end for

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:                   # for each group of matching chars
        possibleLabel = extractPlate(img, listOfMatchingChars)         # attempt to extract plate

        if possibleLabel.imgPlate is not None:                          # if plate was found
            listOfPossibleLabels.append(possibleLabel)                  # add to list of possible plates
        # end if
    # end for

    print("\n" + str(len(listOfPossibleLabels)) + " possible labels found")  # 13 with MCLRNF1 image

    return listOfPossibleLabels


def findPossibleCharsInScene(imgThresh, imgContours):
    listOfPossibleChars = []                # this will be the return value

    intCountOfPossibleChars = 0

    imgThreshCopy = imgThresh.copy()

    contours, npaHierarchy = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   # find all contours

    height, width = imgThresh.shape
    imgContours = np.zeros((height, width, 3), np.uint8)

    if Main.showContours == True: # show steps ###################################################
        cv2.drawContours(imgContours, contours, -1, Main.SCALAR_RED)
        cv2.drawContours(imgThresh, contours, -1, Main.SCALAR_RED)
    # # end if # show steps #####################################################################

    for i in range(0, len(contours)):                       # for each contour
        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):                   # if contour is a possible char, note this does not compare to other chars (yet) . . .
            intCountOfPossibleChars = intCountOfPossibleChars + 1           # increment count of possible chars
            listOfPossibleChars.append(possibleChar)                        # and add to list of possible chars
            x1 = possibleChar.intBoundingRectX
            y1 = possibleChar.intBoundingRectY
            x2 = possibleChar.intBoundingRectX + possibleChar.intBoundingRectWidth
            y2 = possibleChar.intBoundingRectY + possibleChar.intBoundingRectHeight
            cv2.rectangle(imgContours, (x1,y1), (x2,y2), Main.SCALAR_GREEN)
        # end if
    # end for

    if Main.showContours == True: # show steps #######################################################
        print("\nstep 2 - len(contours) = " + str(len(contours)))  # 2362 with MCLRNF1 image
        print("step 2 - intCountOfPossibleChars = " + str(intCountOfPossibleChars))  # 131 with MCLRNF1 image
        cv2.imshow("imgContours", imgContours)
        cv2.imshow("Thresh", imgThresh)
    # end if # show steps #########################################################################

    return listOfPossibleChars
# end function


###################################################################################################
def extractPlate(imgOriginal, listOfMatchingChars):
    possibleLabel = PossibleLabel.PossibleLabel()           # this will be the return value
    if len(listOfMatchingChars) != 7:
        return possibleLabel

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)        # sort chars from left to right based on x position

    # calculate the center point of the plate
    fltPlateCenterX = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in listOfMatchingChars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(listOfMatchingChars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possibleLabel.rrLocationOfPlateInScene = ( tuple(ptPlateCenter), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg )

            # final steps are to perform the actual rotation

            # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = imgOriginal.shape      # unpack original image width and height

    imgRotated = cv2.warpAffine(imgOriginal, rotationMatrix, (width, height))       # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possibleLabel.imgPlate = imgCropped         # copy the cropped plate image into the applicable member variable of the possible plate

    print(len(listOfMatchingChars))
    cv2.imshow('extracted', imgCropped)
    cv2.waitKey(0)

    return possibleLabel
# end function

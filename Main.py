import cv2
import numpy as numpy
import os

import DetectLabels

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = True

def main():
    imgOriginal = cv2.imread("./images/bales/bales.jpg")               # open image

    if imgOriginal is None:                            # if image was not read successfully
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")                                  # pause so user can see error message
        return                                              # and exit program
    # end if

    if showSteps == True: # show steps #######################################################
        cv2.imshow("Original", imgOriginal)

    listOfPossibleLabels = DetectLabels.detectLabels(imgOriginal)           # detect plates

    if len(listOfPossibleLabels) == 0:                          # if no plates were found
        print("\nno license plates were detected\n")  # inform user no plates were found

###################################################################################################
if __name__ == "__main__":
    main()
    if showSteps == True: # show steps #######################################################
        cv2.waitKey(0)
        cv2.destroyAllWindows()

import cv2
import os
import glob
import PossibleTag

print(cv2.__version__)

# Globals
lh = 0
ls = 0
lv = 0
hh = 80
hs = 255
hv = 255

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (0, 0, 255)
fontThickness = 2

def nothing(x):
    pass

def processImage(imagePath):
    print('Processing ', imagePath)
    frame = cv2.imread(imagePath)
    if frame is None:
        print('File Not found: ', imagePath)
        quit()
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(frame_HSV, (lh, ls, lv), (hh, hs, hv))
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print('contours found: ', len(contours))
    tags = findTags(contours)
    for i, tag in enumerate(tags):
        cv2.drawContours(frame, tag.contour, -1, (0,0,255), 3)
        cv2.putText(frame, str(i), (tag.intX, tag.intY), font, fontScale, fontColor, fontThickness )      

    smallerFrame = cv2.resize(frame, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
    return smallerFrame

def findTags(contours):
    tags = []
    for i, contour in enumerate(contours):
        tag = PossibleTag.PossibleTag(contour)
        if tag.meetsCriteria():
            tags.append(tag)  
    return tags

    # print('# tags: ', len(possibleTags))
    # smaller = cv2.resize(frame, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
    # cv2.imshow('Original', smaller)   
    # cv2.waitKey(0)


cv2.namedWindow('Original')
cv2.moveWindow('Original', 0, 0)
# cv2.namedWindow('Thresh')
# cv2.moveWindow('Thresh', 700, 0)


# path = './images/bales/bb*.*'
# images = os.listdir(path)
images =  glob.glob('./images/bales/bb*.*')
for idx, image in enumerate(images):
    print('Processing image ', str(idx), '-', image)
    frame = processImage(image)
    cv2.imshow('Original', frame)   
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# if cv2.waitKey(1)==ord('q'):
#     break
cv2.destroyAllWindows()

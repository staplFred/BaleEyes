import cv2
import os
import glob
import PossibleTag
import Preprocess

print(cv2.__version__)

boxes = []

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

def processImage(frame):
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(frame_HSV, (lh, ls, lv), (hh, hs, hv))
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tags = findTags(contours)
    print('Tags found: ',  len(tags))
    for i, tag in enumerate(tags):
        roi = tag.ROI = frame[tag.intY : tag.intY + tag.intHeight,
                           tag.intX : tag.intX + tag.intWidth]
        # cv2.imshow('roi', larger(roi))
        cv2.drawContours(frame, tag.contour, -1, (0,0,255), 3)
        # cv2.putText(frame, str(i), (tag.intX, tag.intY), font, fontScale, fontColor, fontThickness )      
        # cv2.waitKey(0)

    return frame

def findBales(imagePath):
    frame = cv2.imread(imagePath)
    if frame is None:
        print('File Not found: ', imagePath)
        quit()
    imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(frame)
    edged = cv2.Canny(imgGrayscaleScene, 100, 200)
    contours, hierarchy = cv2.findContours(imgThreshScene, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,0,255), 3)

    gray = cv2.resize(imgGrayscaleScene, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
    thresh = cv2.resize(imgThreshScene, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
    edged = cv2.resize(edged.copy(), None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
    smallerFrame = cv2.resize(frame, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )

    cv2.imshow('Original', smallerFrame)
    cv2.imshow('Thresh', thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findTags(contours):
    tags = []
    for i, contour in enumerate(contours):
        tag = PossibleTag.PossibleTag(contour)
        if tag.meetsCriteria():
            tags.append(tag)  
    return tags

def show_rgb_equalized(image):
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)
    cv2.imshow('eq', smaller(eq_image))

def show_hsv_equalized(image):
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(H)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
    cv2.imshow('hsv_eq', smaller(eq_image))

def smaller(image):
    smallerImage = cv2.resize(image, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
    return smallerImage

def larger(image):
    return cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_AREA)

def on_mouse(event, x, y, flags, params):
   # global img

   if event == cv2.EVENT_LBUTTONDOWN:
      print('Start Mouse Position: ', str(x), str(y))
      sbox = [x, y]
      boxes.append(sbox)

   elif event == cv2.EVENT_LBUTTONUP:
      print('End Mouse Position: ', str(x), str(y))
      ebox = [x, y]
      boxes.append(ebox)


cv2.namedWindow('Original')
cv2.moveWindow('Original', 0, 0)
# cv2.namedWindow('Thresh')
# cv2.moveWindow('Thresh', 700, 0)
# cv2.namedWindow('eq')
# cv2.moveWindow('eq', 700, 0)


imagePath = './images/bales/bb_ot4.jpg'
frame = cv2.imread(imagePath)
overlay = frame.copy()
alpha = 0.4

if frame is None:
    print('File Not found: ', imagePath)
    quit()
#frame = processImage(frame)
cv2.rectangle(overlay, (292*4,245*4), (419*4,587*4), (0,255,0), -1)
cv2.rectangle(overlay, (429*4,593*4), (561*4,916*4), (0,0,255), -1)
frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)    
cv2.putText(frame, "1211562", (292*4+75,245*4+100), 
    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 7)
cv2.putText(frame, "1211562", (429*4+75, 593*4+100), 
    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 7)
cv2.imshow('Original', smaller(frame))   

while True:
    cv2.setMouseCallback('Original', on_mouse, 0)
    kp = cv2.waitKey(1)
    if kp > 0:
        print('key pressed: ', kp)
    if kp==ord('q'):
        break

cv2.imwrite('cgwa.jpg', frame)
# cv2.waitKey(0)
cv2.destroyAllWindows()

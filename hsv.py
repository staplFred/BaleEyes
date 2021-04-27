import cv2
print(cv2.__version__)

MIN_CONTOUR_AREA = 8000
MAX_CONTOUR_AREA = 12000
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 0.7

# Globals
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value

def nothing(x):
    pass

cv2.namedWindow('Original')
cv2.moveWindow('Original', 0, 0)

cv2.namedWindow('Thresh')
cv2.moveWindow('Thresh', 700, 0)

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars', 1400, 0)

cv2.createTrackbar('lo_Hue',  'Trackbars', 0, 179, nothing)
cv2.createTrackbar('hi_Hue',  'Trackbars', 80, 179, nothing)
cv2.createTrackbar('lo_Sat',  'Trackbars', 0, 255, nothing)
cv2.createTrackbar('hi_Sat',  'Trackbars', 255, 255, nothing)
cv2.createTrackbar('lo_Val',  'Trackbars', 0, 255, nothing)
cv2.createTrackbar('hi_Val',  'Trackbars', 255, 255, nothing)
cv2.createTrackbar('another_Val',  'Trackbars', 0, 255, nothing)

frame = cv2.imread('../images/blue_stack_orange_tags.jpg')
# frame = cv2.imread('../images/small.jpg')
frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lh = cv2.getTrackbarPos('lo_Hue', 'Trackbars')
ls = cv2.getTrackbarPos('lo_Sat', 'Trackbars')
lv = cv2.getTrackbarPos('lo_Val', 'Trackbars')

hh = cv2.getTrackbarPos('hi_Hue', 'Trackbars')
hs = cv2.getTrackbarPos('hi_Sat', 'Trackbars')
hv = cv2.getTrackbarPos('hi_Val', 'Trackbars')

thresh = cv2.inRange(frame_HSV, (lh, ls, lv), (hh, hs, hv))
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

smallThresh = cv2.resize(thresh, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
cv2.imshow('Thresh', smallThresh)

possibleTags = []

for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
        continue
    [intX, intY, intW, intH] = cv2.boundingRect(contours[i])
    aspectRatio = float(intW)/intH
    if aspectRatio < MIN_ASPECT_RATIO or aspectRatio > MAX_ASPECT_RATIO:
        continue
    print('area: ', area)
    print('a/r: ', aspectRatio)
    possibleTags.append(contours[i])        
    # cv2.drawContours(frame, contours, i, (0,0,255), 3)
    cv2.rectangle(frame,                        # draw rectangle on original training image
                  (intX, intY),                 # upper left corner
                  (intX+intW,intY+intH),        # lower right corner
                  (0, 0, 255),                  # red
                   2)                            # thickness

print('# tags: ', len(possibleTags))
smaller = cv2.resize(frame, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA )
cv2.imshow('Original', smaller)   
cv2.waitKey(0)


    # if cv2.waitKey(1)==ord('q'):
    #     break

cv2.destroyAllWindows()

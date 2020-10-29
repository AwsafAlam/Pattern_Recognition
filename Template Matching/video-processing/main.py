import numpy as np
import cv2

INPUT_FILE = "input.mov"
FRAMES_DIR = "./frames"

vc = cv2.VideoCapture(INPUT_FILE)
c=1

if vc.isOpened():
    rval , frame = vc.read()
else:
    rval = False

while rval:
    cv2.imwrite((FRAMES_DIR + "/" +str(c) + '.jpg'),frame)
    c = c + 1
    cv2.waitKey(1)
    rval, frame = vc.read()
vc.release()

# cap = cv2.VideoCapture("C:\\Python27\\clip1.avi")
# fgbg = cv2.BackgroundSubtractorMOG()
# while(1):
#     ret, frame = cap.read()

#     fgmask = fgbg.apply(frame)
#     # res,thresh = cv2.threshold(fgmask,127,255,0)
#     kernel = np.ones((10,10),np.uint8)
#     dilation = cv2.dilate(fgmask,kernel,iterations = 1)
#     erosion = cv2.erode(fgmask,kernel,iterations = 1)
#     contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#     for i in range(0, len(contours)):
#         if (i % 1 == 0):
#             cnt = contours[i]


#             x,y,w,h = cv2.boundingRect(cnt)
#             cv2.drawContours(fgmask ,contours, -1, (255,255,0), 3)
#             cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)



# cv2.imshow('frame',fgmask)
# cv2.imshow("original",frame)

# if cv2.waitKey(30) == ord('a'):
#     break
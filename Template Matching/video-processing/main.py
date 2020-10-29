import numpy as np
import cv2
import glob
import os
from os.path import isfile, join

INPUT_FILE = "input.mov"
OUTPUT_FILE = "output.mov"
REFERENCE = "reference.jpg"
FRAMES_DIR = "./frames/"
MODIFIED_FRAMES_DIR = "./out_frames/"
FRAME_RATE = 15

reference = 0
#  Follow
# https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481

def extract_frames():
  """
  Extracts video frames into images
  """
  vc = cv2.VideoCapture(INPUT_FILE)
  c=1

  if vc.isOpened():
      rval , frame = vc.read()
  else:
      rval, frame = False, False

  while rval:
      # cv2.imwrite((MODIFIED_FRAMES_DIR + 'img' + str(c) + '.jpg'),frame)
      cv2.imwrite((MODIFIED_FRAMES_DIR + str(c) + '.jpg'),frame)
      c = c + 1
      cv2.waitKey(1)
      rval, frame = vc.read()
  vc.release()
  print("All frames extracted successfully...")


def create_video():
  """
  Creates output video from images
  """
  print("Generating output video")
  frame_array = []
  files = [f for f in os.listdir(MODIFIED_FRAMES_DIR) if isfile(join(MODIFIED_FRAMES_DIR, f))]
  #for sorting the file names properly
  # files.sort(key = lambda x: x[3:-4])
  files = sorted(files,key=lambda x: int(os.path.splitext(x)[0]))
  for i in range(len(files)):
      filename= MODIFIED_FRAMES_DIR + files[i]
      # print(filename)
      #reading each files
      img = cv2.imread(filename)
      height, width, layers = img.shape
      size = (width,height)
      
      #inserting the frames into an image array
      frame_array.append(img)
  
  out = cv2.VideoWriter(OUTPUT_FILE,cv2.VideoWriter_fourcc(*'DIVX'), FRAME_RATE, size)
  for i in range(len(frame_array)):
      # writing to a image array
      out.write(frame_array[i])
  out.release()
  print("Output video generated successfully...")

  # img_array = []
  # for filename in glob.glob(MODIFIED_FRAMES_DIR+'/*.jpg'):
  #   img = cv2.imread(filename)
  #   height, width, layers = img.shape
  #   size = (width,height)
  #   img_array.append(img)

  # height, width, layers = img_array[0].shape
  # size = (width,height)
  # out = cv2.VideoWriter('output.mov',cv2.VideoWriter_fourcc(*'DIVX'), 15, size) 
  # for i in range(len(img_array)):
  #     out.write(img_array[i])
  # out.release()

def read_reference():
  """
  Convert reference image into array
  """
  global reference
  reference = cv2.imread(REFERENCE)
  print(reference)
  print(reference.shape)

if __name__ == "__main__":
  # extract_frames()
  # create_video()
  read_reference()






#----------------------------------------------------------------
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
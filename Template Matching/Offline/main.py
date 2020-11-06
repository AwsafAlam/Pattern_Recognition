from glob import glob
import cv2
import numpy as np
import os
from os.path import isfile, join

INPUT_FILE = "input.mov"
OUTPUT_FILE = "output.mov"
OUTPUT_LOG = "output.txt"
REFERENCE = "reference.jpg"
FRAMES_DIR = "./frames/"
MODIFIED_FRAMES_DIR = "./out_frames/"
FRAME_RATE = 15
INF = 9999999999999

reference, frames, p = 0, 0, 0
Xi, Xj = 0,0
f = open(OUTPUT_LOG, "w")
f.write("Starting template matching")

template = cv2.imread(REFERENCE)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# w, h = template.shape[::-1]
w, h = template.shape

if not os.path.exists('out_frames'):
    os.makedirs('out_frames')


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
  return c


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

def calculate_distance(img_gray,row = 0, col = 0):
  """
  calculate the relative difference
  """
  global p
  f.write("Started matching [{},{}]\n".format(row, col))
  M1 = template
  M2 = img_gray[row:row+w,col: col+h]
  diff_mat = M1-M2
  M = np.absolute(diff_mat)
  final = M * M
  sum = np.sum(final, dtype = np.int64)
  # sum, diff = 0 , 0
  # for i in range(h):
  #   for j in range(w):
  #     diff = int(template[i][j]) - int(img_gray[row + i][col + j])
  #     sum = sum + pow(int(diff), 2)
  #     if sum != 0:
  #       # f.write(str(sum)+"\n")
  #       f.write("Diff {}: [{},{}] - SUM={} \n".format(diff,i,j,sum))
  #   f.write("\n\n-------------------\n\n")
  # f.write("--------------------------------\n\n")
  return int(sum)


def template_match_exhaustive(frame):
  """
  docstring
  """
  global template,w,h,p, Xi, Xj
  # Read and convert both images to grayscale
  test_img = MODIFIED_FRAMES_DIR + str(frame+1) + '.jpg'
  print(test_img)
  img_rgb = cv2.imread(test_img)
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

  test_width, test_height = img_gray.shape

  img_gray = np.array(img_gray)
  img_gray.astype(np.int64)

  template = np.array(template)
  template.astype(np.int64)

  # row_sums = template.sum(axis=1)
  # template = template / row_sums[:, np.newaxis]

  # row_sums = img_gray.sum(axis=1)
  # img_gray = img_gray / row_sums[:, np.newaxis]

  print(img_gray.shape, template.shape)
  print("({},{}) - ({},{})".format(test_width,test_height,w,h))
  print(str(test_height - h), str(test_height), str(h))

  min = INF
  centre_i, centre_j = 0,0
  # start = (Xi - p, Xi+p)
  # end = (Xj - p, Xj + p)
  if frame == 0:
    m_start, m_end = Xi,test_width - w + 1
    n_start, n_end = Xj,test_height - h + 1
  else:
    m_start, m_end = Xi - p, Xi+p
    n_start, n_end = Xj - p, Xj + p

  for i in range(m_start, m_end):
    for j in range(n_start, n_end):
      # calculating the mean distance
      dist = calculate_distance(img_gray, i,j)
      f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
      if(dist < min):
        print("Found min ({})--- [{},{}]\n".format(dist,i,j))
        min = dist
        centre_i = i
        centre_j = j
      
  print("done")
  print(centre_i,centre_j)
  if frame == 0:
    Xi = centre_i
    Xj = centre_j
  # cv2.rectangle(img_rgb, (centre_i,centre_j), (centre_i + h, centre_j + w), (0,0,255), 2)
  cv2.rectangle(img_rgb, (centre_j,centre_i), (centre_j + h, centre_i + w), (0,0,255), 2)
  
  # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
  # threshold = 0.9
  # loc = np.where( res >= threshold)

  # for pt in zip(*loc[::-1]):
  #   cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

  cv2.imwrite(test_img,img_rgb)


def read_reference():
  """
  Convert reference image into array
  """
  global reference
  reference = cv2.imread(REFERENCE)
  print(reference)
  print(reference.shape)


if __name__ == "__main__":
  print("Enter the value of p")
  p = int(input())
  frames = extract_frames()
  print("Total Frames: {}".format(frames))
  for i in range(frames-2):
    print("Running for frame {}".format(i))
    template_match_exhaustive(i)
  
  create_video()
  # read_reference()
  f.close()


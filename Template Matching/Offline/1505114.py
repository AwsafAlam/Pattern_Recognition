from glob import glob
import cv2
import numpy as np
import os
import math
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

def calculate_distance(img_gray, template, row = 0, col = 0):
  """
  calculate the relative difference
  """
  f.write("Started matching [{},{}]\n".format(row, col))
  w, h = template.shape

  M1 = template.astype(np.int64)
  M2 = img_gray[row:row+w,col: col+h]
  M2 = M2.astype(np.int64)
  diff_mat = M1 - M2
  M = np.absolute(diff_mat)
  final = M * M
  sum = np.sum(final, dtype = np.int64)

  return int(sum)

def template_match_2D_log(frame, p):
  """
  2D Log
  """
  print("Starting 2D log ...")
  global template,w,h, Xi, Xj
  times_searched = 0
  
  # Read and convert both images to grayscale
  test_img = MODIFIED_FRAMES_DIR + str(frame+1) + '.jpg'
  print(test_img)
  img_rgb = cv2.imread(test_img)
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

  test_width, test_height = img_gray.shape

  img_gray = np.array(img_gray)
  img_gray = img_gray.astype(np.int64)

  template = np.array(template)
  template = template.astype(np.int64)

  print(img_gray.shape, template.shape)
  print("({},{}) - ({},{})".format(test_width,test_height,w,h))
  print(str(test_height - h), str(test_height), str(h))
  print("Starting p = {}".format(p))
  min = INF
  centre_i, centre_j = 0,0
  

  if frame == 0:
    m_start, m_end = Xi,test_width - w + 1
    n_start, n_end = Xj,test_height - h + 1

    for i in range(m_start, m_end):
      for j in range(n_start, n_end):
        # calculating the mean distance
        dist = calculate_distance(img_gray,template, i,j)
        f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
        times_searched = times_searched + 1
        if(dist < min):
          print("Found min ({})--- [{},{}]\n".format(dist,i,j))
          min = dist
          centre_i = i
          centre_j = j
  else:
    
    while True:
      # find value of k and d
      k = np.ceil(np.log2(p))
      distance = int(np.power(2, k - 1))

      # traverse over the bounded box
      m_start, m_end = math.floor(Xi - distance), math.floor(Xi+ distance)
      n_start, n_end = math.floor(Xj - distance), math.floor(Xj + distance)

      for i in range(m_start, m_end):
        for j in range(n_start, n_end):
          # calculating the mean distance
          dist = calculate_distance(img_gray,template, i,j)
          f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
          times_searched = times_searched + 1
          if(dist < min):
            print("Found min ({})--- [{},{}]\n".format(dist,i,j))
            min = dist
            centre_i = i
            centre_j = j

      Xi = centre_i
      Xj = centre_j
      p = p / 2.0

      # when the distance is 1, break the while loop
      if (distance == 1):
          break
      
  print("done")
  print(centre_i,centre_j)
  if frame == 0:
    Xi = centre_i
    Xj = centre_j
  
  # cv2.rectangle(img_rgb, (centre_i,centre_j), (centre_i + h, centre_j + w), (0,0,255), 2)
  cv2.rectangle(img_rgb, (centre_j,centre_i), (centre_j + h, centre_i + w), (0,0,255), 2)
  
  cv2.imwrite(test_img,img_rgb)
  return times_searched


def template_match_Hierarchical(frame, p):
  """
  Hierarchical search
  """
  print("Starting Hierarchical...")
  global template,w,h, Xi, Xj
  times_searched = 0
  
  # Read and convert both images to grayscale
  test_img = MODIFIED_FRAMES_DIR + str(frame+1) + '.jpg'
  print(test_img)
  img_rgb = cv2.imread(test_img)
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

  test_width, test_height = img_gray.shape

  # error unsupported format --
  # img_gray = np.array(img_gray)
  # img_gray = img_gray.astype(np.int64)

  # template = np.array(template)
  # template = template.astype(np.int64)

  print(img_gray.shape, template.shape)
  print("({},{}) - ({},{})".format(test_width,test_height,w,h))
  print(str(test_height - h), str(test_height), str(h))
  print("Starting p = {}".format(p))
  min = INF
  centre_i, centre_j = 0,0
  

  if frame == 0:
  
    m_start, m_end = Xi,test_width - w + 1
    n_start, n_end = Xj,test_height - h + 1

    for i in range(m_start, m_end):
      for j in range(n_start, n_end):
        # calculating the mean distance
        dist = calculate_distance(img_gray, template, i,j)
        f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
        times_searched = times_searched + 1
        if(dist < min):
          print("Found min ({})--- [{},{}]\n".format(dist,i,j))
          min = dist
          centre_i = i
          centre_j = j
  else:
    templateLevels, testLevels = [], []

    # L0. original image
    templateLevels.append(template)
    testLevels.append(img_gray)

    # L1 -- downgrade
    templateLevels.append(cv2.pyrDown(template))
    testLevels.append(cv2.pyrDown(img_gray))

    # L2 -- downgrade
    templateLevels.append(cv2.pyrDown(templateLevels[1]))
    testLevels.append(cv2.pyrDown(testLevels[1]))
    
    # -----------------------------------------------------------------------------------------------
    #step 2
    p = p//4
    Xi = Xi//4
    Xj = Xj//4

    ref_width, ref_height = templateLevels[2].shape
    test_width, test_height= testLevels[2].shape
    
    pointsY = [Xi-p , Xi , Xi+p]
    pointsX = [Xj-p , Xj , Xj+p]
          
    for i in pointsY:
      for j in pointsX:
        if i < 0 or i > test_width-ref_width or j < 0 or j > test_height - ref_height:
            continue
        # calculating the mean distance
        dist = calculate_distance(testLevels[2], templateLevels[2], int(i),int(j))
        f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
        times_searched = times_searched + 1
        if(dist < min):
          print("Found min ({})--- [{},{}]\n".format(dist,i,j))
          min = dist
          centre_i = i
          centre_j = j

    Xi = centre_i
    Xj = centre_j

    # -----------------------------------------------------------------------------------------------
    # step 3
    p = 1
    Xi = int(2*Xi)
    Xj = int(2*Xj)

    
    min = INF
    centre_i, centre_j = 0,0
  
    ref_width, ref_height= templateLevels[1].shape
    test_width, test_height= testLevels[1].shape
    
    pointsY = [Xi-p , Xi , Xi+p]
    pointsX = [Xj-p , Xj , Xj+p]
          
    for i in pointsY:
      for j in pointsX:
        if i < 0 or i > test_width-ref_width or j < 0 or j > test_height - ref_height:
            continue
        # calculating the mean distance
        dist = calculate_distance(testLevels[1], templateLevels[1], int(i),int(j))
        f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
        times_searched = times_searched + 1
        if(dist < min):
          print("Found min ({})--- [{},{}]\n".format(dist,i,j))
          min = dist
          centre_i = i
          centre_j = j

    Xi = centre_i
    Xj = centre_j

    # -----------------------------------------------------------------------------------------------
    # step 4
    p = 1
    Xi = int (2*Xi)
    Xj = int (2*Xj)

    min = INF
    centre_i, centre_j = 0,0
  
    ref_width, ref_height= templateLevels[0].shape
    test_width, test_height= testLevels[0].shape
    
    pointsY = [Xi-p , Xi , Xi+p]
    pointsX = [Xj-p , Xj , Xj+p]
          
    for i in pointsY:
      for j in pointsX:
        if i < 0 or i > test_width-ref_width or j < 0 or j > test_height - ref_height:
            continue
        # calculating the mean distance
        dist = calculate_distance(testLevels[0], templateLevels[0], int(i),int(j))
        f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
        times_searched = times_searched + 1
        if(dist < min):
          print("Found min ({})--- [{},{}]\n".format(dist,i,j))
          min = dist
          centre_i = i
          centre_j = j

    Xi = centre_i
    Xj = centre_j
  
  print("done")
  print(centre_i,centre_j)
  if frame == 0:
    Xi = centre_i
    Xj = centre_j
  
  # cv2.rectangle(img_rgb, (centre_i,centre_j), (centre_i + h, centre_j + w), (0,0,255), 2)
  cv2.rectangle(img_rgb, (centre_j,centre_i), (centre_j + h, centre_i + w), (0,0,255), 2)
  
  cv2.imwrite(test_img,img_rgb)
  return times_searched



def template_match_exhaustive(frame):
  """
  Exhaustive search
  """
  global template,w,h,p, Xi, Xj
  times_searched = 0
  
  print("Starting Exhaustive search...")
  # Read and convert both images to grayscale
  test_img = MODIFIED_FRAMES_DIR + str(frame+1) + '.jpg'
  print(test_img)
  img_rgb = cv2.imread(test_img)
  img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

  test_width, test_height = img_gray.shape

  img_gray = np.array(img_gray)
  img_gray = img_gray.astype(np.int64)

  template = np.array(template)
  template = template.astype(np.int64)

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
    m_start, m_end = math.floor(Xi - p), math.floor(Xi+p)
    n_start, n_end = math.floor(Xj - p), math.floor(Xj + p)

  for i in range(m_start, m_end):
    for j in range(n_start, n_end):
      # calculating the mean distance
      dist = calculate_distance(img_gray,template,i,j)
      f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
      times_searched = times_searched + 1
      if(dist < min):
        print("Found min ({})--- [{},{}]\n".format(dist,i,j))
        min = dist
        centre_i = i
        centre_j = j
      
  print("done")
  print(centre_i,centre_j)
  # if frame == 0:
  Xi = centre_i
  Xj = centre_j
  
  # cv2.rectangle(img_rgb, (centre_i,centre_j), (centre_i + h, centre_j + w), (0,0,255), 2)
  cv2.rectangle(img_rgb, (centre_j,centre_i), (centre_j + h, centre_i + w), (0,0,255), 2)
  
  cv2.imwrite(test_img,img_rgb)
  return times_searched


def read_reference():
  """
  Convert reference image into array
  """
  global reference
  reference = cv2.imread(REFERENCE)
  print(reference)
  print(reference.shape)


if __name__ == "__main__":

  print("Enter search method:\n1. Exhaustive\n2. 2D Log\n3. Hierarchical\n")
  method = int(input())
  
  print("Enter the value of p")
  p = float(input())

  frames = extract_frames()
  print("Total Frames: {}".format(frames))
  performance = 0
  for i in range(frames - 2):
    print("Running for frame {}".format(i))
    if method == 3:
      performance = performance + template_match_Hierarchical(i, p)
    elif method == 2:
      performance = performance + template_match_2D_log(i , p)
    else:
      performance = performance + template_match_exhaustive(i)
  
  performance = performance / frames
  print("Performance: {}".format(performance))
  create_video()
  # read_reference()
  f.close()


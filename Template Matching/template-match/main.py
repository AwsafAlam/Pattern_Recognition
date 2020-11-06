import cv2
import numpy as np

f = open("output.txt", "w")
f.write("Starting template matching")

# Read and convert both images to grayscale
img_rgb = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('reference.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# w, h = template.shape[::-1]
w, h = template.shape

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

def calculate_distance(row = 0, col = 0):
  """
  calculate the relative difference
  """
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

print(str(calculate_distance(0,0)))
min = 9999999999999
centre_i, centre_j = 0,0
for i in range(test_width - w + 1):
  for j in range(test_height - h + 1):
    # calculating the mean distance
    dist = calculate_distance(i,j)
    f.write("Coord: ({},{}) - Dist={}".format(i,j,dist))
    if(dist < min):
      print("Found min ({})--- [{},{}]\n".format(dist,i,j))
      min = dist
      centre_i = i
      centre_j = j
    
print("done")
print(centre_i,centre_j)

# centre_i,centre_j = 150, 240
# cv2.rectangle(img_rgb, (centre_j,centre_i), (centre_j + h, centre_i + w), (0,0,255), 2)
cv2.rectangle(img_rgb, (centre_i,centre_j), (centre_i + h, centre_j + w), (0,0,255), 2)

cv2.imwrite('final.jpg',img_rgb)

f.close()

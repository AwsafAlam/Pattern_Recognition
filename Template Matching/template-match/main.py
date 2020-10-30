import cv2
import numpy as np

f = open("output.txt", "w")
f.write("Starting template matching")

img_rgb = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('reference.jpg',0)
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

test_width, test_height = img_gray.shape

print(img_gray.shape, template.shape)

x_axis, y_axis = [], []


def calculate_distance(row = 0, col = 0):
  """
  calculate the relative difference
  """
  f.write("Started matching [{},{}]\n".format(row, col))
  sum, diff = 0 , 0
  for i in range(h):
    for j in range(w):
      diff = int(template[i][j]) - int(img_gray[row + i][col + j])
      sum = sum + pow(int(diff), 2)
      if diff != 0:
        f.write("Diff {}: [{},{}] ".format(diff,i,j))
    f.write("\n\n-------------------\n\n")
  f.write("--------------------------------\n\n")
  return int(sum)

f.write(str(calculate_distance(0,0)))

for i in range(test_height - h):
  for j in range(test_width - w):
    # calculating the mean distance
    min = calculate_distance(i,j)
    # if int(img_gray[i][j]) != 255:
    #   # print(img_gray[i][j])
    #   x_axis.append(i)
    #   y_axis.append(j)
    #   # print("Coord: ({},{})".format(i,j))
      

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where( res >= threshold)
print(loc)

for pt in zip(*loc[::-1]):
  cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('final.jpg',img_rgb)

f.close()

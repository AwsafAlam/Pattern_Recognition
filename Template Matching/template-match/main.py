import cv2
import numpy as np

img_rgb = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('reference.jpg',0)
w, h = template.shape[::-1]

test_width, test_height = img_gray.shape

print(img_gray.shape)
x_axis, y_axis = [], []
for i in range(test_width):
  # print(len(img_gray[i]))
  for j in range(test_height):
    if int(img_gray[i][j]) != 255:
      # print(img_gray[i][j])
      x_axis.append(i)
      y_axis.append(j)
      print("Coord: ({},{})".format(i,j))
      
x_avg = sum(x_axis)/len(x_axis)
y_avg = sum(y_axis)/len(y_axis)
print(x_avg,y_avg)
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.9
loc = np.where( res >= threshold)
print(loc)
print(w,h)
for pt in zip(*loc[::-1]):
    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.rectangle(img_rgb, pt, (int(x_avg) + w, int(y_avg) + h), (0,0,255), 2)

cv2.imwrite('final.jpg',img_rgb)
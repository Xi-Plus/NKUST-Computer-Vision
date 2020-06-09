# https://blog.csdn.net/shenglong456/java/article/details/71174175
import cv2
import numpy

img = cv2.imread('img.jpg', 1)
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if imgHSV[i, j, 0] < 80 and imgHSV[i, j, 0] > 50:
            img[i, j] = [255, 255, 255]
cv2.imwrite('out.jpg', img)

import numpy as np
import cv2
import glob

# From https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

fname = 'camera1/train/9.jpg'
SIZE = (9, 6)

img = cv2.imread(fname)
print(fname, img.shape)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((SIZE[1] * SIZE[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:SIZE[0], 0:SIZE[1]].T.reshape(-1, 2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (SIZE[0], SIZE[1]), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
print(ret)
if ret == True:
    # If found, add object points, image points (after refining them)
    # print(objp)
    objpoints.append(objp)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners)
    # print(corners)
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (SIZE[0], SIZE[1]), corners2, ret)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('camera1/9-draw.jpg', img)

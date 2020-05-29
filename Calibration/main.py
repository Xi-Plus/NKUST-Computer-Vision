import numpy as np
import cv2
import glob

# From https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

trains = [
    # ('camera1/train/1.jpg', (7, 6)),
    # ('camera1/train/3.jpg', (9, 6)),
    ('camera1/train/5.jpg', (9, 6)),
    ('camera1/train/7.jpg', (9, 6)),
    ('camera1/train/8.jpg', (9, 6)),
    ('camera1/train/9.jpg', (9, 6)),
    # ('camera1/train/10.jpg', (9, 6)),
    # ('camera1/train/11.jpg', (9, 6)),
    # ('camera1/train/12.jpg', (9, 6)),
    # ('camera1/train/13.jpg', (9, 6)),
    # ('camera1/train/14.jpg', (9, 6)),
    # ('camera1/train/15.jpg', (9, 6)),
    # ('camera1/train/16.jpg', (9, 6)),
]
test = 'camera1/train/3.jpg'
test_out = 'camera1/3-out.jpg'

for train in trains:
    fname = train[0]
    SIZE = train[1]

    img = cv2.imread(fname)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((SIZE[1] * SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:SIZE[0], 0:SIZE[1]].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (SIZE[0], SIZE[1]), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
    print(fname, SIZE, img.shape, ret)
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
        # cv2.imwrite('camera1/out-draw.jpg', img)
# print(objpoints)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('ret', ret)
print('mtx', mtx)
print('dist', dist)
print('rvecs')
print(rvecs)
print('tvecs', tvecs)

img = cv2.imread(test)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x, y, w, h = roi
cv2.imwrite(test_out, dst)

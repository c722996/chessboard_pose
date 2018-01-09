#! /usr/bin/env python
import numpy as np
import cv2
import glob
import math

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((4*5,3), np.float32)
objp[:,:2] = np.mgrid[0:5,0:4].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('/home/kuo/chessboard_pose/data/calibrationdata_camera_1/*.jpg')
for fname in images:
    # print "++++"
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5,4), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        # print "+++++"
        objpoints.append(objp)
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (5,4), corners2, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        mean_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        print( "total error: {}".format(mean_error/len(objpoints)) )

       
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)



        # x = cv2.solvePnP(objp, corners2, mtx, dist)
        # print x
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)

        roll, pitch, yaw = rvecs
        pitch = math.pi/2 - pitch
        print pitch
        roll = round(math.degrees(roll), 3)
        pitch = round(math.degrees(pitch), 3)
        yaw = round(math.degrees(yaw), 3)

        print "\nangle(Deg):\n",roll, pitch, yaw
        print "\nrvec:\n", rvecs 

        print "\ntvec:\n", tvecs 


        cv2.imshow('img', img)
        cv2.waitKey(15000)
cv2.destroyAllWindows()
#!/usr/bin/env python

"""
Camera Calibration script taken from the Tinker-Twins, Camera-Calibration repo
https://github.com/Tinker-Twins/Camera-Calibration/blob/main/Camera%20Calibration.ipynb

Orignal Authors: Tanmay Samak and Chinmay Samak
"""

######################################################################

import numpy as np
import cv2
import glob
import pickle
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

######################################################################
ROW = 7 - 1
COL = 10 - 1
PLOT_CALIBRATIONS = False

cell_size = 0.021 #21cm, Size of the square cell (should not affect the result)
objp = np.zeros((ROW*COL,3), np.float32) # Using a 6x9 grid image for calibration
objp[:,:2] = np.mgrid[0:COL,0:ROW].T.reshape(-1,2)*cell_size # Grid points

# Arrays to store object points and image points from all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Make a list of calibration images
images = glob.glob('/root/AprilTag/media/calib_img/*.jpg')
print("calibrating according to", len(images), "images")
img1 = mpimg.imread(images[0])
img_size = (img1.shape[1], img1.shape[0]) # Get the image dimensions

# Step through the list and search for grid corners
for image in images:
    img = mpimg.imread(image)
    gray_img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # Find the grid corners
    ret, corners = cv2.findChessboardCorners(gray_img, (COL,ROW),None)

    # If found, append object points and image points to the respective arrays
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.namedWindow('Corner Detection', cv2.WINDOW_NORMAL)
        img = cv2.drawChessboardCorners(img, (COL,ROW), corners, ret)
        # Plot in a new window
        if PLOT_CALIBRATIONS:
            cv2.imshow('Corner Detection',img)
            cv2.waitKey(500) # Viewing delay in milliseconds

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None) # Calibrate the camera

# Print the camera calibration result
print('Distortion Coefficients:')
print('dist = \n' + str(dist))
print('')

print('Camera Intrinsic Parameters:')
print('mtx = \n' + str(mtx))
print('')

print('Camera Extrinsic Parameters:')
print('rvecs = \n' + str(rvecs))
print('')
print('tvecs = \n' + str(tvecs))

# Save the camera calibration result for later use
camera_pickle = {}
camera_pickle["mtx"] = mtx
camera_pickle["dist"] = dist
camera_pickle["rvecs"] = rvecs
camera_pickle["tvecs"] = tvecs
pickle.dump(camera_pickle, open("camera_parameters.p", "wb"))

undistorted_image = cv2.undistort(img1, mtx, dist, None, mtx) # Generate the undistorted image

# Plot inline
# %matplotlib inline
# View the results side-by-side
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# f.tight_layout()
# ax1.imshow(img1)
# ax1.set_title('Original Image', fontsize=20)
# ax2.imshow(undistorted_image)
# ax2.set_title('Undistorted Image', fontsize=20)
# plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)
# plt.show()
print("original image")
cv2.imshow('Corner Detection',img1)
cv2.waitKey(500) # Viewing delay in milliseconds
cv2.imwrite('original.png',img1)
print("undistorted image")
cv2.imshow('Corner Detection',undistorted_image)
cv2.waitKey(500) # Viewing delay in milliseconds
cv2.imwrite('undistorted.png',undistorted_image)




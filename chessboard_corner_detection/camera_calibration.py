#!/usr/bin/env python

"""

	Camera Calibration

	1.) Create 4 aruco markers images for chessboard corner detection.

								m0	 m1
								   B
								m3   m2

						  (marker arrangment)

			where,

				mi ~ Aruco marker with associate id "i"
				B  ~ Chessboard

	2.) Create json settings file.


	TODO: use marker images to calibrate camera in single step

"""

__author__ = "l.j. Brown"
__version__ = "1.0.2"

#
#
#									imports	
#
#

# internal
import logging

# external
import pandas as pd
import numpy as np
import cv2
import glob

#
#
#									Settings
#
#

# camera device identifier to be calibrated
CAMERA_DEVICE = "laptop"

ARUCO_MARKER_SETTINGS_FNAME = "aruco_markers/aruc_markers_settings.json" 		# Do not change
aruco_markers_ftemplate = "aruco_markers/aruco_%s.png"

calibration_images_directory = "calibration_images/%s/" % CAMERA_DEVICE	# relative
calibration_images_ftemplate = calibration_images_directory + '%s'
camera_parameters_file = "calibration_data/%s/calibration.npz"  % CAMERA_DEVICE
output_images_ftemplate = "output_images/%s/" % CAMERA_DEVICE + "%s.png"

nrow_corners, ncol_corners = 7, 7 	# 8x8 chessboard
bh, bw = 7.75, 7.75 	# inches, chessboard hieght and width measurments
aruco_marker_length = bh*nrow_corners*100

#
#
#									Logging
#
#

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
#
#									Methods
#
#

#
#	Calibration
#

def calibrate_camera(calibration_images_ftemplate):

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((nrow_corners*ncol_corners,3), np.float32)
	objp[:,:2] = np.mgrid[0:nrow_corners,0:ncol_corners].T.reshape(-1,2)

	# using square size
	objpoints = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	objpoints[:,:,:2] = np.mgrid[0:aruco_marker_length:7j,  0:aruco_marker_length:7j].T.reshape(-1,1,2)
	objp = objpoints

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	# registered image paths
	registered_image_paths = []

	images = glob.glob(calibration_images_ftemplate % '*')
	counter = 0
	print(images)
	for fname in images:

		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# resize for speed - by .. of axes
		#img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
		#gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5) 

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (nrow_corners,nrow_corners),None)

		print(ret)
		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)

			# save list of discovered image paths
			registered_image_paths.append(fname)

			# Draw
			img = cv2.drawChessboardCorners(img, (nrow_corners,nrow_corners), corners2,ret)
	
			# write
			fpath = output_images_ftemplate % counter
			counter += 1
			cv2.imwrite(fpath, img)

			# display the corners
			cv2.imshow('img',img)
			cv2.waitKey(500)

	cv2.destroyAllWindows()

	# camera matrix, distortion coefficients, rotation and translation vectors
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	np.savez(camera_parameters_file, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


#
#
#								Program
#
#

if __name__ == '__main__':

	print("running")
	
	# calibrate camera and save intrinsic parameters
	calibrate_camera(calibration_images_ftemplate)

"""

#
#	Settings
#
CALIBRATE = False
TEST_UNDISTORT = False
POSE_ESTIMATION = False
BIRDS_EYE_WARP = True




#
#	Board Dimensions
#

square_length = 0.02423 			# meters measured with calimeter
board_length = 0.19384				# meters - 8 * square_length
# aruco_marker_length	= 0.14538		# meters - 6 * square_length

#square_length = 0.02460625 		# meters, "31/32 inches"
#board_length = 0.19685 			# meters, "7.75 inches"
#aruco_marker_length = 0.1476375 	# meters, 6*square_length
#chess_board_dimensions = (6,6) 		# 6,6?
chess_board_dimensions = (7,7) 		# 6,6?
#aruco_markers = (6,6)
nrow_corners, ncol_corners = 7, 7 	# 8 * 8 chess board

#aruco_marker_length	= square_length * 100 #* nrow_corners * 100		# meters - 7 * square_length
aruco_marker_length	= square_length * nrow_corners * 100		# mm - 7 * square_length


def draw(img, corners, imgpts):
	line_width = 3
	corner = tuple(corners[6].ravel())
	img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), line_width)
	img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), line_width)
	img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), line_width)
	return img
"""

"""
#
#	Calibration
#

if CALIBRATE:

	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((nrow_corners*ncol_corners,3), np.float32)
	objp[:,:2] = np.mgrid[0:nrow_corners,0:ncol_corners].T.reshape(-1,2)

	# using square size
	objpoints = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	objpoints[:,:,:2] = np.mgrid[0:aruco_marker_length:7j,  0:aruco_marker_length:7j].T.reshape(-1,1,2)
	objp = objpoints

	
	# using square size
	#objpoints = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	#objpoints[:,:,:2] = np.mgrid[0:aruco_marker_length:6j,  0:aruco_marker_length:6j].T.reshape(-1,1,2)
	#objp = objpoints
	

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	# registered image paths
	registered_image_paths = []

	images = glob.glob(calibration_images_ftemplate % '*')
	print(images)
	for fname in images:
		print(fname)
		img = cv2.imread(fname)

		# resize for speed
		
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (nrow_corners,nrow_corners),None)

		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)

			# save list of discovered image paths
			registered_image_paths.append(fname)

			# Draw and display the corners
			img = cv2.drawChessboardCorners(img, (nrow_corners,nrow_corners), corners2,ret)

			#cv2.imshow('img',img)
			#cv2.waitKey(500)

	#cv2.destroyAllWindows()

	# camera matrix, distortion coefficients, rotation and translation vectors
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	np.savez(camera_parameters_file, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

"""






# python 3

"""
		camera calibration

"""

import pandas as pd
import numpy as np
import cv2
import glob

#
#	Settings
#
CALIBRATE = False
TEST_UNDISTORT = False
POSE_ESTIMATION = False
BIRDS_EYE_WARP = True

#
# 	Paths
#

calibration_images_directory = "iphone_calibration_images/" 	# relative
#calibration_images_template = calibration_images_directory + 'image_%s.png'
calibration_images_template = calibration_images_directory + '%s'
camera_parameters_file = "calibration_data/calibration.npz"
test_undistort_image_output_template = "output_images/calibresult_iphone_%s.png"
test_images_file_template = "test_images_iphone/%s"
test_pose_image_output_template = "output_images/pose_iphone_%s.png"
test_box_image_output_template = "output_images/box_iphone_%s.png"
birds_eye_template = "output_images/birds_eye_iphone_%s.png"


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

	"""
	# using square size
	objpoints = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	objpoints[:,:,:2] = np.mgrid[0:aruco_marker_length:6j,  0:aruco_marker_length:6j].T.reshape(-1,1,2)
	objp = objpoints
	"""

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	# registered image paths
	registered_image_paths = []

	images = glob.glob(calibration_images_template % '*')
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
			#img = cv2.drawChessboardCorners(img, (nrow_corners,nrow_corners), corners2,ret)
			#cv2.imshow('img',img)
			#cv2.waitKey(500)

	#cv2.destroyAllWindows()

	# camera matrix, distortion coefficients, rotation and translation vectors
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	np.savez(camera_parameters_file, ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


#
#	Test undistort
#

if TEST_UNDISTORT:
	# Load previously saved data
	with np.load(camera_parameters_file) as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

	images = glob.glob(test_images_file_template % '*')
	image_counter = 1
	for fname in images:
		img = cv2.imread(fname)

		# resize 1024,768 - ensure resize to calibration phot dimsensions
		# img = cv2.resize(img, (1024, 768))

		h,  w = img.shape[:2]
		newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

		# undistort
		dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
		cv2.imwrite(test_undistort_image_output_template % image_counter, dst)
		image_counter += 1


#
#	Pose Estimation - Draw Axis
#

if POSE_ESTIMATION:

	# Load previously saved data
	with np.load(camera_parameters_file) as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	#
	# for solvePnPRansac
	#

	objp = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	objp[:,:,:2] = np.mgrid[0:nrow_corners,  0:ncol_corners].T.reshape(-1,1,2)

	# using square size
	objpoints = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	objpoints[:,:,:2] = np.mgrid[0:aruco_marker_length:7j,  0:aruco_marker_length:7j].T.reshape(-1,1,2)
	objp = objpoints
	print(objp)


	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)


	#for fname in glob.glob('left*.jpg'):
	images = glob.glob(test_images_file_template % '*')
	#images = glob.glob(calibration_images_template % '*')
	image_counter = 1
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, chess_board_dimensions, None)

		if (ret == True):

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

			# Find the rotation and translation vectors.
			_, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
			#rvecs, tvecs, inliers = cv2.solvePnP(objp, corners2, mtx, dist)


			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

			img = draw(img,corners2,imgpts)
			#cv2.imshow('img',img)
			#k = cv2.waitKey(0) & 0xff
			
			cv2.imwrite(test_pose_image_output_template % image_counter, img)
			image_counter += 1
			
			#if k == 's':
			#	cv2.imwrite(fname[:6]+'.png', img)

	cv2.destroyAllWindows()

"""
				Perspective Transformations on Image Matrices
				definitions:
						* Top Left Origin Coordinate System With positive y axis flipped - S (standard)
						* Image Center Origin Coordinate System - C (centered)
				use homogeneous coordinate systems to translate Matricies
				Transformation_S2C - T_sc
				|	1 	 0	 -img_width/2  |
				| 	0	-1	 img_height/2  | 
				|	0	 0		  1		   |
				Transformation_C2S - T_cs
				|	1 	 0	 img_width/2   |
				| 	0	-1	 img_height/2  | 
				|	0	 0		  1		   |
				T_sc use:
				|	1 	 0	 -img_width/2  | | s_x |     | c_x |
				| 	0	-1	 img_height/2  | | s_y |  =  | c_y |
				|	0	 0		  1		   | |  1  |     |  1  |
 precieved radius
 - pr
 actual radius
 -ar
 
"""

"""
#
# Transformation Matrices
#

t_x, t_y = (img_width/2), (img_height/2)

# standard to center coordinate system transofrmation matrix
T_sc = np.array([[1,0,-t_x],[0,-1,t_y],[0,0,1]])

# center to standard coordinate system transofrmation matrix
T_cs = np.array([[1,0,t_x],[0,-1,t_y],[0,0,1]])

"""


def to_homogeneous(np_array):
	# [x, y] -> [x, y, 1]
	# add extra dimension slice of 0's, make final elemnt 1
	lshape = list(np_array.shape)
	lshape[0]=1
	new_row = np.zeros(tuple(lshape), dtype=int)
	np_array_add0s = np.concatenate((np_array, new_row), axis=0)
	indexer = np.array(np_array_add0s.shape)
	indexer = tuple(np.subtract(indexer, 1))
	np_array_add0s[indexer] = 1
	return np_array_add0s

def from_homogeneous(np_array):
	# [x, y, 1] -> [x, y]
	# remove extra dimension slice
	return np.delete(np_array, (-1), axis=0)


def standard_to_centered(standard_vector, img_height, img_width, integers=False):
	# takes 1d numpy array
	# [s_x, s_y] -> [c_x, c_y]

	# standard to center coordinate system transofrmation matrix
	t_x, t_y = (img_width/2), (img_height/2)
	T_sc = np.array([[1,0,-t_x],[0,-1,t_y],[0,0,1]])	# transformation matrix

	s_vh = to_homogeneous(standard_vector)		# [s_x, s_y, 1]
	c_vh = T_sc.dot(s_vh)						# [c_x, c_y, 1]
	centered_vector = from_homogeneous(c_vh)	# [c_x, c_y]

	if integers:
		centered_vector = centered_vector.astype(int)

	return centered_vector

def centered_to_standard(centered_vector, img_height, img_width, integers=False):
	# takes 1d numpy array
	# [s_x, s_y] -> [c_x, c_y]

	# center to standard coordinate system transofrmation matrix
	t_x, t_y = (img_width/2), (img_height/2)
	T_cs = np.array([[1,0,t_x],[0,-1,t_y],[0,0,1]])		# transformation matrix

	c_vh = to_homogeneous(centered_vector)					# [c_x, c_y, 1]
	s_vh = T_cs.dot(c_vh)									# [s_x, s_y, 1]
	standard_vector = from_homogeneous(s_vh)				# [s_x, s_y]

	if integers:
		standard_vector = standard_vector.astype(int)

	return standard_vector


#
#	Birds Eye Warp
#

if BIRDS_EYE_WARP:

	# Load previously saved data
	with np.load(camera_parameters_file) as X:
		mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]


	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	#
	# for solvePnPRansac
	#

	# using square size
	objp = np.zeros((nrow_corners*ncol_corners,1,3), np.float32)
	objp[:,:,:2] = np.mgrid[0:aruco_marker_length:7j,  0:aruco_marker_length:7j].T.reshape(-1,1,2)
	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

	#images = glob.glob(test_images_file_template % '*')
	images = glob.glob(calibration_images_template % '*')
	image_counter = 1
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, chess_board_dimensions, None)

		if (ret == True):

			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

			# Find the rotation and translation vectors.
			_, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
			#rvecs, tvecs, inliers = cv2.solvePnP(objp, corners2, mtx, dist)


			# project 3D points to image plane
			imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

			#print(imgpts)
			#print(len(corners2))
			#print(tuple(imgpts[0].ravel()))
			#print(tuple(corners2[0].ravel()))

			top_left_aruco_point = tuple(corners[0].ravel())
			top_right_aruco_point = tuple(corners[6].ravel())
			bottom_left_aruco_point = tuple(corners[42].ravel())
			bottom_right_aruco_point = tuple(corners[-1].ravel())

			#img = cv2.line(img,top_left_aruco_point,top_right_aruco_point, (255,0,0), 5)
			#img = cv2.line(img,top_left_aruco_point,bottom_left_aruco_point, (255,0,0), 5)
			#img = cv2.line(img,top_right_aruco_point,bottom_right_aruco_point, (255,0,0), 5)
			#img = cv2.line(img,bottom_left_aruco_point, bottom_right_aruco_point, (255,0,0), 5)

			# get new positions
			h,w = gray.shape
			be_square_length = int(min([h,w])/8) # birdseye square length
			
			# 4 squares above and bellow 0 y line (middle y line) - point is third
			# center coordinates
			be_y1_c = be_square_length*3	
			be_y2_c = -be_y1_c
			be_x1_c = -be_square_length*3
			be_x2_c = -be_x1_c

			be_top_left_c = np.array([be_x1_c, be_y1_c])
			be_top_right_c = np.array([be_x2_c, be_y1_c])
			be_bottom_left_c = np.array([be_x1_c, be_y2_c])
			be_bottom_right_c = np.array([be_x2_c, be_y2_c])

			be_top_left_s = centered_to_standard(be_top_left_c, h, w) #, True)
			be_top_right_s = centered_to_standard(be_top_right_c, h, w) #, True)
			be_bottom_left_s = centered_to_standard(be_bottom_left_c, h, w) #, True)
			be_bottom_right_s = centered_to_standard(be_bottom_right_c, h, w) #, True)

			"""
			# for visulizing
			be_crop_y1_c = be_square_length*4
			be_crop_y2_c = -be_crop_y1_c
			be_crop_x1_c = -be_square_length*4
			be_crop_x2_c = -be_crop_x1_c

			be_crop_top_left_c = np.array([be_crop_x1_c, be_crop_y1_c])
			be_crop_top_right_c = np.array([be_crop_x2_c, be_crop_y1_c])
			be_crop_bottom_left_c = np.array([be_crop_x1_c, be_crop_y2_c])
			be_crop_bottom_right_c = np.array([be_crop_x2_c, be_crop_y2_c])

			be_crop_top_left_s = centered_to_standard(be_crop_top_left_c, h, w) #, True)
			be_crop_top_right_s = centered_to_standard(be_crop_top_right_c, h, w) #, True)
			be_crop_bottom_left_s = centered_to_standard(be_crop_bottom_left_c, h, w) #, True)
			be_crop_bottom_right_s = centered_to_standard(be_crop_bottom_right_c, h, w) #, True)

			#be_top_left_s = tuple(be_top_left_c.astype(int))
			#be_top_right_s = tuple(be_top_right_c.astype(int))
			#be_bottom_left_s = tuple(be_bottom_left_c.astype(int))
			#be_bottom_right_s = tuple(be_bottom_right_c.astype(int))
			#tuple(map(tuple, arr))
			"""


			original_points = np.float32([list(top_left_aruco_point), list(top_right_aruco_point), list(bottom_left_aruco_point)]) #, list(bottom_right_aruco_point)])
			birds_eye_points = np.float32([be_top_left_s, be_top_right_s, be_bottom_left_s]) #, be_bottom_right_s])

			M = cv2.getAffineTransform(original_points,birds_eye_points)
			dst = cv2.warpAffine(img,M,(w,h))

			"""
			# crop region of intrest
			print(be_crop_top_left_s[0])
			print(be_crop_top_right_s[0])
			dst = dst[int(be_crop_top_left_s[1]):int(be_crop_bottom_right_s[1]), int(be_crop_top_left_s[0]):int(be_crop_top_right_s[0])]
			"""

			cv2.imwrite(birds_eye_template % image_counter, dst)
			image_counter += 1

			"""
			# draw new box
			be_top_left_s = tuple(centered_to_standard(be_top_left_c, h, w, True))
			be_top_right_s = tuple(centered_to_standard(be_top_right_c, h, w, True))
			be_bottom_left_s = tuple(centered_to_standard(be_bottom_left_c, h, w, True))
			be_bottom_right_s = tuple(centered_to_standard(be_bottom_right_c, h, w, True))
			img = cv2.line(img,be_top_left_s,be_top_right_s, (255,0,0), 5)
			img = cv2.line(img,be_top_left_s,be_bottom_left_s, (255,0,0), 5)
			img = cv2.line(img,be_top_right_s,be_bottom_right_s, (255,0,0), 5)
			img = cv2.line(img,be_bottom_left_s, be_bottom_right_s, (255,0,0), 5)
			"""







			#be_square_height = int(h/8)
			#birds_eye_square_xs = np.

			#img = draw(img,corners2,imgpts)
			#cv2.imshow('img',img)
			#k = cv2.waitKey(0) & 0xff
			
			#cv2.imwrite(test_box_image_output_template % image_counter, img)
			#image_counter += 1
			
			#if k == 's':
			#	cv2.imwrite(fname[:6]+'.png', img)

	cv2.destroyAllWindows()






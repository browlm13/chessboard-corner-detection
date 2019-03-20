#!/usr/bin/env python

"""
	
		Aruco Marker Detection.

		Performs aruco marker corner detection on images and 
		can return the 2D pixel locations in the following formats:
		
			- (3x4x2) Matrix of 2D coordinates of aruco marker corners, A.
			- (12x2) Matrix of 2D coordinates of aruco marker corners, aruco_image_points.

		* Format of A is described by diagram 4 bellow.
		* Format of aruco_image_points is described by diagram 5 bellow.

		Required measurements:

			- l or marker side length (inches), descirbed in Diagram 3
			- a, b, c measurents descirbed in Diagram 3

		Diagram 1:
		--------------------------------------------------------------------------

									m0	 m1
									   
									     m2

							  (marker arrangment)

			where,

				mi ~ Aruco marker with associate id "i"

				* Settings stored in json aruco markers settings file
				* Setting file and images created in "aruco_marker_creation.py"

		--------------------------------------------------------------------------

		Bellow is the aruco marker arrangment above with each marker expanded 
		to reveal its four corners to be detected.

		Diagram 2:
		--------------------------------------------------------------------------


								m00 m01 	m10 m11
								m03 m02 	m13 m12

								        	m20 m21
								        	m23 m22

				(marker arrangment expanded to reveal all corners to detect)

			where,

				mij ~ 2d pixel coordinate [x,y] of corner index "j" of aruco marker with id "i". 

				* Coordinates of corners use standard image axis - "(0,0) is at the top left of the image"

		--------------------------------------------------------------------------
		Diagram 3:
		--------------------------------------------------------------------------
							  

							  =======================
							  Real World Measurements
							  =======================


							----------------------------
							Marker Dimension Measurments:
							----------------------------

										l
									|--------|
								    mi0 	mi1
	

								    mi3 	mi2

					where,

						l ~ width of square marker 
							in inches, "marker side length"

							

							 -----------------------------
							 Marker Arrangment and Spacing:
							 -----------------------------


								|--------------------|			
											a
								m00 --------------- m11  ---
							  /.     .		  theta  |	  |	
								  .     .		  '- |	  |
									 .     .		 |  b |
										. c   .		 |	  |
										   .	 .   |    |
											  .		m22  ---
												/	  	      
								
					where,

						m00 ~ top left corner of marker 0
						m11 ~ top right corner of marker 1
						m22 ~ bottom right corner of marker 2

						a ~ distance between far corners of markers 0 
							and 1 in inches
						b ~ distance between far corners of markers 1 
							and 2 in inches
						c ~ distance between far corners of markers 2 
							and 0 in inches

						theta ~ right angle 


			List of required measurements:

				l, a, b, c


		--------------------------------------------------------------------------

		The corner locations can be stored in a (3x4x2) numpy array, A.

		Diagram 4:
		--------------------------------------------------------------------------


									A = 

					[ [[m00_x, m00_y], ..., [m03_x, m03_y]],
					 	... , 
					[[m20_x, m30_y], ..., [m23_x, m23_y]] ]


				  (Aruco Marker Corner 2D Image Locations Matrix)

			where,

				A ~ Matrix of 2D image coordinates of aruco marker corners
				mij_x ~ x coordinate of aruco marker "i"'s "j"th corner.
				mij_y ~ y coordinate of aruco marker "i"'s "j"th corner.

			
				* (3x4x2) ~ (marker_id, corner_id, 2D_image_coordinate)
				* Coordinates of corners use standard image axis - "(0,0) is at the top left of the image"
		
		--------------------------------------------------------------------------
		
		The corner locations can be stored in a (12x2) numpy array, A.

		Diagram 5:
		--------------------------------------------------------------------------
					
					  0 >-----1            4 >-----5
							  |					   |
					  3 <-----2			   7 <-----6


						
					  		               8 >-----9
							  					   |
					  	  				   11 <----10

			(aruco corner 2D location array index clockwise ordering)
						
				
								aruco_image_points = 

		[ [m00_x, m00_y], ..., [m03_x, m03_y], ... , [m20_x, m20_y], ..., [m23_x, m23_y] ]


				  (Aruco Marker Corner 2D Image Locations Matrix in image point representation)

			where,

				A ~ Matrix of 2D image coordinates of aruco marker corners
				mij_x ~ x coordinate of aruco marker "i"'s "j"th corner.
				mij_y ~ y coordinate of aruco marker "i"'s "j"th corner.

			
				* (12x2) ~ (marker_id, corner_id, 2D_image_coordinate)
				* Coordinates of corners use standard image axis - "(0,0) is at the top left of the image"
		
		-------------------------------------------------------------------

"""

__author__ = "l.j. Brown"
__version__ = "1.0.4"

#
#
#									imports	
#
#

# internal
import os
import logging
import random
import math
import copy

# external
import glob
import pandas as pd
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy import stats

# my lib
import aruco_marker_detection as marker_detector

#
#
#									Settings
#
#

ARUCO_MARKER_SETTINGS_FNAME = "aruco_markers/aruc_markers_settings.json" 		# Do not change

input_ftemplate = "test_aruco_images_input/%s"
output_ftemplate = "test_aruco_images_output/%s"

#
#
#	Generate aruco marker 3D model points, or "world points" of corners, from required measurements: l, a, b, and c.
#
#

def squared_error(true, measured):
	return math.sqrt((true - measured)**2)

def aruco_measurements_to_model_points(l, a, b, c, tol):
	# all measurements in inches
	# c just use to ensure angle is within tolerence around 90 degrees
	c_ideal = math.sqrt(a**2 + b**2)
	assert squared_error(c_ideal, c) < tol

	#
	# 	Aruco marker 3D model points. "world points" of corners
	#
	aruco_model_points = np.array([
							# 	x,  	y,   	z= 0.0

							(	0.0, 	0.0, 	0.0		),             # m00
							(	l, 		0.0, 	0.0		),			   # m01
							(	l, 		l, 		0.0		),             # m02
							(	0.0, 	l,		0.0		),			   # m03

							(	a-l, 	0.0, 	0.0		),         	   # m10
							(	a, 		0.0, 	0.0		),			   # m11
							(	a, 		l, 		0.0		),             # m12
							(	a-l, 	l, 		0.0		),			   # m13

							(	a-l, 	b-l, 	0.0		),     		   # m20
							(	a, 		b-l, 	0.0		),			   # m21
							(	a, 		b, 		0.0		),             # m22
							(	a-l, 	b, 		0.0		)			   # m23

						])

	return aruco_model_points


def draw_box(B, input_image, color=None, lineThickness = 4):
	# B - 4x2 matrix following clockwise convention
	np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

	if color is None:
		color = (0,255,0)

	for i in range(4):
		start_coor = B[i,:]
		if i == 3:
			end_coor = B[0,:]
		else:
			end_coor = B[i+1,:]

		# draw line on image
		cv2.line(input_image, np_2_tupple_int(start_coor), np_2_tupple_int(end_coor), color, lineThickness)

	return input_image

# initialize the list of image chessboard corner points points
chessboard_corners_image_points = []

def select_chessboard_image_points(event, x, y, flags, param):
	# grab references to the global variables
	global chessboard_corners_image_points
 
	# if the left mouse button was clicked, record the 
	# (x, y) coordinates 

	if event == cv2.EVENT_LBUTTONDOWN:

		if len(chessboard_corners_image_points) == 4:
			chessboard_corners_image_points = []

		if len(chessboard_corners_image_points) <= 4:
			chessboard_corners_image_points.append((x,y))


def user_input_chessboard_corners(image):

	global chessboard_corners_image_points

	# clone image, and setup the mouse callback function
	clone = copy.deepcopy(image)
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", select_chessboard_image_points)
	 
	# keep looping until the 'q' key is pressed
	while True:

		# draw the current points
		B = np.array(chessboard_corners_image_points) 
		image = draw_point_matrix(B, image, lineThickness=5)


		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, clear/reset the collected points
		if key == ord("r"):
			image = copy.deepcopy(clone)
			chessboard_corners_image_points = []

			cv2.destroyAllWindows()
			cv2.namedWindow("image")
			cv2.setMouseCallback("image", select_chessboard_image_points)
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			return np.array(chessboard_corners_image_points)
	 
	# if there are two reference points, then crop the region of interest
	# from teh image and display it
	if len(chessboard_corners_image_points) == 4:
		B = np.array(chessboard_corners_image_points) 
		image = draw_box(B, image)
		#cv2.imshow("selected corners", image)
		cv2.imshow("image", image)
		cv2.waitKey(0)
	 
	# close all open windows
	cv2.destroyAllWindows()


"""
#
# 	Aruco marker 3D model points. "world points" of corners
#

mh,mw = 3.0,3.0 	# inches
mH,mW = 31.5, 31.25 # inches 

aruco_model_points = np.array([
							# 	x,  	y,   	z= 0.0

							(	0.0, 	0.0, 	0.0		),             # m00
							(	mw, 	0.0, 	0.0		),			   # m01
							(	mw, 	mh, 	0.0		),             # m02
							(	0.0, 	mh,		0.0		),			   # m03

							(	mW-mw, 	0.0, 	0.0		),         	   # m10
							(	mW, 	0.0, 	0.0		),			   # m11
							(	mW, 	mh, 	0.0		),             # m12
							(	mW-mw, 	mh, 	0.0		),			   # m13

							(	mW-mw, 	mH-mh, 	0.0		),     		   # m20
							(	mW, 	mH-mh, 	0.0		),			   # m21
							(	mW, 	mH, 	0.0		),             # m22
							(	mW-mw, 	mH, 	0.0		)			   # m23

						])

# Board measurements
bh, bw = 7.75, 7.75 	# inches
tlx, tly = 11.50, 11.50 # inches
height = -0.75 			# inches
chessboard_model_points = np.array([

							# 	x,  	y,   	z= -0.75

							(	tlx, 	tly, 	height		),             # b0 - top left
							(	tlx+bw, tly, 	height		),			   # b1 - top right
							(	tlx+bw, tly+bh, height		),             # b2 - bottom right
							(	tlx, 	tly+bh,	height		)			   # b3 - bottom left
						])


# Base measurments
bh, bw = 11.75, 10.75 	# inches
tlx, tly = 10.15, 11.0	# inches
chesboard_lower_border_model_corner_points  = np.array([

							# 	x,  	y,   	z= 0.0

							(	tlx, 	tly, 	0.0		),             # b0 - top left
							(	tlx+bw, tly, 	0.0		),			   # b1 - top right
							(	tlx+bw, tly+bh, 0.0		),             # b2 - bottom right
							(	tlx, 	tly+bh,	0.0		)			   # b3 - bottom left
						])
"""

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
#   Load input images
#

def load_images_from_template(image_file_template):

	# dictonary : { base_fname : {'cv2 image' : image, 'full path' : fpath}, ...}
	fname_image_dict = {}

	images = glob.glob(image_file_template % '*')
	for fpath in images:
		base_fname = os.path.split(fpath)[1]
		image = cv2.imread(fpath)
		fname_image_dict[base_fname] = {'cv2 image' : image, 'full path' : fpath}

	return fname_image_dict

#
#   Write output images
#

def write_image_dict(output_image_dict, output_image_file_template):

	for base_fname,v in output_image_dict.items():
		fpath = output_image_file_template % base_fname
		image = v['cv2 image']
		cv2.imwrite(fpath, image)

#
# Estimate Camera Matrix
#

def estimate_camera_matrix(image, focal_length=None):
	size = input_image.shape

	# Camera internals (estimated)
	
	if focal_length is None:
		focal_length = size[1]

	center = (size[1]/2, size[0]/2)
	camera_matrix = np.array(
							 [[focal_length, 0, center[0]],
							 [0, focal_length, center[1]],
							 [0, 0, 1]], dtype = "double"
							 )

	return camera_matrix

#
# Estimate Distortion Coefficients
#

def estimate_distortion_coefficients():
	dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	return dist_coeffs

#
# Solve for rotation matrix and translation vector
#

def solve_PNPransac(model_points, image_points, camera_matrix, dist_coeffs):	# use aruco_image_points and aruc_model_points
	_, rotation_matrix, translation_vector, _ = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs)
	return rotation_matrix, translation_vector


#
#	Solve PNP 
#

def solve_PNP(model_points, image_points, camera_matrix, dist_coeffs):	# use aruco_image_points and aruc_model_points
	_, rotation_matrix, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs) #, flags = cv2.CV_EPNP)
	
	return rotation_matrix, translation_vector

#
# Project world points to 2D image points. World points is an nx3 matrix for n points (x,y,z).
#

def world_to_image_point(world_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs):

	image_points, _ = cv2.projectPoints(world_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)

	# clean up corners
	cleaned_image_points = np.array(list([ip[0].tolist() for ip in image_points]))
	image_points = cleaned_image_points.reshape(-1,2)

	return image_points


#
#
#	Drawing Methods
#
#

#
#	Draw nx2 point matrix onto image and return image
#

def draw_point_matrix(image_points, image, color=None, lineThickness=15):

	np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

	for i in range(image_points.shape[0]):
		center = np_2_tupple_int(image_points[i])
		radius = 1
		if color is None:
			color = (0,0,255)
		cv2.circle(image, center, radius, color, lineThickness, lineType=8, shift=0)

	return image

def draw_triangle(T, input_image, color=None, lineThickness = 4):
	# T - 3x2 matrix following clockwise convention
	np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

	if color is None:
		color = (0,255,0)

	for i in range(3):
		start_coor = T[i,:]
		if i == 2:
			end_coor = T[0,:]
		else:
			end_coor = T[i+1,:]

		# draw line on image
		cv2.line(input_image, np_2_tupple_int(start_coor), np_2_tupple_int(end_coor), color, lineThickness)

	return input_image

def draw_box(B, input_image, color=None, lineThickness = 4):
	# B - 4x2 matrix following clockwise convention
	np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

	if color is None:
		color = (0,255,0)

	for i in range(4):
		start_coor = B[i,:]
		if i == 3:
			end_coor = B[0,:]
		else:
			end_coor = B[i+1,:]

		# draw line on image
		cv2.line(input_image, np_2_tupple_int(start_coor), np_2_tupple_int(end_coor), color, lineThickness)

	return input_image

def draw_box_diagnols(B, input_image, color=None, lineThickness=4):
	# B - 4x2 matrix following clockwise convention
	np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

	if color is None:
		color = (0,255,0)

	# draw positive diagnol
	cv2.line(input_image, np_2_tupple_int(B[3]), np_2_tupple_int(B[1]), color, lineThickness)

	# draw negitive diagnol
	cv2.line(input_image, np_2_tupple_int(B[0]), np_2_tupple_int(B[2]), color, lineThickness)

	return input_image


# regression [slope, intercept] of best fit line from points
def regression_line_poly_coeffs(xs,ys):

	# perform linear regression
	slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)

	# polynomial coefficents for best fit line
	p = [slope, intercept]
	
	return p


def interpolate_additional_points(aruco_image_points, line_point_indices, input_image):

		line_image_points = aruco_image_points[line_point_indices]  # get image points that should lay on the line

		#	seperate x and y columns
		x = line_image_points[:,0]
		y = line_image_points[:,1]

		# perform linear regression and retreive polynomial coefficents for best fit line
		p = regression_line_poly_coeffs(x,y)

		# lambda function for y value given x -- rounded to an integer
		line_y_hat = lambda x: int(np.polyval(p,x))
		start_coor = (int(x[0]), line_y_hat(x[0]))
		end_coor = (int(x[-1]), line_y_hat(x[-1]))

		# draw line on image
		#lineThickness = 4
		#cv2.line(input_image, start_coor, end_coor, (255,0,0), lineThickness)

		return input_image

def point_of_intersection(coeffs_line_1, coeffs_line_2):

		"""

			line intersection

			ax+c=bx+d.

			rearrange to extract value of x
			ax-bx=d-c

			x=(d-c)/(a-b)
			y=a(d-c)/(a-b) +c.

			point of interscetion is

				<(d-c)/(a-b) ,  (ad - bc)/(a-b)>

		"""

		a, c = coeffs_line_1
		b, d = coeffs_line_2

		point = [int((d-c)/(a-b)), int((a*d - b*c)/(a-b))]

		return point

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

def coordinate_matrix_to_homogeneous(np_array):
	# [[x1,y1,z1],...,[xn,yn,zn]] -> [[x1,y1,z1,1],...,[xn,yn,zn,1]]
	new_col = np.ones((np_array.shape[0], 1))
	return np.concatenate((np_array, new_col), axis=1)

def from_homogeneous(np_array):
	# [x, y, 1] -> [x, y]
	# remove extra dimension slice
	return np.delete(np_array, (-1), axis=0)

#
#	New: camera to screen transformation
#

# def camera_to_screen_transformation():

#
#	world to camera transformation using rotation vector and translation vector
#

#
# Do camera and distortion really transformations apply here?
#
def world_to_camera_transformation(rotation_vector, translation_vector, camera_matrix, dist_coeffs, world_coordinates):

	# takes a vector of floats for input world coordinates

	# note: dividing by z will give u the projection

	# obtain 3x3 rotation matrix from rodrigues rotation vector
	rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

	# append translation vector to make 3x4 transformation matrix
	transformation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)

	# convert input world coordinates to homogeneous
	#p_world = to_homogeneous(world_coordinates)
	new_col = np.ones((world_coordinates.shape[0], 1))
	p_world = np.concatenate((world_coordinates, new_col), axis=1)

	# apply transformation and return world point in camera coordinates
	#p_camera = transformation_matrix.dot(p_world)
	p_camera = transformation_matrix.dot(p_world.T)

	#
	#	TODO: remove camera matrix intrinsic transformation
	#

	# apply camera matrix transformation
	# p_camera = camera_matrix.dot(p_camera)

	#
	# TODO: apply distortion !!! important !!!
	#


	#
	#	Note: or using transformation matrix
	#

	# world to camera transofmration matrix
	#tm = world_to_camera_transformation_matrix(rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	#homo_world_coordinates = coordinate_matrix_to_homogeneous(world_coordinates)
	#p_camera = tm.dot(homo_world_coordinates.T)	#.T (done in return statement)


	# return 
	# note: dividing by z will give u the projection
	return p_camera.T
	#return p_camera # neglect transpose to match opencv format (makes for easier transformations) 


#
#	return world to camera transformation matrix given rotation vector, translation vector, camera_matrix, dist_coeffs
#	### TODO: apply distortion !!! important !!! -- does camera and distortion belong here?

def world_to_camera_transformation_matrix(rotation_vector, translation_vector, camera_matrix, dist_coeffs):

	# takes a vector of floats for input world coordinates

	# note: dividing by z will give u the projection

	# obtain 3x3 rotation matrix from rodrigues rotation vector
	rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

	#
	#	note: inverse rotation matrix i think is just equal to its transpose***
	#

	# append translation vector to make 3x4 transformation matrix
	transformation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)

	#
	# Do camera and distortion really transformations apply here?
	#
	# multiply by camera matrix
	#transformation_matrix = camera_matrix.dot(transformation_matrix)

	#
	# TODO: apply distortion !!! important !!!
	#

	# return transformation matrix 3x4, must make 3d coordinates homgeneous to apply transformaton
	return transformation_matrix

#
#	Identity extend matrix for translation M -> |M 0s|
#												|0s 1|
#
def identity_extend_matrix(M):
	# M is placed in top left corner of square Identity matrix 
	new_col = np.zeros((M.shape[0], 1))
	extended_M = np.concatenate((M, new_col), axis=1)
	new_row = np.zeros((extended_M.shape[1], 1))
	extended_M = np.concatenate((extended_M, new_row.T), axis=0)
	n = max(extended_M.shape)
	#I = np.eye(n)
	B = np.zeros((n,n))
	extended_M = np.add(B,extended_M)
	# change bottom right corner to a 1
	extended_M[n-1,n-1]=1
	return extended_M

#
#	Translation matrix from translation vector - puts tanslation vector in top right column of Indentiy matrix of size n
#
def translation_matrix_from_vector(translation_vector, n):

	# put translation vector into top right corner of 4x4 identity matrix
	I = np.eye(n)

	#I[:3,3] = translation_vector.flatten()
	I[:n-1,n-1] = translation_vector.flatten()

	return I

#
#	return camera to world transformation matrix given rotation vector, translation vector, camera_matrix, dist_coeffs
#	or return world to camera inverse transformation matrix given rotation vector, translation vector, camera_matrix, dist_coeffs

def camera_to_world_transformation_matrix(rotation_vector, translation_vector, camera_matrix, dist_coeffs):

	# takes a vector of floats for input world coordinates

	# note: dividing by z will give u the projection

	# obtain 3x3 rotation matrix from rodrigues rotation vector
	rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

	#
	#	note: inverse rotation matrix i think is just equal to its transpose***
	#
	inverse_rotation_matrix = rotation_matrix.T
	#inverse_rotation_matrix = np.linalg.inv(rotation_matrix) # alternative same result


	"""
	#
	#	note: inverse translation matrix is translation matrix with signs inverted
	#
	inverse_translation_vector = np.negative(translation_vector)
	#print(inverse_translation_vector)
	#inverse_translation_vector = np.linalg.inv(translation_vector)
	#	Try inverse translation vector is -R.t*v - where v is the translation vector. they do not agree
	#inverse_translation_vector = np.negative(rotation_matrix).T.dot(translation_vector)


	#
	# Note: inverse transformation of translation and rotation add column and row to inv_rot of zeros except for the last diagnol
	#		use identity of same dimensions and put inv_translation vector in top three rows of last column
	#		multiply for inverse transformation
	#

	# put rotation matrix in top left section on identity matrix 4x4
	R_inv = identity_extend_matrix(inverse_rotation_matrix)



	# put translation vector into top right corner of 4x4 identity matrix
	T_inv = translation_matrix_from_vector(inverse_translation_vector, 4)

	# create inverse rotation and translation matrix by left multiplying inverse transformation matrix
	inverse_transformation_matrix = R_inv.dot(T_inv)
	"""


	#
	#	TMP check inverses
	#

	original_transformation_matrix = np.concatenate((rotation_matrix, translation_vector), axis=1)

	print("Checking Inverse rotation doted with self:")
	print(rotation_matrix.dot(inverse_rotation_matrix))
	print("\n")
	#print(original_transformation_matrix.dot(inverse_transformation_matrix))
	# inverse transformation matrix not creating identiy
	#print(inverse_transformation_matrix)
	#print(original_transformation_matrix) 



	#	Try inverse translation vector is -R.t*v - where v is the translation vector
	#
	inverse_translation_vector = np.negative(rotation_matrix).T.dot(translation_vector)
	#R_inv_zeros = identity_extend_matrix(inverse_rotation_matrix)
	#R_inv_zeros[3,3] = 0
	inverse_transformation_matrix = identity_extend_matrix(inverse_rotation_matrix)
	# make top right column the inverse translation vector
	inverse_transformation_matrix[:3,3] = inverse_translation_vector.flatten()


	#print(original_transformation_matrix.dot(inverse_transformation_matrix))


	# return transformation matrix 3x4, must make 3d coordinates homgeneous to apply transformaton
	return inverse_transformation_matrix

	#
	# No need for camera matrix or distortion coefficents i think since coordinates should already be in camera coordinates

	"""
	# append translation vector to make 3x4 transformation matrix
	#inverse_transformation_matrix = np.concatenate((inverse_rotation_matrix, inverse_translation_vector), axis=1)

	# 	invert camera matrix
	inverse_camera_matrix = np.linalg.inv(camera_matrix)

	# 
	#	C^-1*R^-1*T^-1
	#
	CR_inv = inverse_camera_matrix.dot(inverse_rotation_matrix)	# is this correct?
	# extend to apply inverse translation
	CR_inv_ext = identity_extend_matrix(CR_inv)

	# apply inverse translation
	inverse_transformation_matrix = CR_inv_ext.dot(T_inv)

	# try lopping off final row?
	inverse_transformation_matrix = inverse_transformation_matrix[:3,:]


	# multiply by camera matrix -- reverse order?
	#transformation_matrix = camera_matrix.dot(transformation_matrix)
	#inverse_transformation_matrix = inverse_transformation_matrix.dot(inverse_camera_matrix)
	# unreversed order
	#inverse_transformation_matrix = inverse_camera_matrix.dot(inverse_transformation_matrix)

	# left multiply the inverse camera matrix by the inverse rotation matrix?

	#
	# TODO: apply distortion !!! important !!!
	#
	"""

"""
def homogeneous_point_matrix_to_cartesian(homogeneous_point_matrix):


	# divide by last  coordinate for the projection and then remove last column

	# note: FutureWarning: in the future negative indices will not be ignored by `numpy.delete`.
	logger.warning("note: FutureWarning: in the future negative indices will not be ignored by `numpy.delete`.")
	hs = homogeneous_point_matrix[:,-1]

	cart_point_matrix = homogeneous_point_matrix / hs[:,None]

	return np.delete(cart_point_matrix, [-1], axis=1)
"""


def project_camera_points_to_image_plane(camera_point_matrix):

	#
	# note: in next implimentation add camera matrix and distortion trasformation
	#	aruco_3dimage_points = camera_matrix.dot(camera_point_matrix.T) ...

	# divide by z coordinate for the projection and remove z column

	zs = camera_point_matrix[:,2]
	image_point_matrix = camera_point_matrix / zs[:,None]

	return np.delete(image_point_matrix, [2], axis=1)
#
#	equation of plane from 3 points
#

def plane_from_3_points(point_matrix):
	#
	#	point matrix 3x3 where columns correspond to x,y,z and rows correspond to points p,q,r respectivley
	#
	#	return the vector n perpendicular to the plane
	#						and the vector p (given as first row in point matrix)

	p, q, r = point_matrix[0], point_matrix[1], point_matrix[2]
	# 
	#	find equation for aruco plane in the camera coordinate system
	#

	#
	#	p,q,r = m00_camera, m11_camera, m22_camera - points on the aruco plane in the camera coordinate system
	#	
	#	n = <a,b,c> ~ is a vector perpendicular to the aruco plane
	#
	#	<x0, y0, z0> = p ~ is a point on the plane
	#
	#	a(x - x0) + b(y - y0) + c(z - z0) = 0
	#
	#   a(x - px) + b(y - py) + c(z - pz) = 0
	#	
	#	solve for n
	#
	#	pq cross pr 
	#   pq = <qx - px, qy - py, qz - pz>
	# 	pr = <rx - px, ry - py, rz - pz>
	#	n = np.cross(pq, pr)
	#
	#

	# find pq and pr
	pq = np.subtract(q,p)
	pr = np.subtract(r,p)

	# find n
	n = np.cross(pq, pr)

	return n,p

#
#	find the point of intersection of a line through the origin and a plane formed by 3 points: p,q,r -- m00, m11, m22 or any aruco points
#

def find_intersection_with_aruco_plane(image_point, aruco_model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
	#
	#	image_point - pixel values [x,y]
	#	aruco model points ~ world coordinates
	#
	#	aruco_camera_points points of aruco model in camera coordinates
	#
	# 	returns in camera coordinates - next step transform to world coordinates 
	#	"image to camera"
	#
	#	TODO: operate on point matrix and not individual point


	#
	#	p,q,r = m00_camera, m11_camera, m22_camera - points on the aruco plane in the camera coordinate system
	#	

	# find corresponding points in R3 camera coordinates
	#p, q, r = world_to_camera_transformation(rotation_vector, translation_vector, camera_matrix, dist_coeffs, aruco_model_points[[0, 5, 10]])
	#
	#	Should remove camera matrix and distortion correction i think
	#
	pqr_camera_point_matrix = world_to_camera_transformation(rotation_vector, translation_vector, camera_matrix, dist_coeffs, aruco_model_points[[0, 5, 10]])

	#
	#	find 2 vectors necissary for the equation of a plane - n (normal to plane) and p, the first point in the point matrix (m00)
	#
	n, p = plane_from_3_points(pqr_camera_point_matrix)


	#
	#	find parametric representation for the line going through any desired pixel on the screen
	#

	# take a point on the line
	# - can use origin since it is camptured by the pinhole
	# so the parametric equation is simply
	# <0, 0, 0> +	t<lx, ly, 1> ~ where lx, ly are the pixel values of the point in quesiton
	# x = tlx, y = tly, z = t

	# now solve for t where line intersects the plane
	#
	#	a(x - px) + b(y - py) + c(z - pz) = 0
	# sub in line equations for x, y and z
	#	a(t*lx - px) + b(t*ly - py) + c(t - pz) = 0
	#
	#  solve for t
	#
	#	a*t*lx -a*px + b*t*ly -b*py + c*t -c*pz = 0
	#
	#	a*t*lx + b*t*ly + c*t = a*px + b*py + c*pz
	#
	#   t(a*lx + b*ly + c) = a*px + b*py + c*pz
	#
	#	t = (a*px + b*py + c*pz)/(a*lx + b*ly + c)
	#

	a, b, c = n
	px, py, pz = p
	lx, ly = image_point

	# solve for t
	t = (a*px + b*py + c*pz)/(a*lx + b*ly + c)

	# then plug that value of t back in to the equation of the line to find the point of intersection
	intersection_point = np.array([[t*lx, t*ly, t]])

	return intersection_point

#
#	TODO: 
#

#
# TODO: image to model
#
def image_points_to_model_points(image_points, rotation_vector, translation_vector, constant_z_height= -0.75):

	#
	# find inverse transformation
	#

	# obtain 3x3 rotation matrix from rodrigues rotation vector
	rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

	inverse_rotation_matrix = np.linalg.inv(rotation_matrix)
	inverse_translation_vector = np.linalg.inv(translation_vector)

	# add z model coordinate
	new_world_col = np.array([[constant_z_height], [constant_z_height], [constant_z_height], [constant_z_height]])

	# find z image coordinates


	print(new_col)
	print(image_points)
	camera_coors = np.hstack((image_points, new_col))


#
#
#								Program
#
#

if __name__ == '__main__':

	#marker_measurements_to_model_points(3.0,  31.25, 31.5, 44.3713, 1.5)
	#aruco_model_points = aruco_measurements_to_model_points(3.0,  31.25, 31.5, 42, 1.5)
	aruco_model_points = aruco_measurements_to_model_points(3.0,  31.25, 31.5, 44.3713, 1.5)



	input_image_dict = load_images_from_template(input_ftemplate)   # originals
	output_image_dict = input_image_dict.copy() 


	for k,v in input_image_dict.items():

		#
		# 	load image
		#

		input_image = v['cv2 image']

		#
		#	perform basic detection on aruco corners, solve pnpransac, project points
		#

		# detect image points
		aruco_image_points = marker_detector.detect_aruco_image_points(input_image)

		camera_matrix = estimate_camera_matrix(input_image)
		dist_coeffs = estimate_distortion_coefficients()
		rotation_vector, translation_vector = solve_PNP(aruco_model_points, aruco_image_points, camera_matrix, dist_coeffs)

		#
		# Have the user select the image points coresponding to corners
		#

		#selected_image_point_matrix = user_input_chessboard_corners(input_image)

		#p_world = np.array([[0.0,0.0,0.0],[1.0,1.0,1.0]])
		#p_camera = world_to_camera_transformation(rotation_vector, translation_vector, camera_matrix, dist_coeffs, p_world)
		#p_image = project_camera_points_to_image_plane(p_camera)
		#print(selected_image_point_matrix[0])
		#intersection_point_camera = find_intersection_with_aruco_plane(selected_image_point_matrix[0], aruco_model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		#intersection_point_world = find_intersection_with_aruco_plane(np.array([0,0]), aruco_model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		#print(intersection_point_camera)
		# check intersection point by re projecting it
		#intersection_point_camera = world_to_camera_transformation(rotation_vector, translation_vector, camera_matrix, dist_coeffs, intersection_point_world)
		#intersection_point_image = project_camera_points_to_image_plane(intersection_point_camera)
		#print(intersection_point_image)
		#draw
		#input_image = draw_point_matrix(intersection_point_image, input_image, lineThickness=50)

		#
		# check world to camera transformation matrix against world to camera transformation
		#

		"""
		check_p_world = np.array([[0.0,0.0,0.0],[1.0,1.0,1.0]])

		# world to camera transofmration matrix - note TODO: remove commented code applying camera intrinisc transformations
		tm_world_to_camera = world_to_camera_transformation_matrix(rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		homo_check_p_world = coordinate_matrix_to_homogeneous(check_p_world)
		#print(homo_check_p_world)
		#print(tm_world_to_camera.dot(homo_check_p_world.T).T)

		# world to camera transformation
		check_p_camera = world_to_camera_transformation(rotation_vector, translation_vector, camera_matrix, dist_coeffs, check_p_world)

		tm_camera_to_world = camera_to_world_transformation_matrix(rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		
		# check inverse transformation should give identity
		#print(tm_camera_to_world)
		print(tm_world_to_camera)
		#print(tm_camera_to_world.dot(tm_world_to_camera.T))
		"""




		#
		#	TMP** tmp** Try using pnp solve for inverse transformation with camera intrinsics as identity
		#			-wont work because its meant to solve in 2d

		I = np.eye(3)
		d = np.zeros((4,1)) 
		# world to camera transofmration matrix - note TODO: remove commented code applying camera intrinisc transformations
		tm_world_to_camera = world_to_camera_transformation_matrix(rotation_vector, translation_vector, I, d)
		

		#
		#	camera points can be obtained using this method
		#

		aruco_camera_points = world_to_camera_transformation(rotation_vector, translation_vector, I, d, aruco_model_points)
		
		#
		# check camera points retreived using above method by applying intrinsic transformations (tmp ignore dist)
		#		(aruco_camera_points = world_to_camera_transformation(rotation_vector, translation_vector, I, d, aruco_model_points))

		# apply camera matrix transformation to aruco camera points
		aruco_3dimage_points = camera_matrix.dot(aruco_camera_points.T)
		#print(aruco_3dimage_points.T)
		# temporarily ignore distortion affects
		check_aruco_image_points = project_camera_points_to_image_plane(aruco_3dimage_points.T)

		# draw to check sucess
		#input_image = draw_point_matrix(check_aruco_image_points, input_image, color=(0,0,255))

		#
		# 	set output image in dictonary
		#

		#output_image_dict[k]['cv2 image'] = input_image

		#sucess!!

		#
		#	Find a transformation matrix to turn camera points back into world points
		#


		c2w_T = camera_to_world_transformation_matrix(rotation_vector, translation_vector, I, d)
		#print(aruco_model_points)
		#print(aruco_camera_points)


		homo_aruco_camera_points = coordinate_matrix_to_homogeneous(aruco_camera_points)
		print(homo_aruco_camera_points)
		check_homo_world_point_matrix = c2w_T.dot(homo_aruco_camera_points.T).T
		#print(check_homo_world_point_matrix)
		#print(check_homo_world_point_matrix.shape)
		check_world_point_matrix = np.delete(check_homo_world_point_matrix, [3], axis=1)
		#check_world_point_matrix = homogeneous_point_matrix_to_cartesian(check_homo_world_point_matrix)
		print(check_world_point_matrix)
		print(check_world_point_matrix.shape)
		print(aruco_model_points)

		print("\ndiffrence:")
		diff = np.subtract(aruco_model_points,check_world_point_matrix)
		print(np.rint(diff))
		print("\n")
		w2c_T = world_to_camera_transformation_matrix(rotation_vector, translation_vector, I, d)
		homo_aruco_model_points = coordinate_matrix_to_homogeneous(aruco_model_points)
		print(w2c_T.dot(homo_aruco_model_points.T).T)

		#
		#	split operations into seperate rotation and translation matrices
		#

		print("Attempting transformation by applying inverse rotation matrix first...")
		# obtain 3x3 rotation matrix from rodrigues rotation vector
		rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
		inverse_rotation_matrix = rotation_matrix.T

		print("Original Camera coordinates:")
		print(aruco_camera_points)
		
		print("Rotated Camera coordinates to world")
		rotated = inverse_rotation_matrix.dot(aruco_camera_points.T).T
		print(rotated)

		print("Rotated coordinates back to camera")
		check_rerotate = rotation_matrix.dot(rotated.T).T
		print(check_rerotate)

		print("\n")
		print("Attempting to find inverse translation matrix by first converting sucsesffuly rotated coordinates to homogeneous coordinates")
		homogeneous_rotated_points = coordinate_matrix_to_homogeneous(rotated)
		print(homogeneous_rotated_points)

		print("\nThe negitive of the first row should be the translation vector")
		print(homogeneous_rotated_points[0,:])

		#print("\n Compare this to the original translation vector") NOT THE SAME
		#print(translation_vector)

		print("\n Compare this to -R.T*t")
		inverse_translation_vector = np.negative(rotation_matrix).T.dot(translation_vector)
		print(inverse_translation_vector)

		print("It comes out the same")

		print("extend inverse rotation matrix to get:")
		extended_inv_R = identity_extend_matrix(inverse_rotation_matrix)
		print(extended_inv_R)

		print("with original inverse roation matrix as:")
		print(inverse_rotation_matrix)

		print("\nput the inverse translation matrix in top left corner of 4x4 identity")
		extended_inv_t = translation_matrix_from_vector(inverse_translation_vector, 4)
		print(extended_inv_t)

		print("with original inverse translation vector as:")
		print(inverse_translation_vector)

		print("multiply invRextended by invTranslationextended to get inverse transformation matrix (camra to world)")
		c2w_T = extended_inv_R.dot(extended_inv_t.T)
		print(c2w_T)

		print("\nTest new inverse transformation matrix (c->w) on aruco_camera_points")
		print("first make aruco camera points homogeneous...")
		print("Original aruco camera points:")
		print(aruco_camera_points)
		print("Homogeneous aruco camera points:")
		homo_aruco_camera_points = coordinate_matrix_to_homogeneous(aruco_camera_points)
		print(homo_aruco_camera_points)
		print("\nTry out new camera to world transformation matrix:")
		check_homo_world_point_matrix = c2w_T.dot(homo_aruco_camera_points.T).T
		print(check_homo_world_point_matrix.shape)
		print(check_homo_world_point_matrix)
		print("\nCompare with model points:")
		print(aruco_model_points)

		print("\nDiffrence:")
		diff = np.subtract(aruco_model_points,check_world_point_matrix)
		print(diff)
		print("\nRounded Diff:")
		print(np.rint(diff))

		"""
		print("combine inverse rotation matrix and inverse translation matrix into single matrix")
		inverse_transformation_matrix = identity_extend_matrix(inverse_rotation_matrix)
		# make top right column the inverse translation vector
		inverse_transformation_matrix[:3,3] = inverse_translation_vector.flatten()
		print(inverse_transformation_matrix)


		print("\nTest new inverse transformation matrix (c->w) on aruco_camera_points")
		print("first make aruco camera points homogeneous...")
		print("Original aruco camera points:")
		print(aruco_camera_points)
		print("Homogeneous aruco camera points:")
		homo_aruco_camera_points = coordinate_matrix_to_homogeneous(aruco_camera_points)
		print(homo_aruco_camera_points)
		print("\nTry out new camera to world transformation matrix:")
		check_homo_world_point_matrix = inverse_transformation_matrix.dot(homo_aruco_camera_points.T).T
		print(check_homo_world_point_matrix)
		print("\nCompare with model points:")
		print(aruco_model_points)
		#print("\nDiffrence:")
		#diff = np.subtract(aruco_model_points,check_world_point_matrix)
		"""



		break


		#print(aruco_model_points.shape)
		#print(aruco_camera_points.shape)
		#inverse_rotation_vector, inverse_translation_vector = solve_PNP(aruco_camera_points, aruco_model_points, I, d)
		#tm_camera_to_world = world_to_camera_transformation_matrix(inverse_rotation_vector, inverse_translation_vector, I, d)

		#print(tm_world_to_camera)
		#print(tm_camera_to_world)
		#print(tm_world_to_camera.dot(tm_camera_to_world))
		
		#print(check_p_world)
		#print(testing_check_p_world)

		#break
		# 
		#	find equation for aruco plane in the camera coordinate system
		#

		#
		#	p,q,r = m00_camera, m11_camera, m22_camera - points on the aruco plane in the camera coordinate system
		#	
		#	n = <a,b,c> ~ is a vector perpendicular to the aruco plane
		#
		#	<x0, y0, z0> = p ~ is a point on the plane
		#
		#	a(x - x0) + b(y - y0) + c(z - z0) = 0
		#
		#   a(x - px) + b(y - py) + c(z - pz) = 0
		#	
		#	solve for n
		#
		#	pq cross pr 
		#   pq = <qx - px, qy - py, qz - pz>
		# 	pr = <rx - px, ry - py, rz - pz>
		#	n = np.cross(pq, pr)
		#
		#

		#
		#	find parametric representation for the line goind through any desired pixel on the screen
		#

		# take a point on the line
		# - can use origin since it is camptured by the pinhole
		# so the parametric equation is simply
		# <0, 0, 0> +	t<lx, ly, 1> ~ where lx, ly are the pixel values of the point in quesiton
		# x = tlx, y = tly, z = t

		# now solve for t where line intersects the plane
		#
		#	a(x - px) + b(y - py) + c(z - pz) = 0
		# sub in line equations for x, y and z
		#	a(t*lx - px) + b(t*ly - py) + c(t - pz) = 0
		#
		#  solve for t
		#
		#	a*t*lx -a*px + b*t*ly -b*py + c*t -c*pz = 0
		#
		#	a*t*lx + b*t*ly + c*t = a*px + b*py + c*pz
		#
		#   t(a*lx + b*ly + c) = a*px + b*py + c*pz
		#
		#	t = (a*px + b*py + c*pz)/(a*lx + b*ly + c)
		#

		# then plug that value of t back in to the equation of the line to find the point of intersection


		"""
		p_camera_a = world_to_camera_transformation_matrix(rotation_vector, translation_vector, camera_matrix, dist_coeffs, p_world)
		print(p_camera_a)

		# divide by z coordinate
		p_camera_a = np.divide(p_camera_a, p_camera_a[2])
		print(p_camera_a)
		p_camera_b = world_to_image_point(np.array([p_world]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
		print(p_camera_b)
		"""

		#B_image_points = user_input_chessboard_corners(input_image)
		# add z coordinate
		#height = -0.75 			# inches
		#image_points_to_model_points(B_image_points, rotation_matrix, translation_vector, height)

		"""
		# add z coordinate
		height = -0.75 			# inches
		new_col = np.array([[height], [height], [height]])
		print(new_col)
		print(B_image_points)
		camera_coors = np.hstack((B_image_points, new_col))

		print(camera_coors)

		inverse_rotation_matrix = numpy.linalg.inv(rotation_matrix)
		inverse_translation_vector = numpy.linalg.inv(translation_vector)
		"""



		"""

		#
		#	perform basic detection on aruco corners, solve pnpransac, project points
		#

		# detect image points
		aruco_image_points = marker_detector.detect_aruco_image_points(input_image)

		camera_matrix = estimate_camera_matrix(input_image)
		dist_coeffs = estimate_distortion_coefficients()
		rotation_matrix, translation_vector = solve_PNPransac(aruco_model_points, aruco_image_points, camera_matrix, dist_coeffs)

		# NEW solvePNP
		rotation_matrix, translation_vector = solve_PNP(aruco_model_points, aruco_image_points, camera_matrix, dist_coeffs)

		# project 3d world points of chessboard corners to image points
		chessboard_corner_points = world_to_image_point(chessboard_model_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)

		# project correct 3d world points of aruco markers to image points
		check_aruco_image_points = world_to_image_point(aruco_model_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)

		# project 3d world points of chessboard base frame to image points
		chessboard_base_model_image_points = world_to_image_point(chesboard_lower_border_model_corner_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)

		#
		# draw all points
		#

		# draw detected aruco trianlge ingonoring pnpsolver
		# indices 0,5,10
		#input_image = draw_triangle(aruco_image_points[[0,5,10]], input_image, color=(255,0,0))


		# draw projected aruco trianlge using pnpsolver
		# indices 0,5,10
		#input_image = draw_triangle(check_aruco_image_points[[0,5,10]], input_image, color=(0,0,255), lineThickness=10)

		# draw detected aruco box ingonoring pnpsolver
		#indices 0,5,10,15
		#input_image = draw_box(aruco_image_points[[0,5,10,15]], input_image, color=(255,0,0))
		#input_image = draw_box_diagnols(aruco_image_points[[0,5,10,15]], input_image, color=(255,0,0))

		# draw projected aruco box using pnpsolver
		#indices 0,5,10,15
		#input_image = draw_box(check_aruco_image_points[[0,5,10,15]], input_image, color=(0,0,255))
		#input_image = draw_box_diagnols(check_aruco_image_points[[0,5,10,15]], input_image, color=(0,0,255), lineThickness=10)

		# draw error points/ predicted points
		#input_image = draw_point_matrix(check_aruco_image_points, input_image)	# "error"

		# draw detected aruco points
		#input_image = draw_point_matrix(aruco_image_points, input_image, color=(255,0,0))

		# draw predicted chessboard corner points
		#input_image = draw_point_matrix(chessboard_corner_points, input_image, color=(0,255,0))


		# draw chessboard frame
		input_image = draw_box(chessboard_corner_points, input_image, color=(0,255,0), lineThickness=2)
		#input_image = draw_box_diagnols(chessboard_corner_points, input_image, color=(0,255,0))

		# draw base of chessboard frame
		input_image = draw_box(chessboard_base_model_image_points, input_image, color=(0,255,0))
		#input_image = draw_box_diagnols(chessboard_base_model_image_points, input_image, color=(0,255,0), lineThickness=20)



		#
		# 	set output image in dictonary
		#

		output_image_dict[k]['cv2 image'] = input_image
		"""


	# write output image dictonary
	#write_image_dict(output_image_dict, output_ftemplate)




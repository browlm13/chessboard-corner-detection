#!/usr/bin/env python

"""

	World to image points.

	World coordinate to image coordinate. Move from any 3D world coordinate to 
	a pixel on the given image. 

	1.) Detect location of aruco marker corners on 2D image.

			detect all corners mij for 0 < i < 4 and  0 < j < 4.


							m00 m01 	m10 m11
							m03 m02 	m13 m12

								(chessboard)

							m30 m31 	m20 m21
							m33 m32 	m23 m22


			where,

				mij ~ 2d pixel coordinate [x,y] of corner index "j" of aruco marker with id "i". 

				* Coordinates of corners use standard image axis - "(0,0) is at the top left of the image"

		This is done in the file "aruco_marker_detection.py" by the method: 

			??? ### detect_aruco_marker_corners(image, aruco_dict)

		Which returns a matrix, A, of 2D coordinates of aruco marker corners on the image.


		
									A = 

					[ [[m00_x, m00_y], ..., [m03_x, m03_y]],
						... , 
					[[m30_x, m30_y], ..., [m33_x, m33_y]] ]


				  (Aruco Marker Corner 2D Image Locations Matrix)

			where,

				A ~ Matrix of 2D image coordinates of aruco marker corners
				mij_x ~ x coordinate of aruco marker "i"'s "j"th corner.
				mij_y ~ y coordinate of aruco marker "i"'s "j"th corner.

			
				* (4x4x2) ~ (marker_id, corner_id, 2D_image_coordinate)
				* Coordinates of corners use standard image axis - "(0,0) is at the top left of the image"
		
	2.) Solve for Rotation matrix and translation vector 
		to move between camera coordinates and world coordinates.

			Using 
				- known world coordinates of the aruco markers (object points from real world measurements) 
				- detected corners on a 2D image (A)
				- instrinsic camera parameters file from "camera_calibration.py" (mtx, dist)

			solve for the rotation matrix, R, and the translation vector, t,
			using opencv's cv2.solvePnPRansac function. 

	3.) Move from any 3D wolrd coordinate to 2D image coordinate.

			Now the 2D coordinate of any point on the image can be found if the 3D world coordinate
			is know by using R, t, mtx, dist to go from world coordinates to camera coordinates, then by 
			projecting the 3D camera coordinates point to the image plane using opencv's cv2.projectPoints.

			where,

				mtx ~ camera matrix.
				dist ~ distortion coefficients.
				R ~ rotation matrix 
				t ~ translation vector



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

# external
import glob
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

def draw_point_matrix(image_points, image, color=None, thickness=15):

	np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

	for i in range(image_points.shape[0]):
		center = np_2_tupple_int(image_points[i])
		radius = 1
		if color is None:
			color = (0,0,255)
		cv2.circle(image, center, radius, color, thickness, lineType=8, shift=0)

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


#
#
#								Program
#
#

if __name__ == '__main__':
	

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


	# write output image dictonary
	write_image_dict(output_image_dict, output_ftemplate)


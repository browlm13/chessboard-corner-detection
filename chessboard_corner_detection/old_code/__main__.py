#!/usr/bin/env python

"""

	Create annotated data set for semantic segmentation training using 3 aruco markers and a video.

"""

__author__ = "l.j. Brown"
__version__ = "1.0.4"

#
#
#									imports	
#
#

# internal
import logging


# external
#import glob
#import numpy as np
#import cv2


# my lib
import frames
import aruco_marker_detection as marker_detector

#
#
#									Settings
#
#

ARUCO_MARKER_SETTINGS_FNAME = "aruco_markers/aruc_markers_settings.json" 		# Do not change

MOVIE_NUM = 12
input_video_path = "test_input_video/%s.mov" % MOVIE_NUM
output_frames_directory_path = "test_aruco_images_input_%s" % MOVIE_NUM
input_ftemplate = output_frames_directory_path + "/%s"
output_ftemplate = "test_aruco_images_output_%s/%s" % (MOVIE_NUM, "%s")


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
#
#								Program
#
#

if __name__ == '__main__':

	#
	#	convert input video into individual frames
	#

	frames.to_frames_directory(input_video_path, output_frames_directory_path)


	#
	# Annotate frames directory
	#

	#
	#	write test images to check bounding polygon
	#

	"""

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
		input_image = draw_triangle(aruco_image_points[[0,5,10]], input_image, color=(255,0,0))


		# draw projected aruco trianlge using pnpsolver
		# indices 0,5,10
		input_image = draw_triangle(check_aruco_image_points[[0,5,10]], input_image, color=(0,0,255), lineThickness=10)

		# draw error points/ predicted points
		#input_image = draw_point_matrix(check_aruco_image_points, input_image)	# "error"

		# draw detected aruco points
		#input_image = draw_point_matrix(aruco_image_points, input_image, color=(255,0,0))

		# draw predicted chessboard corner points
		#input_image = draw_point_matrix(chessboard_corner_points, input_image, color=(0,255,0))


		# draw chessboard frame
		input_image = draw_box(chessboard_corner_points, input_image, color=(0,255,0), lineThickness=15)
		#input_image = draw_box_diagnols(chessboard_corner_points, input_image, color=(0,255,0))

		# draw base of chessboard frame
		input_image = draw_box(chessboard_base_model_image_points, input_image, color=(0,255,0))
		input_image = draw_box_diagnols(chessboard_base_model_image_points, input_image, color=(0,255,0), lineThickness=20)



		#
		# 	set output image in dictonary
		#

		output_image_dict[k]['cv2 image'] = input_image


	# write output image dictonary
	#write_image_dict(output_image_dict, output_ftemplate)
	"""

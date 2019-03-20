#!/usr/bin/env python

"""

		Aruco Marker Detection.

		Performs aruco marker corner detection on images 
		and return matrix of 2D coordinates of aruco marker corners, A.

		* Format of A is described by diagram 3 bellow.

		Diagram 1:
		--------------------------------------------------------------------------

									m0	 m1
									   B
									m3   m2

							  (marker arrangment)

			where,

				mi ~ Aruco marker with associate id "i"
				B  ~ Chessboard

				* Settings stored in json aruco markers settings file
				* Setting file and images created in "aruco_marker_creation.py"

		--------------------------------------------------------------------------

		Bellow is the aruco marker arrangment above with each marker expanded 
		to reveal its four corners to be detected.

		Diagram 2:
		--------------------------------------------------------------------------


								m00 m01 	m10 m11
								m03 m02 	m13 m12

								m30 m31 	m20 m21
								m33 m32 	m23 m22

				(marker arrangment expanded to reveal all corners to detect)

			where,

				mij ~ 2d pixel coordinate [x,y] of corner index "j" of aruco marker with id "i". 

				* Coordinates of corners use standard image axis - "(0,0) is at the top left of the image"
				* Chessboard not shown

		--------------------------------------------------------------------------

		The corner locations are stored in a (4x4x2) numpy array, A.

		Diagram 3:
		--------------------------------------------------------------------------


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
		
		--------------------------------------------------------------------------
"""

__author__ = "l.j. Brown"
__version__ = "1.0.2"

#
#
#									imports	
#
#

# internal
import os
import json
import logging

# external
import numpy as np
import glob
import cv2
import cv2.aruco as aruco

#
#
#									Settings
#
#

aruco_markers_settings_fname = "aruco_markers/aruc_markers_settings.json" 		# Do not change

input_ftemplate = "test_aruco_images_input/%s"
output_ftemplate = "test_aruco_images_output/%s"

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
#	load aruco marker settings dictonary
#

def load_settings_dictonary(settings_file_path):

	logger.info("Reading settings dictonary \"%s\"..." % settings_file_path)

	with open(settings_file_path) as f:
	    settings_dict = json.load(f)

	return settings_dict

#
#	load aruco marker dictonary from settings file
#

def load_aruco_dict(settings_file_path):

	logger.info("Reading aruco marker dictonary from settings file \"%s\"..." % settings_file_path)

	# load settings dict from file path
	settings_dict = load_settings_dictonary(settings_file_path)

	# retreive aruco dictonary id from settings dictonary
	aruco_dict_id = settings_dict["aruco_dict_id"]

	# retreive aruco dictonary using the id from cv2.aruco
	aruco_dict = aruco.Dictionary_get(aruco_dict_id)

	return aruco_dict

#
#	load corner id to aruco marker location sting, "corner_id_dict", from settings file
#	

def load_corner_id_dict(settings_file_path):

	logger.info("Reading \'corner_id_dict\' from settings file \"%s\"..." % settings_file_path)

	# load settings dict from file path
	settings_dict = load_settings_dictonary(settings_file_path)

	# retreive corner_id_dict from settings dictonary
	corner_id_dict = settings_dict["corner_id_dict"]

	return corner_id_dict


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
#   Detect aruco markers corner locations on image -- Return matrix of 2D coordinates of aruco marker corners, A
#

#def detect_aruco_marker_corners(image, aruco_dict):		# detect_A(image, aruco_dict)
def detect_A(image, aruco_dict):

	# cv2.aruco method call
	parameters = aruco.DetectorParameters_create()

	# convert to gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Detect marker corners
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	
	#
	#		Return matrix of 2D coordinates of aruco marker corners, A
	#	

	# clean up corners
	cleaned_corners = np.array(list([c[0].tolist() for c in corners]))

	# sort by marker id
	sorted_indices = np.argsort(ids, axis=0)
	A = cleaned_corners[sorted_indices] ### ??? ### [::-1] reverse to match defined format? or already there?

	# return aruco marker corner locations in 2D image matrix A,
	return A


#
#
#								Program
#
#

if __name__ == '__main__':

	input_image_dict = load_images_from_template(input_ftemplate)   # originals
	output_image_dict = input_image_dict.copy() 

	# load aruco dictonary
	aruco_dict = load_aruco_dict(aruco_markers_settings_fname)

	for k,v in input_image_dict.items():

		# tmp
		A = detect_A(v['cv2 image'], aruco_dict)


	write_image_dict(output_image_dict, output_ftemplate)


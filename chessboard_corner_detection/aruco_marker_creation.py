#!/usr/bin/env python

"""

	Aruco Marker Creation

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
__version__ = "1.0.4"

#
#
#									imports	
#
#

# internal
import json
import logging

# external
import numpy as np
import cv2
import cv2.aruco as aruco

#
#
#									Settings
#
#

ARUCO_MARKER_SETTINGS_FNAME = "aruco_markers/aruc_markers_settings.json" 		# Do not change
aruco_markers_ftemplate = "aruco_markers/aruco_%s.png"

marker_size = 400	# pixels

aruco_dict_id = aruco.DICT_6X6_250
corner_id_dict = {
	0 : 'top_left',
	1 : 'top_right',
	2 : 'bottom_right'
	#3 : 'bottom_left'
}

#
#	aruco marker settings dictonary
#

settings_dict = {
	"ARUCO_MARKER_SETTINGS_FNAME" : ARUCO_MARKER_SETTINGS_FNAME,
	"aruco_markers_ftemplate" : aruco_markers_ftemplate,
	"corner_id_dict" : corner_id_dict,
	"aruco_dict_id" : aruco_dict_id,
	"marker_size" : marker_size
}

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
#	write settings to json file
#

def write_settings_file(settings_dict):

	# retreive file name
	ARUCO_MARKER_SETTINGS_FNAME = settings_dict["ARUCO_MARKER_SETTINGS_FNAME"]

	# logger
	logger.info("Writing aruco marker settings file (json file) too \"%s\"" % ARUCO_MARKER_SETTINGS_FNAME)

	# write json file
	with open(ARUCO_MARKER_SETTINGS_FNAME, 'w') as outfile:
	    json.dump(settings_dict, outfile)

	# logger
	logger.info("Finished writing aruco marker settings file.")


#
#	create markers -- and write settings file
#

def create_markers(settings_dict):

	# write settings dict to json file
	write_settings_file(settings_dict)

	# get file template for individual aruco marker image
	aruco_markers_ftemplate = settings_dict["aruco_markers_ftemplate"]

	# load aruco dict
	aruco_dict = aruco.Dictionary_get(settings_dict["aruco_dict_id"])

	# get number of markers
	nMarkers = len(settings_dict["corner_id_dict"])
	
	# get marker size
	markerSize = settings_dict["marker_size"]

	# create marker image files and write them
	for k,v in settings_dict["corner_id_dict"].items():
		marker_img = aruco.drawMarker(aruco_dict, k, markerSize)
		fpath = aruco_markers_ftemplate % v

		# logger
		logger.info("Writing aruco marker image file to \"%s\"" % fpath)
		
		cv2.imwrite(fpath, marker_img)

	# logger
	logger.info("Finished writing aruco marker image files.")

#
#
#								Program
#
#

if __name__ == "__main__":

	#
	# create marker images and settings file
	#

	create_markers(settings_dict)

import numpy as np
import cv2
import cv2.aruco as aruco


"""

			Create 4 aruco markers for chessboard corner detection

			TODO: move aruco dictonary information to external file

"""

#
#	paths
#

aruco_markers_ftemplate = "aruco_markers/aruco_%s.png"

#
# 	dictonaries
#
corner_id_dict = {
	0 : 'top_left',
	1 : 'top_right',
	2 : 'bottom_left',
	3 : 'bottom_right'
}
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

#
#	create markers
#

nMarkers, markerSize = len(corner_id_dict), 400
for k,v in corner_id_dict.items():
	marker_img = aruco.drawMarker(aruco_dict, k, markerSize)
	fpath = aruco_markers_ftemplate % v
	cv2.imwrite(fpath, marker_img)

	#cv2.imshow('frame',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
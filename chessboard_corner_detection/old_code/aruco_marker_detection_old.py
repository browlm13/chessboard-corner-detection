import numpy as np
import glob
import cv2
import cv2.aruco as aruco
import os

"""

			Create 4 aruco markers for chessboard corner detection

			TODO: move aruco dictonary information to external file

"""




#
#
# 				aruco marker spacing
#
#

#
#				(8x8 imaginary grid)
#

#marker_length = 4 # in inches
#connection_length = 32 # in inches from aruco far corner to other aruco far corner
#imaginary_aruco_board_dimensions = (connection_length/marker_length, connection_length/marker_length) # "8x8" in markers

#
#	pixels per inch
#
#ppi = 1

#	imaginary board dimensions in pixels
#iab_dim_pixels = (connection_length*ppi, connection_length*ppi)


#
#	birds eye aruco coordinates
#

#
#	m00, m01		m10, m11
#	m02, m03		m12, m13
#
#	m20, m21		m30, m31
#	m22, m23		m32, m33
#

#
# (0,0),(1,0) ... (6,0),(7,0)
# (0,1),(1,1) ... (6,1),(7,1)
#
#		...			...
#
# (0,6),(1,6) ... (6,6),(7,6)
# (0,7),(1,7) ... (6,7),(7,7)

#
#
# 				aruco marker spacing
#
#

#
#				(8x8 imaginary grid)
#

#
#	pixels per inch
#

ppi = 100


#
#	birds eye aruco coordinates
#

#
#	(0,0) ... (8,0) 
#
#	 ...	   ...
#
#	(0,8) ... (8,8)
#

# assumes 8x8 imaginary board
aruco_keypoints = np.array([[0,0], [8,0], [0,8]]) #, [8,8]]) # [m00, m11, m22, m33] points
multiplier = 8*ppi
be_aruco_keypoints_pixels = np.multiply(aruco_keypoints, multiplier) #pixels






#
# paths
#

input_ftemplate = "test_aruco_images_input/%s"
#input_ftemplate = "aruco_markers/%s"
output_ftemplate = "test_aruco_images_output/%s"

#
#   dictonaries
#
corner_id_dict = {
	0 : 'top_left',
	1 : 'top_right',
	2 : 'bottom_left',
	3 : 'bottom_right'
}
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

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
#   Detect Aruco markers
#

def detect_aruco_markers(image, aruco_dict):
	parameters = aruco.DetectorParameters_create()

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	"""
	#tmp
	ret, thresh = cv2.threshold(gray, 127, 255, 0)
	im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # dont have to approx! can get all points
	cv2.drawContours(image, contours, -1, (0,255,0), 5)
	return image
	"""

	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	



	#
	#		Return matrix of 2D coordinates of aruco marker corners
	#	

	# TBD






	#gray = aruco.drawDetectedMarkers(gray, corners)
	#
	# 	draw detection
	#
	# aruco.drawDetectedMarkers(image, corners, ids, 5)



	##
	## Testing birds eye warp
	##

	#
	#	birds eye aruco coordinates
	#

	#
	#	m00, m01		m10, m11
	#	m02, m03		m12, m13
	#
	#	m30, m31		m20, m21
	#	m32, m33		m22, m23
	#

	#
	# (0,0),(1,0) ... (6,0),(7,0)
	# (0,1),(1,1) ... (6,1),(7,1)
	#
	#		...			...
	#
	# (0,6),(1,6) ... (6,6),(7,6)
	# (0,7),(1,7) ... (6,7),(7,7)

	#
	#
	# 				aruco marker spacing
	#
	#

	h,w = image.shape[:2]

	#
	#				(8x8 imaginary grid)
	#

	#
	#	pixels per inch
	#
	#board_length = 32 #inches

	#ppi = int(min(h,w)/board_length)
	


	#
	#	birds eye aruco coordinates
	#

	#
	#	(0,0) ... (8,0) 
	#
	#	 ...	   ...
	#
	#	(0,8) ... (8,8)
	#

	# assumes 8x8 imaginary board
	#aruco_keypoints = np.array([[0,0], [8,0], [0,8]]) #, [8,8]]) # [m00, m11, m22, m33] points (clockwise!!!!)
	#multiplier = 8*ppi
	#be_aruco_keypoints_pixels = np.multiply(aruco_keypoints, multiplier) #pixels

	v = min(w,h)
	be_aruco_keypoints_pixels =np.array([[0,0], [v,0], [v,v], [0,v]]) # [m00, m11, m22] pixels
	#be_aruco_keypoints_pixels =np.array([[0,0], [v,0], [0,v]]) #, [v,v]]) # [m00, m11, m22] pixels

	#print(corners)
	#print(ids)

	#original_points = np.float32([list(top_left_aruco_point), list(top_right_aruco_point), list(bottom_left_aruco_point)]) #, list(bottom_right_aruco_point)])
	#birds_eye_points = np.float32([be_top_left_s, be_top_right_s, be_bottom_left_s])
	cleaned_corners = np.array(list([c[0].tolist() for c in corners]))
	sorted_indices = np.argsort(ids, axis=0)

	#cleaned_corners[sorted_indices]
	marker_corners_sorted = cleaned_corners[sorted_indices] #[::-1]

	# tmp swap 2 and 3
	marker_corners_sorted[[3,2]] = marker_corners_sorted[[2,3]]
	#input_seq[[ix1, ix2]] = input_seq[[ix2, ix1]]

	original_keypoints = np.zeros((4,2))
	for i, marker_corner in enumerate(marker_corners_sorted):
		original_keypoints[i] = marker_corner[0,i]


	# take only 0,1,---,3
	#birds_eye_keypoints = np.float32(be_aruco_keypoints_pixels[[0,1,3]])
	#original_keypoints =np.float32(original_keypoints[[0,1,3]])

	# take only tl, tr, br (first 3)
	#original_keypoints = np.float32(original_keypoints[:3,:])
	#birds_eye_keypoints = np.float32(be_aruco_keypoints_pixels[:3,:])

	# take all
	original_keypoints = np.float32(original_keypoints)
	birds_eye_keypoints = np.float32(be_aruco_keypoints_pixels)



	#print(birds_eye_keypoints)
	#print(original_keypoints)

	#M = cv2.getAffineTransform(original_keypoints,birds_eye_keypoints)
	#image = cv2.warpAffine(image,M,(w,h))

	# new
	M = cv2.getPerspectiveTransform(original_keypoints,birds_eye_keypoints)
	image = cv2.warpPerspective(image,M,(w,h))

	print(M)

	#return corners, ids
	# tmp
	return image


if __name__ == '__main__':

	input_image_dict = load_images_from_template(input_ftemplate)   # originals
	output_image_dict = input_image_dict.copy()       

	for k,v in input_image_dict.items():

		# tmp
		gray = detect_aruco_markers(v['cv2 image'], aruco_dict)
		output_image_dict[k]['cv2 image'] = gray


	write_image_dict(output_image_dict, output_ftemplate)

"""
cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	#print(frame.shape) #480x640
	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
	parameters =  aruco.DetectorParameters_create()

	#print(parameters)

	'''    detectMarkers(...)
		detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
		mgPoints]]]]) -> corners, ids, rejectedImgPoints
		'''
		#lists of ids and the corners beloning to each id
	corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
	print(corners)

	#It's working.
	# my problem was that the cellphone put black all around it. The alrogithm
	# depends very much upon finding rectangular black blobs

	gray = aruco.drawDetectedMarkers(gray, corners)

	#print(rejectedImgPoints)
	# Display the resulting frame
	cv2.imshow('frame',gray)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""
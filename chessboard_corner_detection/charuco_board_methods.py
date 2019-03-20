#!/usr/bin/env python

"""

	ChAruco Board Creation and Detection

  1.) Create charuco board image and config file

  2.) Load CharucoBoard object from config file

  3.) Detect corners

  4.) Detect pose

  5.) Find rotation and translation vectors

  6.) Calibrate camera


	TODO: use marker images to calibrate camera in single step

	Useful methods from CharucoBoard class

	# retrieving board dimensions
    squares_x, squares_y = cv2.aruco_CharucoBoard.getChessboardSize(board)

    # retrieving marker length
    marker_length = cv2.aruco_CharucoBoard.getMarkerLength(board)

    #
    square_length = cv2.aruco_CharucoBoard.getSquareLength(board)

"""

__author__ = "l.j. Brown"
__version__ = "1.0.1"

# imports

# internal
import logging
import os

# external
import numpy as np
import cv2
import cv2.aruco as aruco

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings

CHARUCO_BOARD_SETTINGS_FILE = "aruco_markers/charuco_board_settings.json" 		# Do not change
charuco_board_image_file = "aruco_markers/charuco_board.png"


def write_charuco_board_config_file(CHARUCO_BOARD_SETTINGS_FILE, squaresX, squaresY, squareLength, markerLength,
                                    charuco_board_image_file, aruco_dict_num, outSize, marginSize, borderBits):
    """

    :param CHARUCO_BOARD_SETTINGS_FILE: file storage file for cv2.FileStorage
    :param squaresX: number of chessboard squares in X direction.
    :param squaresY: number of chessboard squares in Y direction.
    :param squareLength: chessboard square side length (usually meters).
    :param markerLength: aruco marker side length (same unit than squareLength).
    :param charuco_board_image_file: image file containing charuco board.
    :param aruco_dict_num:  Enum number for call for: dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_num).
    :param outSize: size of the output image in pixels.
    :param marginSize: minimum margins (in pixels) of the board in the output image.
    :param borderBits: width of the marker borders.
    """
    # file storage
    config_writer = cv2.FileStorage(CHARUCO_BOARD_SETTINGS_FILE, cv2.FILE_STORAGE_WRITE)

    # number of chessboard squares in X direction
    config_writer.writeComment('squaresX - number of chessboard squares in X direction')
    config_writer.write('squaresX', squaresX)

    # number of chessboard squares in Y direction
    config_writer.writeComment('squaresY - number of chessboard squares in Y direction')
    config_writer.write('squaresY', squaresY)

    # chessboard square side length (usually meters)
    config_writer.writeComment('squareLength - chessboard square side length (usually meters)')
    config_writer.write('squareLength', squareLength)

    # aruco marker side length (same unit than squareLength)
    config_writer.writeComment('markerLength - aruco marker side length (same unit than squareLength)')
    config_writer.write('markerLength', markerLength)

    # image file containing charuco board
    config_writer.writeComment('charuco_board_image_file - image file containing charuco board')
    config_writer.write('charuco_board_image_file', charuco_board_image_file)

    # Enum number for call for: dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_num)
    # dictionary of markers indicating the type of markers,  The first markers in the dictionary are
    # used to fill the white chessboard squares.
    aruco_dict_num_comment = """
        aruco_dict_num - Enum number for call for: 
            dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_num)
        dictionary of markers indicating the type of markers,  The first markers in the dictionary are 
        used to fill the white chessboard squares.
    """
    config_writer.writeComment(aruco_dict_num_comment)
    config_writer.write('aruco_dict_num', aruco_dict_num)

    # size of the output image in pixels.
    config_writer.writeComment('outSize - size of the output image in pixels.')
    config_writer.write('outSize', outSize)

    # minimum margins (in pixels) of the board in the output image.
    config_writer.writeComment('marginSize - minimum margins (in pixels) of the board in the output image.')
    config_writer.write('marginSize', marginSize)

    # width of the marker borders.
    config_writer.writeComment('borderBits - width of the marker borders.')
    config_writer.write('borderBits', borderBits)

    # release file writer
    config_writer.release()

def load_charuco_board(CHARUCO_BOARD_SETTINGS_FILE):
    """
    Load charuco board settings file and return CharucoBoard object configured accordingly.
    :param CHARUCO_BOARD_SETTINGS_FILE:
    :return: CharucoBoard object
    """

    # read aruco_dict_num, squaresX, squaresY, squareLength, and markerLength from config file
    config_reader = cv2.FileStorage(CHARUCO_BOARD_SETTINGS_FILE, cv2.FileStorage_READ)
    assert config_reader.isOpened()

    aruco_dict_num = int(config_reader.getNode('aruco_dict_num').real())
    squaresX = int(config_reader.getNode('squaresX').real())
    squaresY = int(config_reader.getNode('squaresY').real())
    squareLength = float(config_reader.getNode('squareLength').real())
    markerLength = float(config_reader.getNode('markerLength').real())

    # release reader
    config_reader.release()

    # load aruco board dictonary
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_num)

    # create board
    board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)

    # return the CharucoBoard object
    return board


def create_charuco_board(CHARUCO_BOARD_SETTINGS_FILE, squaresX, squaresY, squareLength, markerLength,
                         charuco_board_image_file, aruco_dict_num, outSize, marginSize, borderBits):
    """

    :param CHARUCO_BOARD_SETTINGS_FILE: file storage file for cv2.FileStorage
    :param squaresX: number of chessboard squares in X direction.
    :param squaresY: number of chessboard squares in Y direction.
    :param squareLength: chessboard square side length (usually meters).
    :param markerLength: aruco marker side length (same unit than squareLength).
    :param charuco_board_image_file: image file containing charuco board.
    :param aruco_dict_num:  Enum number for call for: dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_num).
    :param outSize: size of the output image in pixels.
    :param marginSize: minimum margins (in pixels) of the board in the output image.
    :param borderBits: width of the marker borders.
    :return: return created CharucoBoard object
    """
    # write charuco board settings file
    write_charuco_board_config_file(CHARUCO_BOARD_SETTINGS_FILE, squaresX, squaresY, squareLength, markerLength,
                         charuco_board_image_file, aruco_dict_num, outSize, marginSize, borderBits)

    # create CharucoBoard object from newly created settings file
    board = load_charuco_board(CHARUCO_BOARD_SETTINGS_FILE)

    # write the image by first loading image settings: outSize, marginSize, borderBits
    # Output image with the board. The size of this image will be outSize
    # and the board will be on the center, keeping the board proportions.
    config_reader = cv2.FileStorage(CHARUCO_BOARD_SETTINGS_FILE, cv2.FileStorage_READ)
    assert config_reader.isOpened()

    outSize = tuple(config_reader.getNode('outSize').mat().astype(int).reshape(1, -1)[0])
    marginSize = int(config_reader.getNode('marginSize').real())
    borderBits = int(config_reader.getNode('borderBits').real())

    # release file reader
    config_reader.release()

    # create image
    img = board.draw(outSize, marginSize, borderBits)

    # write the image to a file
    cv2.imwrite(charuco_board_image_file, img)

    # return CharucoBoard object
    return board


def detect_charuco_board_corners(image, CHARUCO_BOARD_SETTINGS_FILE):
    """
    :param image:
    :param CHARUCO_BOARD_SETTINGS_FILE:
    :return: ret, ch_corners, ch_ids of interpolated charuco board detected in image
    """

    # read aruco dict num from config file
    config_reader = cv2.FileStorage(CHARUCO_BOARD_SETTINGS_FILE, cv2.FileStorage_READ)
    assert config_reader.isOpened()
    aruco_dict_num = int(config_reader.getNode('aruco_dict_num').real())
    config_reader.release()

    # generate aruco dictonary
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_num)

    # aruco parameters TODO: find out what ever this means
    aruco_parameters = aruco.DetectorParameters_create()

    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect aruco marker corners
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary,
                                                          parameters=aruco_parameters)

    # if at least one corner was detected interpolate corners for more acurate result
    if (ids is not None):
        # load board from settings file
        board = load_charuco_board(CHARUCO_BOARD_SETTINGS_FILE)

        # interpolate corners
        ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
        return ret, ch_corners, ch_ids

    # otherwise
    logger.info("No corners Detected")
    return None, None, None


def calibrate_camera_with_charuco_board(CHARUCO_BOARD_SETTINGS_FILE, input_calibration_image_directory,
                                        output_calibration_data_file):
    """
    TODO:
    :param CHARUCO_BOARD_SETTINGS_FILE:
    :param input_calibration_image_directory:
    :param output_calibration_data_file:
    :return: rvec, tvec ... TODO:
    """
    # TODO:
    # The calibrateCameraCharuco() function will fill the cameraMatrix and distCoeffs arrays
    # with the camera calibration parameters. It will return the reprojection error obtained
    # from the calibration. The elements in rvecs and tvecs will be filled with the estimated
    # pose of the camera (respect to the ChArUco board) in each of the viewpoints.

    pass


def estimate_charuco_board_pose(image, CHARUCO_BOARD_SETTINGS_FILE, camera_matrix, dist_coefficients):
    """

    :param image:
    :param CHARUCO_BOARD_SETTINGS_FILE:
    :param camera_matrix:
    :param dist_coefficients:
    :return: retval, rvec, tvec - rotation and translation vector
    """

    # load ChAruco board object
    board = load_charuco_board(CHARUCO_BOARD_SETTINGS_FILE)

    # detect charuco board corners and ids
    retval, ch_corners, ch_ids = detect_charuco_board_corners(image, CHARUCO_BOARD_SETTINGS_FILE)

    if retval:
        # find rotation and translation vectors
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, board, camera_matrix, dist_coefficients)

        if retval:
            return retval, rvec, tvec

    logger.info("No corners Detected")
    return False, None, None


def estimate_camera_matrix(image, focal_length=None):
    size = image.shape

    # Camera internals (estimated)

    if focal_length is None:
        focal_length = size[1]

    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    return camera_matrix


def estimate_distortion_coefficients():
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    return dist_coeffs


def write_camera_intrinsics(camera_intrinsics_file, camera_matrix, dist_coeffs):
    """

    :param camera_intrinsics_file:
    :param camera_matrix:
    :param dist_coeffs:
    """
    # create output directory if it does not exist
    if not os.path.exists(os.path.split(camera_intrinsics_file)[0]):
        os.makedirs(os.path.split(camera_intrinsics_file)[0])

    np.savez(camera_intrinsics_file, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def load_camera_intrinsics(camera_intrinsics_file):
    # Load previously saved data
    with np.load(camera_intrinsics_file) as X:
        camera_matrix, dist_coeffs = [X[i] for i in ('camera_matrix', 'dist_coeffs')]

    return camera_matrix, dist_coeffs


"""


# detection
#aruco_parameters = aruco.DetectorParameters_create()
#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#res = cv2.aruco.detectMarkers(gray,dictionary)

#corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict,
 parameters=aruco_parameters)

# if enough markers were detected
 # then process the board
# if( ids is not None ):
ret, ch_corners, ch_ids = aruco.interpolateCornersCharuco(corners, ids, gray, charuco_board)

# if there are enough corners to get a reasonable result
if (ret > 5):
    aruco.drawDetectedCornersCharuco(frame, ch_corners, ch_ids, (0, 0, 255))
retval, rvec, tvec = aruco.estimatePoseCharucoBoard(ch_corners, ch_ids, charuco_board, camera_matrix, dist_coeffs)

# if a pose could be estimated
if (retval):
    frame = aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.032)

#The calibrateCameraCharuco() function will fill the cameraMatrix and distCoeffs arrays
# with the camera calibration parameters. It will return the reprojection error obtained
# from the calibration. The elements in rvecs and tvecs will be filled with the estimated
# pose of the camera (respect to the ChArUco board) in each of the viewpoints.

# cv2.estimatePoseCharucoBoard(charucoCorners, charucoIds, cameraMatrix, distCoeffs)
# returns rvec, tvec
"""
# Program

if __name__ == "__main__":

    squaresX = 8
    squaresY = 5
    squareLength = 0.0311
    markerLength = 0.01571
    aruco_dict_num = cv2.aruco.DICT_4X4_50
    outSize = (200 * squaresX, 200 * squaresY)
    marginSize = 0
    borderBits = 1

    # create charuco board image and config file
    board = create_charuco_board(CHARUCO_BOARD_SETTINGS_FILE, squaresX, squaresY, squareLength, markerLength,
                         charuco_board_image_file, aruco_dict_num, outSize, marginSize, borderBits)

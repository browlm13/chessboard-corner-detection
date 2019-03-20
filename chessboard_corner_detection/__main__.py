#!/usr/bin/env python

"""

    Create annotated data set for semantic segmentation training using charuco board and a video.

"""

__author__ = "l.j. Brown"
__version__ = "1.0.1"

#
#
#									imports	
#
#

# internal
import logging

# external
# import glob
import numpy as np
import cv2


# my lib
import frames
import image_directory_handler
import charuco_board_methods as charuco_detector
import graphics
import user_interface as ui
#
#
#									Settings
#
#

CHARUCO_BOARD_SETTINGS_FILE = "aruco_markers/charuco_board_settings.json"  # Do not change
CAMERA_INTRINSICS_FILE = "calibration_data/camera_intrinsics.npz"

MOVIE_NUM = 1
input_video_path = "input_video/%s.mov" % MOVIE_NUM
raw_frames_directory_path = "raw_charuco_board_frames_video_%s" % MOVIE_NUM
edited_frames_directory_path = "edited_charuco_board_frames_video_%s" % MOVIE_NUM
output_video_path = "output_video/%s.mov" % MOVIE_NUM

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Program
if __name__ == '__main__':

    # TODO: add calibration step

    # convert input video into individual frames
    # frames.to_frames_directory(input_video_path, raw_frames_directory_path)

    # load camera intrinsics from saved calibration data
    camera_matrix, dist_coeffs = charuco_detector.load_camera_intrinsics(CAMERA_INTRINSICS_FILE)

    # find rvec_tvec_frame_map from input image directory
    rvec_tvec_frame_map = image_directory_handler.charuco_image_directory_rvec_tvec_frame_map(raw_frames_directory_path,
                                                            CHARUCO_BOARD_SETTINGS_FILE, camera_matrix, dist_coeffs)

    # get frame list and filtered frame list (only frames where charuco board pose was found)
    frame_list = image_directory_handler.frame_number_list(raw_frames_directory_path)
    filtered_frame_list = [k for k, v in rvec_tvec_frame_map.items() if v['rvec'] is not None]


    # have user select two points...
    ui.gui(raw_frames_directory_path, filtered_frame_list, rvec_tvec_frame_map, camera_matrix, dist_coeffs)

    """
    # temporary define model points and colors
    model_point_cloud = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.25, 0.17, 0.0], [0.0, 0.17, 0.0]])  # floats
    color_descriptor_matrix = np.array([(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255, 0)])  # ints

    # draw model points
    graphics.draw_colored_point_cloud_on_image_directory(raw_frames_directory_path, edited_frames_directory_path,
                            camera_matrix, dist_coeffs, rvec_tvec_frame_map, model_point_cloud, color_descriptor_matrix)
    """

    """
    # get sorted frame numbers list of raw (unedited) image files
    frame_numbers_list = image_directory_handler.frame_number_list(raw_frames_directory_path)

    # loop through frames
    for frame_number in frame_numbers_list:

        # load image
        frame_image = image_directory_handler.load_frame(raw_frames_directory_path, frame_number)

        # temporarily estimate camera_matrix and dist_coefficients
        camera_matrix = estimate_camera_matrix(frame_image)
        dist_coeffs = estimate_distortion_coefficients()

        # check charuco board rvec and tvec method
        retval, rvec, tvec = charuco_detector.estimate_charuco_board_pose(frame_image, CHARUCO_BOARD_SETTINGS_FILE,
                                                                  camera_matrix, dist_coeffs)

        if retval:
            # frame_image = cv2.aruco.drawAxis(frame_image, camera_matrix, dist_coefficients, rvec, tvec, 0.032)
            model_point_cloud = np.array([[0.0, 0.0,0.0], [0.25, 0.0,0.0], [0.25, 0.17,0.0], [0.0, 0.17,0.0]])  # floats
            color_descriptor_matrix = np.array([(0, 0, 255), (0, 255, 255), (255, 0, 0), (0, 255,0 )])          # ints


            frame_image = graphics.draw_colored_point_cloud(frame_image, CHARUCO_BOARD_SETTINGS_FILE, camera_matrix, dist_coeffs,
                                              model_point_cloud, color_descriptor_matrix)
            # frame_image = draw_colored_point_cloud()

            # write image
            image_directory_handler.write_frame(edited_frames_directory_path, frame_number, frame_image)
    """

    # TODO: Annotate frames directory
    # TODO: write test images to check bounding polygon

    # convert output frames directory to video file
    # frames.to_video(edited_frames_directory_path, output_video_path)
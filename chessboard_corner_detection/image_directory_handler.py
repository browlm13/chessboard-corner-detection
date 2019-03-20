#!/usr/bin/env python

"""

    Methods to assist reading and writing to image directories.

    usage:
        # get sorted frame numbers list of input image frame files
        frame_numbers_list = image_directory_handler.frame_number_list(input_frames_directory_path)

        # loop through frames
        for frame_number in frame_numbers_list:
            image = image_directory_handler.load_frame(input_frames_directory_path, frame_number)
            image_directory_handler.write_frame(output_frames_directory_path, frame_number, image)


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
import os


# external
import glob
import numpy as np
import numpy.ma as ma
import cv2


# my lib
import frames
import charuco_board_methods as charuco_detector

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# methods


def load_frame(input_image_directory, frame_number):
    """

    :param input_image_directory: directory containing frames labeled frame_number.*, where * is the extension.
    :param frame_number: The number of the desired frame.
    :return: numpy image array or None
    """
    frame_glob_ftemplate = input_image_directory + '/%s.*' % frame_number
    matching_images = glob.glob(frame_glob_ftemplate)[0]

    if type(matching_images) == str:
        logger.info(matching_images)
        return cv2.imread(matching_images)

    else:
        logger.warning("Error Matching frame: %s, for input directory: %s" % (frame_number, input_image_directory))
        return None


def write_frame(output_image_directory, frame_number, image, extension='jpg', overwrite=True):
    """

    :param output_image_directory: directory of output frames to be labeled 1.ext->n.ext.
    :param frame_number: number to write.
    :param image: image to write.
    :param extension: image file extension, default \'jpg'.
    :param overwrite: whether or not to overwrite existing frames with same path, default True.
    """

    fpath = '%s/%s.%s' % (output_image_directory, frame_number, extension)

    # check if output directory exists, if it doesnt create it
    if not os.path.exists(output_image_directory):
        os.makedirs(output_image_directory)

    # if overwrite is False, ensure fpath does not alread exist
    if not overwrite:
        assert not os.path.exists(fpath)

    # write image
    cv2.imwrite(fpath, image)


def frame_number_list(input_image_directory):
    """

    :param input_image_directory:
    :return: list of frames as integers sorted
    """

    frame_glob_ftemplate = input_image_directory + '/[0-9]*.*'
    matching_images = glob.glob(frame_glob_ftemplate)

    get_frame_number = lambda full_path: int(os.path.splitext(os.path.split(full_path)[1])[0])
    frame_numbers_list = list(map(get_frame_number, matching_images))

    # sort frame numbers
    frame_numbers_list.sort()

    return frame_numbers_list


def charuco_image_directory_rvec_tvec_frame_map(input_image_directory, charuco_board_settings_file,
                                                              camera_matrix, dist_coeffs):
    """

    :param input_image_directory:
    :param charuco_board_settings_file:
    :param camera_matrix:
    :param dist_coeffs:
    :return: rvec_tvec_frame_map, format {frame_number : {rvec: rvec, tvec:tvec}, ...} - None if not found
    """

    # get sorted frame numbers list of raw (unedited) image files
    frame_numbers_list = frame_number_list(input_image_directory)

    # create empty dictionary for rvec_tvec_frame_map
    rvec_tvec_frame_map = {}

    # loop through frames
    for frame_number in frame_numbers_list:
        # load image
        frame_image = load_frame(input_image_directory, frame_number)

        # check charuco board rvec and tvec method
        retval, rvec, tvec = charuco_detector.estimate_charuco_board_pose(frame_image, charuco_board_settings_file,
                                                                          camera_matrix, dist_coeffs)

        rvec_tvec_frame_map[frame_number] = {
            'rvec' : rvec,
            'tvec' : tvec
        }

    return rvec_tvec_frame_map
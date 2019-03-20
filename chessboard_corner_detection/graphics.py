#!/usr/bin/env python

"""
        Graphics
"""

__author__ = "l.j. Brown"
__version__ = "1.0.1"

# imports

# internal
import logging
import random
import math
import copy

# external
import numpy as np
import cv2

# my lib
import charuco_board_methods as charuco_detector
import image_directory_handler

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Methods

def draw_colored_point_cloud(image, camera_matrix, dist_coeffs, rvec, tvec, model_point_cloud,
                             color_descriptor_matrix, return_retval=False):
    """
    draw the colored point cloud on a single image by projecting the model points.
    :param image: Image to draw colored point cloud on.
    :param camera_matrix:
    :param dist_coeffs:
    :param rvec:
    :param tvec:
    :param model_point_cloud:
    :param color_descriptor_matrix:
    :param return_retval:
    :return: modified image
    """

    # copy image
    image_copy = np.copy(image)

    if rvec is not None and tvec is not None:
        image_points, _ = cv2.projectPoints(model_point_cloud, rvec, tvec, camera_matrix,
                                            dist_coeffs)
        image_points = image_points.astype(int)

        for p in range(image_points.shape[0]):
            center = tuple(image_points[p][0])
            point_color = tuple(color_descriptor_matrix[p].tolist())

            radius = 2
            line_thickness = 15

            # draw point
            cv2.circle(image_copy, center, radius, color=point_color, thickness=line_thickness)

    return image_copy

def draw_colored_point_cloud_charuco(image, charuco_board_settings_file, camera_matrix, dist_coeffs, model_point_cloud,
                             color_descriptor_matrix, return_retval=False):
    """
    draw the colored point cloud on a single image after determining the pose and then projecting the model points.
    :param image: Image to draw colored point cloud on.
    :param charuco_board_settings_file:
    :param model_point_cloud: point cloud (float) using model coordinates. \
        Columns correspond to x,y,z, and rows to points.
    :param color_descriptor_matrix: Columns correspond to b,g,r
    :param return_retval: Boolean set to false, whether or not to return if board pose was sucsessful/points were drawn.
    :return: modified image
    """
    # check charuco board rvec and tvec method
    retval, rvec, tvec = charuco_detector.estimate_charuco_board_pose(image, charuco_board_settings_file,
                                                                      camera_matrix, dist_coeffs)

    # copy image
    image_copy = np.copy(image)

    if retval:
        image_points, _ = cv2.projectPoints(model_point_cloud, rvec, tvec, camera_matrix,
                                            dist_coeffs)
        image_points = image_points.astype(int)

        for p in range(image_points.shape[0]):
            center = tuple(image_points[p][0])
            point_color = tuple(color_descriptor_matrix[p].tolist())

            radius = 2
            line_thickness = 15

            # draw point
            cv2.circle(image_copy, center, radius, color=point_color, thickness=line_thickness)

    if return_retval:
        return image_copy, retval

    return image_copy


def draw_colored_point_cloud_on_image_directory(input_image_directory, output_image_directory, camera_matrix,
                                dist_coeffs, rvecs_tvecs_frame_map, model_point_cloud, color_descriptor_matrix):
    """
    draw the colored point cloud on every image in a directory after determining pose and projecting the model points.
    :param input_image_directory: Image directory.
    :param output_image_directory: Image directory.
    :param camera_matrix:
    :param dist_coeffs:
    :param rvecs_tvecs_frame_map: dictonary of frame number to rvecs, tvecs matrices or None
    :param model_point_cloud: point cloud (float) using model coordinates. \
        Columns correspond to x,y,z, and rows to points.
    :param color_descriptor_matrix: Columns correspond to b,g,r
    """
    # get sorted frame numbers list of raw (unedited) image files
    frame_numbers_list = image_directory_handler.frame_number_list(input_image_directory)

    # loop through frames
    for frame_number in frame_numbers_list:
        # load image
        frame_image = image_directory_handler.load_frame(input_image_directory, frame_number)

        # get corresponding transformation vectors
        rvec = rvecs_tvecs_frame_map[frame_number]['rvec']
        tvec = rvecs_tvecs_frame_map[frame_number]['tvec']

        if rvec is not None and tvec is not None:
            # draw colored point cloud on image
            frame_image = draw_colored_point_cloud(frame_image, camera_matrix, dist_coeffs, rvec, tvec,
                                                   model_point_cloud, color_descriptor_matrix)

        # write image
        image_directory_handler.write_frame(output_image_directory, frame_number, frame_image)


def draw_colored_point_cloud_on_charuco_image_directory(input_image_directory, output_image_directory,
                charuco_board_settings_file, camera_matrix, dist_coeffs, model_point_cloud, color_descriptor_matrix):
    """
    draw the colored point cloud on every image in a directory after determining pose and projecting the model points.
    :param input_image_directory: Image directory.
    :param output_image_directory: Image directory.
    :param charuco_board_settings_file:
    :param model_point_cloud: point cloud (float) using model coordinates. \
        Columns correspond to x,y,z, and rows to points.
    :param color_descriptor_matrix: Columns correspond to b,g,r
    """
    # get sorted frame numbers list of raw (unedited) image files
    frame_numbers_list = image_directory_handler.frame_number_list(input_image_directory)

    # loop through frames
    for frame_number in frame_numbers_list:
        # load image
        frame_image = image_directory_handler.load_frame(input_image_directory, frame_number)

        # draw colored point cloud on image
        frame_image = draw_colored_point_cloud_charuco(frame_image, charuco_board_settings_file, camera_matrix, dist_coeffs,
                                               model_point_cloud, color_descriptor_matrix)

        # write image
        image_directory_handler.write_frame(output_image_directory, frame_number, frame_image)


def draw_box_legs(B1, B2, input_image, color=None, lineThickness = 4):
    # B - 4x2 matrix following clockwise convention
    np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

    if color is None:
        color = (0,255,0)

    for i in range(4):
        start_coor = B1[i,:]
        end_coor = B2[i,:]

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


def draw_point_matrix(image_points, image, color=None, lineThickness=15):
    # Draw nx2 point matrix onto image and return image
    np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

    for i in range(image_points.shape[0]):
        center = np_2_tupple_int(image_points[i])
        radius = 1
        if color is None:
            color = (0,0,255)
        cv2.circle(image, center, radius, color, lineThickness, lineType=8, shift=0)

    return image


def draw_triangle(T, input_image, color=None, line_thickness=4):
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
        cv2.line(input_image, np_2_tupple_int(start_coor), np_2_tupple_int(end_coor), color, line_thickness)

    return input_image

def draw_box_diagnols(B, input_image, color=None, line_thickness=4):
    # B - 4x2 matrix following clockwise convention
    np_2_tupple_int = lambda t: tuple(int(e) for e in tuple(t))

    if color is None:
        color = (0,255,0)

    # draw positive diagnol
    cv2.line(input_image, np_2_tupple_int(B[3]), np_2_tupple_int(B[1]), color, line_thickness)

    # draw negitive diagnol
    cv2.line(input_image, np_2_tupple_int(B[0]), np_2_tupple_int(B[2]), color, line_thickness)

    return input_image

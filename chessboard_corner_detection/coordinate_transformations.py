#!/usr/bin/env python

"""
        Pinhole Model Coordinate Transformations
"""

__author__ = "l.j. Brown"
__version__ = "1.0.1"

# imports

# internal
import os
import logging
import random
import math
import copy

# external
import glob
import pandas as pd
import numpy as np
import cv2
import cv2.aruco as aruco
from scipy import stats

# my lib
import aruco_marker_detection as marker_detector
import frames

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Methods


def solve_pnp(model_points, image_points, camera_matrix, distortion_coefficients):
    """
    :param model_points:
    :param image_points:
    :param camera_matrix:
    :param distortion_coefficients:
    :return: rotation_vector,
    :return: translation_vector
    """
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                          distortion_coefficients)

    return rotation_vector, translation_vector


def model_camera_matrices_from_images_points(model_points, image_points, camera_matrix, distortion_coefficients):
    """
    TODO: update for charuco board
    Solve for the homogeneous model / camera transformation matrices using image points and pnp solve
    as this is the most common starting point.

    :param model_points: data type float
    :param image_points: data type float, matching dimensions to model_points
    :param camera_matrix: 3x3 intrinsic camera matrix, needed for cv2.solvePnP
    :param distortion_coefficients:
    :return: M2C: 4x4 homogeneous transformation matrix M2C, model to camera transformation matrix
    :return: C2M: 4x4 homogeneous transformation matrix C2M, camera to model transformation matrix
    """
    # find rotation and translation vectors
    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                          distortion_coefficients)

    # obtain 3x3 rotation matrix, model to camera rotation, from Rodrigues rotation vector
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # obtain 3x3 rotation matrix, camera to model rotation,from the inverse rotation matrix which is equal
    # to its transpose
    inverse_rotation_matrix = rotation_matrix.T

    # obtain inverse translation vector, for camera to model translation, using -R.T*t.
    # Where t is the original translation vector
    inverse_translation_vector = np.negative(rotation_matrix).T.dot(translation_vector)

    # combine rotation matrix and translation vector into 4x4 homogeneous transformation matrix M2C
    M2C = np.concatenate((rotation_matrix, translation_vector), axis=1)   # 3x4
    # TODO: add row 0,0,0,1
    # M2C = np.concatenate((M2C, np.array([0,0,0,1])), axis=0)   # 4x4

    # combine inverse rotation matrix and inverse translation vector into 4x4 homogeneous transformation matrix C2M
    C2M = np.concatenate((inverse_translation_vector, inverse_translation_vector), axis=1)   # 3x4
    # TODO: add row 0,0,0,1
    # C2M = np.concatenate((C2M, np.array([0,0,0,1])), axis=0)   # 4x4

    # note: transformation matrices are homogeneous
    return M2C, C2M

# should have single method that solves for M2C, C2M, C2P, M2P. Note: there is no P2C or P2M as z information is lost

def inverse_affine_transformation_matrix(A):
    """
    A is an nxn affine transformation matrix.
        |R|T|           |R_inv|-R_inv*T|
    A = |---|,  A_inv = |--------------|
        |0|1|           |  0  |    1   |

    :param A: an nxn affine transformation matrix.
    :return: the nxn inverse affine transformation matrix
    """
    n, n = A.shape

    # extract components R, an n-1xn-1 linear transformation matrix, and T, an nx1 translation matrix
    R = A[:n-1, :n-1]
    T = A[:n-1, n-1]

    # find R^-1
    R_inv = np.linalg.inv(R)

    # Find A^-1/A_inv
    A_inv = np.copy(A).astype(float)        # copy A for base of A^-1 matrix and ensure it is of data type float
    A_inv[:n-1, :n-1] = R_inv               # set top left nxn sub matrix equal to R^-1
    A_inv[:n-1, n-1] = np.negative(R_inv.dot(T))    # place -R^-1*T in top right corner

    return A_inv


def inverse_rvec_and_tvec(rvec, tvec):
    """
    :param rvec: euler-rodrigues vector for rotation axis
    :param tvec: tranlation vector
    :return: inverse rvec and inverse tvec
    """
    rmat = cv2.Rodrigues(rvec)[0]           # convert rvec to rotation matrix
    rmat_inv = rmat.T
    rvec_inv = cv2.Rodrigues(rmat_inv)[0]       # convert rotation matrix to  euler-rodrigues vector
    tvec_inv = np.negative(rmat_inv.dot(tvec)) # -R^-1*tvec

    return rvec_inv, tvec_inv


def image_points_to_camera_vectors(image_points, camera_matrix, dist_coeffs):
    """

    :param image_points:
    :return: camera_vectors
    """

    # image_points_h = cv2.ConvertPointsHomogeneous(image_points) # converts points to/from homogeneous
    # K_inv = np.linalg.inv(camera_matrix)
    # cv2.undistortPoints() #function to get undistorted normalized point (x', y')

    image_points = np.array(image_points)
    num_pts = image_points.size / 2
    image_points.shape = (int(num_pts), 1, 2)

    # TODO: undistort!!!
    # image_points = cv2.undistortPoints(image_points, camera_matrix, dist_coeffs)
    # tmp ***
    pts_3d = cv2.convertPointsToHomogeneous(np.float32(image_points))
    pts_3d.shape = (int(num_pts),3)

    # tmp
    K_inv = np.linalg.inv(camera_matrix)
    pts_3d = K_inv.dot(pts_3d.T)

    #return pts_3d
    return pts_3d.T

# cv2.composeRT(rvec1, tvec1, rvec2, tvec2) # Combines two rotation-and-shift transformations, returns rvec3, tvec3


# rmat = cv2.Rodrigues(rvec)[0] #  convert rvec to rotation matrix

# cv2.composeRT(rvec1, tvec1, rvec2, tvec2) # Combines two rotation-and-shift transformations, returns rvec3, tvec3
# rvec3 = rodrigues−1(rodrigues(rvec2) · rodrigues(rvec1))
# tvec3 = rodrigues(rvec2) · tvec1 + tvec2

# cv2.convertPointsToHomogeneous(euclidean_points) # Converts points from Euclidean to homogeneous space.
# cv2.convertPointsFromHomogeneous(homogeneous_points) # Converts points from homogeneous to Euclidean space.
# The function converts points homogeneous to Euclidean space using perspective projection.
# That is, each point (x1, x2, ... x(n-1), xn) is converted to (x1/xn, x2/xn, ..., x(n-1)/xn).
# When xn=0, the output point coordinates will be (0,0,0,...).
# cv2.ConvertPointsHomogeneous(src, dst) # converts points to/from homogeneous

# cv2.decomposeProjectionMatrix(..) # Decomposes a projection matrix into a rotation matrix and a camera matrix

# cv2.findFundamentalMat(...

# cv2.undistortPoints function to get undistorted normalized point (x', y')
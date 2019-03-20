#!/usr/bin/env python

"""
        Maths
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
import pandas as pd
import numpy as np
from scipy import stats

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Methods

def normal_vector_of_plane(plane_points_pqr):
    """
    Find the vector that is normal to the plane formed by the given points.

        equation for the plane formed by points p,q, and r:
            a(x - px) + b(y - py) + c(z - pz) = 0

        n = <a,b,c>
        where n is a vector perpendicular to the plane formed by points p,q and r.

    Solving for the normal vector n:
        n = pq x pr,  where x is the cross product.
        pq = <qx - px, qy - py, qz - pz>
        pr = <rx - px, ry - py, rz - pz>

    :param plane_points_pqr: 3x3 numpy array where columns correspond to x, y, z
                            \   and rows correspond to points p,q,r on the plane respectively,
                            \   "np.array([[px, py, pz], [qx, qy, qz], [rx, ry, rz]])".
    :return: A vector, n, that is normal to the plane formed by points p,q and r.
    """

    # unpack row points
    p, q, r = plane_points_pqr

    # find pq and pr
    pq = np.subtract(q, p)
    pr = np.subtract(r, p)

    # find n, n = pq x pr
    n = np.cross(pq, pr)

    return n


def line_plane_intersection(line_direction, line_point, plane_normal, plane_point):
    """
    Find the intersection point of the line with a point o and direction d
            \ and the plane with normal vector n and point p.

        n . (o + td -p) = 0
    =>  [n . td] + [n . (o -p)] = 0
    =>  t(n . d) + [n . (o - p)] = 0
    =>  t =  [ n . (o - p) ] / (n . d)

    Solve for t:

        t =  -[ n . (o - p) ] / (n . d)

    w ~ o - p
    numerator ~ -(n . diff)
    denominator ~ n . d

    intersection_point = l(t) = o + dt

    :param line_direction:
    :param line_point:
    :param plane_normal:
    :param plane_point:
    :return: intersection point
    """

    # n . d, parallel if the dot product is 0
    denominator = line_direction.dot(plane_normal)
    if denominator == 0: raise RuntimeError("Line does not intersect plane")

    # w = (o - p)
    w = line_point - plane_point

    # n . w
    numerator = np.negative(w.dot(plane_normal))

    # t =  -(n . w) / (n . d)
    t = np.divide(numerator, denominator)

    # substitute back into equation for line to find intersection point
    intersection_point = line_point + t*line_direction

    return intersection_point


def closest_point_between_two_rays(r1_direction, r1_point, r2_direction, r2_point):
    """
    Find the closest point between two rays, if they intersect return their intersection point. If the two rays are
    parallel TODO: what then?

    reference: https://math.stackexchange.com/questions/1036959/midpoint-of-the-shortest-distance-between-2-rays-in-3d

    r1(t) = r1o + tr1d
    r2(s) = r2o + sr2d

    t = [((r2o - r1o) . r1d)(r2d . r2d) + ((r1o - r2o) . r2d)(r1d . r2d)]/[(r1d . r1d)(r2d . r2d) -(r1d . r2d)^2]
    s = [((r1o - r2o) . r2d)(r1d . r1d) + ((r2o - r1o) . r1d)(r1d . r2d)]/[(r1d . r1d)(r2d . r2d) -(r1d . r2d)^2]

    m = [(r1o + r1d*t) + (r2o + r2d*s)]/2, where m is the midpoint.

    :param r1_direction:
    :param r1_point:
    :param r2_direction:
    :param r2_point:
    :return:
    """

    # check if rays are parallel (r1d . r2d)
    if np.dot(r1_direction, r2_direction) == 0:
        raise RuntimeError("Rays are parallel.")

    # (r1d . r1d)(r2d . r2d) -(r1d . r2d)^2
    denominator = np.dot(r1_direction, r1_direction)*np.dot(r2_direction, r2_direction)\
                  -np.dot(r1_direction, r2_direction)**2

    # ((r2o - r1o) . r1d)(r2d . r2d)
    t_numerator_lhs = np.subtract(r2_point,r1_point).dot(r1_direction) * np.dot(r2_direction, r2_direction)
    # ((r1o - r2o) . r2d)(r1d . r2d)
    t_numerator_rhs = np.subtract(r1_point,r2_point).dot(r2_direction) * np.dot(r1_direction, r2_direction)
    # [((r2o - r1o) . r1d)(r2d . r2d) + ((r1o - r2o) . r2d)(r1d . r2d)]
    t_numerator = t_numerator_lhs + t_numerator_rhs
    t = t_numerator/denominator

    # ((r1o - r2o) . r2d)(r1d . r1d)
    s_numerator_lhs = np.subtract(r1_point,r2_point).dot(r2_direction) * np.dot(r1_direction, r1_direction)
    # ((r2o - r1o) . r1d)(r1d . r2d)
    s_numerator_rhs= np.subtract(r2_point,r1_point).dot(r1_direction) * np.dot(r1_direction, r2_direction)
    # [((r1o - r2o) . r2d)(r1d . r1d) + ((r2o - r1o) . r1d)(r1d . r2d)]
    s_numerator = s_numerator_lhs + s_numerator_rhs
    s = s_numerator/denominator

    # find midpoint
    # [(r1o + r1d*t) + (r2o + r2d*s)]/2
    m = (r1_point + r1_direction*t + r2_point + r2_direction*s)/2

    return m


def r2_lines_intersection_point_from_coefficients(coefficients_line_1, coefficients_line_2):

    """
    Find the point of intersection between two lines in R2 given their coefficients.

    Format for coefficients of lines:
        [slope, y-intercept]

    Intersection point:
        [x,y]

    Equation is true at point of intersection:
        y=ax+c=bx+d.

    rearrange to extract value of x
        ax-bx=d-c
        x=(d-c)/(a-b)

    Substitute to find y value
        y=a(d-c)/(a-b) +c.

    point of intersection is:
        [(d-c)/(a-b),  (ad - bc)/(a-b)]

    :param coefficients_line_1: Python list [a,c], where a is the slope and c is the y-intercept.
    :param coefficients_line_2: Python list [b,d], where b is the slope and d is the y-intercept.
    :return: A python list containing the coordinates of the intersection point of the two line, "[x,y]".
    """

    a, c = coefficients_line_1
    b, d = coefficients_line_2

    intersection_point = [int((d-c)/(a-b)), int((a*d - b*c)/(a-b))]

    return intersection_point


def regression_line_coefficients_from_data_points(xs, ys):
    """
    Returns the coefficients [slope, intercept] of the best fit line from given (xi,yi) pairs.
    :param xs: Python list of x values.
    :param ys: Corresponding python list of y values.
    :return: Regression [slope, intercept] of best fit line from points.
    """
    # perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, ys)

    # polynomial coefficients for best fit line
    p = [slope, intercept]

    return p

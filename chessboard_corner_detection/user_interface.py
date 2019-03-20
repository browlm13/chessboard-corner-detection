#!/usr/bin/env python

"""
    User Interface

        selected_points = chessboard_corner_selection(image)
        chessboard_corners_image1, chessboard_corners_image1 = stereo_images_chessboard_corner_selection(image1, image2)


"""
import copy
import time

import numpy as np
import cv2
from tkinter import *
from PIL import Image
from PIL import ImageTk
# import tkFileDialog
import tkinter.filedialog as tkFileDialog

# my lib
import image_directory_handler
import coordinate_transformations
import graphics



def load_image_cv2_to_tkinter(path):
    # load cv2 image
    image = cv2.imread(path)

    # OpenCV represents images in BGR order; however PIL represents
    # images in RGB order, so we need to swap the channels
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # convert the images to PIL format...
    image = Image.fromarray(image)

    # ...and then to ImageTk format
    image = ImageTk.PhotoImage(image)

    return image

# def stereo_images_chessboard_corner_selection(image1, image2):

# cv2.imshow(winname, mat)

class selection_window:

    def select_chessboard_image_points(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the
        # (x, y) coordinates

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.chessboard_corners_image_points) < 4:
                self.chessboard_corners_image_points.append((x,y))
                global UPDATE_WINDOWS
                UPDATE_WINDOWS = True


    def get_chessboard_corners_image_points(self):
        return self.chessboard_corners_image_points

    def reset_chessboard_corners_image_points(self):
        self.chessboard_corners_image_points = []

    def tracker_moved(self, raw_value):
        # cv2.setTrackbarPos(self.tracker_name, self.window_name, closest_available_frame)
        # self.update_image(raw_value)
        pass

    def get_closest_available_value(self, raw_value):
        closest_available_frame = min(self.available_frames_list, key=lambda x: abs(x - raw_value))
        return closest_available_frame

    def get_selected_frame(self):
        raw_value = cv2.getTrackbarPos(self.tracker_name, self.window_name)
        frame_number = self.get_closest_available_value(raw_value)
        return frame_number

    def get_selected_image(self):
        #raw_value = cv2.getTrackbarPos(self.tracker_name, self.window_name)
        #frame_number = self.get_closest_available_value(raw_value)
        frame_number = self.get_selected_frame()
        selected_image = image_directory_handler.load_frame(self.image_directory, frame_number)
        return selected_image

    def update_available_frames(self, available_frames_list):
        self.available_frames_list = available_frames_list

    def deploy_window(self, tracker_value, window_size, window_position):
        min_frame = min(self.available_frames_list)
        max_frame = max(self.available_frames_list)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(self.tracker_name, self.window_name, min_frame, max_frame, self.tracker_moved)

        # set tracker slider value
        frame_number = self.get_closest_available_value(tracker_value)
        cv2.setTrackbarPos(self.tracker_name, self.window_name, frame_number)

        # resize and move window
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])
        cv2.moveWindow(self.window_name, window_position[0], window_position[1])

        # callback function
        cv2.setMouseCallback(self.window_name, self.select_chessboard_image_points)

        # show image
        # selected_image = image_directory_handler.load_frame(self.image_directory, frame_number)
        # cv2.imshow(self.window_name, selected_image)
        #k = cv2.waitKey(0)
        # cv2.waitKey(0)


    def __init__(self, window_name, image_directory, available_frames_list, start_frame, window_size, window_position,
                 chessboard_corners_image_points):
        self.update_available_frames(available_frames_list)
        self.image_directory = image_directory
        self.window_name = window_name
        self.tracker_name = self.window_name + '_tracker'
        self.chessboard_corners_image_points = chessboard_corners_image_points
        self.deploy_window(start_frame, window_size, window_position)


def draw_selected_points(input_image_directory, frame1, points1, frame2, points2, rvec_tvec_frame_map, camera_matrix, dist_coeffs):

    """
    # find c1 -> c2 rvec and tvec and c2 -> c1 rvec and tvec
    rvec1, tvec1 = rvec_tvec_frame_map[frame1]['rvec'], rvec_tvec_frame_map[frame1]['tvec']
    rvec_inv1, tvec_inv1 = coordinate_transformations.inverse_rvec_and_tvec(rvec1, tvec1)
    rvec2, tvec2 = rvec_tvec_frame_map[frame2]['rvec'], rvec_tvec_frame_map[frame2]['tvec']
    rvec_inv2, tvec_inv2 = coordinate_transformations.inverse_rvec_and_tvec(rvec2, tvec2)


    #rvec_1_to_2, tvec_1_to_2, jacobian = cv2.composeRT(rvec1, tvec1, rvec_inv2, tvec_inv2)
    #rvec_2_to_1, tvec_2_to_1, jacobian = cv2.composeRT(rvec2, tvec2, rvec_inv1, tvec_inv1)
    rt_1_to_2_bundel = cv2.composeRT(rvec2, tvec2, rvec_inv1, tvec_inv1)
    rvec_1_to_2, tvec_1_to_2 = rt_1_to_2_bundel[0], rt_1_to_2_bundel[1]

    # undistort not implimented
    c1_points = coordinate_transformations.image_points_to_camera_vectors(points1, camera_matrix, dist_coeffs)
    print(c1_points)

    # try building transformation matrix
    rmat_1_to_2 = cv2.Rodrigues(rvec_1_to_2)[0]
    print(rmat_1_to_2)

    h_rmat_1_to_2 = cv2.convertPointsToHomogeneous(rmat_1_to_2.T).T
    print(h_rmat_1_to_2)

    print("Batman")
    print(tvec_1_to_2)
    h_tvec_1_to_2 = cv2.convertPointsToHomogeneous(tvec_1_to_2.T)
    print(h_tvec_1_to_2.T)

    print("Trex")
    print(np.concatenate((h_rmat_1_to_2, h_tvec_1_to_2), axis=1))

    # fundemental matrix
    #F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # only inlier points
    # pts1 = pts1[mask.ravel() == 1]
    # pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    #lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    #lines1 = lines1.reshape(-1,3)
    #img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    """
    
    # draw selected points on frame from given window
    B1 = np.array(points1)
    original_frame_1 = image_directory_handler.load_frame(input_image_directory, frame1)
    frame_1_copy = np.copy(original_frame_1)
    edited_image_1 = graphics.draw_point_matrix(B1, frame_1_copy, color=None, lineThickness=15)

    B2 = np.array(points2)
    original_frame_2 = image_directory_handler.load_frame(input_image_directory, frame2)
    frame_2_copy = np.copy(original_frame_2)
    edited_image_2 = graphics.draw_point_matrix(B2, frame_2_copy, color=None, lineThickness=15)

    return edited_image_1, edited_image_2

# global variables
UPDATE_WINDOWS = False

def gui(image_directory, available_frames_list, rvec_tvec_frame_map, camera_matrix, dist_coeffs):

    window_one_start_frame = min(available_frames_list)
    window_one_name = "one"
    window_one_size = (300, 400)
    window_one_position = (5, 5)
    window_one_chessboard_corners_image_points = []

    window_one = selection_window(window_one_name, image_directory, available_frames_list, window_one_start_frame,
                                  window_one_size, window_one_position, window_one_chessboard_corners_image_points)
    window_one_image = window_one.get_selected_image()


    window_two_start_frame = max(available_frames_list)
    window_two_name = "two"
    window_two_size = window_one_size
    window_two_position = (window_one_position[0]+window_one_size[0], window_one_position[1])
    window_two_chessboard_corners_image_points = []

    window_two = selection_window(window_two_name, image_directory, available_frames_list, window_two_start_frame,
                                  window_two_size, window_two_position, window_two_chessboard_corners_image_points)
    window_two_image = window_two.get_selected_image()


    while True:
        cv2.imshow(window_one.window_name, window_one_image)
        cv2.imshow(window_two.window_name, window_two_image)

        global UPDATE_WINDOWS

        key = cv2.waitKey(1)
        if key == ord("u") or UPDATE_WINDOWS:
            UPDATE_WINDOWS = False
            window_one_frame = window_one.get_selected_frame()
            window_one_image = window_one.get_selected_image()
            window_one_chessboard_corners_image_points = window_one.get_chessboard_corners_image_points()

            window_two_frame = window_two.get_selected_frame()
            window_two_image = window_two.get_selected_image()
            window_two_chessboard_corners_image_points = window_two.get_chessboard_corners_image_points()

            window_one_image, window_two_image = draw_selected_points(image_directory, window_one_frame,
                                                    window_one_chessboard_corners_image_points,window_two_frame,
                                                    window_two_chessboard_corners_image_points, rvec_tvec_frame_map,
                                                                      camera_matrix, dist_coeffs)

            cv2.destroyAllWindows()

            window_one.deploy_window(window_one_frame, window_one_size, window_one_position)
            window_two.deploy_window(window_two_frame, window_two_size, window_two_position)

        if key == ord("r"):
            window_one_chessboard_corners_image_points = []
            window_two_chessboard_corners_image_points = []

            window_one_frame = window_one.get_selected_frame()
            window_one_image = image_directory_handler.load_frame(image_directory, window_one_frame)

            window_two_frame = window_two.get_selected_frame()
            window_two_image = image_directory_handler.load_frame(image_directory, window_two_frame)

        if key == ord("q"):
            cv2.destroyAllWindows()
            return


    """
    test_image1_path = 'raw_charuco_board_frames_video_1/11.jpg'
    test_image2_path = 'raw_charuco_board_frames_video_1/44.jpg'

    original_image_window_1 = cv2.imread(test_image1_path)

    # clone image, and setup the mouse callback function
    image_window_1 = copy.deepcopy(original_image_window_1)
    window_name_1 = "image 1"
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("image", select_chessboard_image_points)
    # tracker bar
    tracker_name = "frame number window 1"
    cv2.createTrackbar(tracker_name, window_name_1, 0, 255, update_selected_frames)
    # value = cv2.getTrackbarPos(tracker_name, window_name)

    cv2.imshow(window_name_1, image_window_1)
    k = cv2.waitKey(0)
    print(k)
    if (k == 27):  # wait for ESC key to exit
        cv2.destroyWindow(window_name_1)
    """

    # createTrackbar for image frame selection

    # close all open windows
    # cv2.destroyAllWindows()


#!/usr/bin/env python

"""
    User Selection - Basic GUI for stereo chessboard corner selection

    chessboard_corners_image1, chessboard_corners_image1 = stereo_images_chessboard_corner_selection(image1, image2)

    TODO: extend functionality to entire program GUI
"""
import numpy as np
import cv2
from tkinter import *    # Possibly have this module initilize tk within class to be used
import PIL.Image
from PIL import ImageTk

# my lin
import image_directory_handler

# some temporary globals
input_directory = "raw_charuco_board_frames_video_1"

def cv2_to_pil_image(cv2_image):
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGBA)
    pil_image = PIL.Image.fromarray(cv2_image)
    return pil_image

def cv2_image_to_imagetk(cv2_image):
    pil_image = cv2_to_pil_image(cv2_image)
    imagetk = PIL.ImageTk.PhotoImage(image=pil_image)
    return imagetk



class ImageDisplayer(Frame):

    def load_current_frame_into_image_container(self):
        # load cv2 image corresponding to current frame
        self.cv2_image = image_directory_handler.load_frame(self.image_directory, self.current_frame)

        # convert to pillow image and save a copy
        self.pill_image = cv2_to_pil_image(self.cv2_image)
        # resize new
        new_width, new_height = int(self.pill_image.width*0.5),int(self.pill_image.height*0.5)
        self.pill_image = self.pill_image.resize((new_width, new_height))
        self.pill_image_copy = self.pill_image.copy()

        # set the current image after converting to ImageTk
        self.current_image = ImageTk.PhotoImage(self.pill_image)

        # load the current image into the image container
        self.image_container.configure(image=self.current_image)


    def change_frame(self, new_frame):
        # update current frame member
        self.current_frame = new_frame

        # reload the image container now that the frame value has changed
        self.load_current_frame_into_image_container()

    def _resize_image(self, event):
        #new_width, new_height = event.width, event.height
        # new_width, new_height = int(event.width*0.75), int(event.height*0.75)
        new_width, new_height = int(self.master.winfo_width()*0.90), int(self.master.winfo_height() * 0.75)

        # resize image and update copy buffer
        self.pill_image = self.pill_image_copy.resize((new_width, new_height))
        self.current_image = ImageTk.PhotoImage(self.pill_image)

        # update the image container
        self.image_container.configure(image=self.current_image)

    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args)

        # set current frame and image directory members
        assert kwargs is not None
        self.master = master
        self.image_directory = kwargs['image_directory']
        self.current_frame = kwargs['start_frame']

        # create image container
        # resizable, displays image corresponding to current frame
        self.image_container = Label(self, bg="black", borderwidth=15)
        self.image_container.pack()

        # bind resize event to resize method
        self.image_container.bind('<Configure>', self._resize_image)

        # load image container with current frame
        self.load_current_frame_into_image_container()



class FrameSelector(Frame):

    def get_closest_available_value(self, raw_value):
        closest_available_frame = min(self.available_frames, key=lambda x: abs(x - raw_value))
        return closest_available_frame

    def update_frame(self, slider_head):
        #self.image_displayer
        #print(self.current_slider_head.get())
        #self.image_displayer.change_frame(value)
        value = self.get_closest_available_value(int(slider_head))
        self.slider.set(value)
        self.image_displayer.change_frame(value)

    def _create_widgets(self, *args, **kwargs):

        # create image displayer
        self.image_displayer = ImageDisplayer(self, *args, **kwargs)
        self.image_displayer.pack()

        # create slider
        self.current_slider_head = IntVar()
        self.slider = Scale(self, from_=min(self.available_frames),to=max(self.available_frames), bg="black", fg="white", orient=HORIZONTAL, borderwidth=10,
                            variable=self.current_slider_head, command=self.update_frame)
        self.slider.pack(fill=BOTH, expand=YES)





    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args)

        assert kwargs is not None
        self.master = master
        self.available_frames = kwargs['available_frames']
        self._job = None
        self._create_widgets(*args, **kwargs)

        #assert kwargs is not None
        #self.image_directory = kwargs['image_directory']
        #self.current_frame = kwargs['start_frame']


        # create image displayer
        #self.img_displayer = ImageDisplayer(self.container_frame, *args, **kwargs)
        #self.img_displayer.pack(fill=BOTH) #, expand=YES)
        #self.img_displayer.grid(row=0, column=0, sticky=E+W)


        # frame selector, slider, chooses from available current frames
        #self.slider = Scale(self.container_frame, bg="red", fg="white", orient=HORIZONTAL)
        #self.slider.pack(fill=X, expand=YES)
        #self.slider.grid(row=1,column=0, sticky=E+W)



if __name__ == '__main__':

    window = Tk()
    window.geometry("400x400")
    window.configure(background='black')

    #container = Frame(window)
    #container.pack()

    #tmp avaialble frames
    available_frames = list(range(64))
    frame_selector = FrameSelector(window, image_directory=input_directory, start_frame=1, available_frames=available_frames)
    frame_selector.pack(fill=BOTH, expand=YES)





    """
    frame_selector = Frame(window)
    frame_selector.pack()

    slider = Scale(frame_selector, bg="blue", fg="red", orient=HORIZONTAL)
    slider2 = Scale(frame_selector, bg="green", fg="yellow", orient=HORIZONTAL)
    img_displayer = ImageDisplayer(frame_selector, image_directory=input_directory,start_frame=1)


    slider.pack(fill=BOTH, expand=YES)
    slider2.pack()
    img_displayer.pack()
    """


    #img_selector = ImageSelector(window, image_directory=input_directory, start_frame=1)
    #img_selector.pack(fill=BOTH, expand=YES)
    #img_selector.grid(row=0, column=0, sticky=E+W)
    #img_selector.pack(fill=BOTH, expand=YES)

    # mainloop
    window.mainloop()

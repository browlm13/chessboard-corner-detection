#!/usr/bin/env python

"""
    User Selection - Basic GUI for stereo chessboard corner selection

    chessboard_corners_image1, chessboard_corners_image1 = stereo_images_chessboard_corner_selection(image1, image2)

    TODO: extend functionality to entire program GUI
"""
import numpy as np
import cv2
from tkinter import *    # Possibly have this module initilize tk within class to be used
import tkinter.ttk as ttk
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

    def displayImage(self, cv2_image):

        # convert to pillow image and save a copy
        self.pill_image = cv2_to_pil_image(cv2_image)
        self.pill_image_copy = self.pill_image.copy()

        # resize pill_image to label size
        self.pill_image = self.pill_image_copy.resize((self.width, self.height))

        # set the current image after converting to ImageTk
        self.current_image = ImageTk.PhotoImage(self.pill_image)

        # load the current image into the image container
        self.image_container.configure(image=self.current_image)
        self.image_container.image = self.current_image

    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args)

        # set current frame and image directory members
        self.master = master
        self.width = kwargs['img_width']
        self.height = kwargs['img_height']

        # create image container
        self.image_container = Label(self, bg="white", width=self.width, height=self.height, anchor='n', borderwidth=15)
        self.image_container.grid(row=0, column=0)


        #innerWindow.bind('<ButtonPress-1>', closewindow)
        #closeImage = self.current_image
        #innerWindow.closeImage = closeImage
        #innerWindow.create_image(554, 10, image=closeImage, anchor='e')

        # load image container with current frame
        #self.displayImage()

class FrameSelector(Frame):

    def next_frame(self):
        index = self.available_frames.index(self.current_frame) + 1
        if index < len(self.available_frames):
            self.current_frame = self.available_frames[index]
        self.update_image()

    def previous_frame(self):
        index = self.available_frames.index(self.current_frame) - 1
        if index > - 1:
            self.current_frame = self.available_frames[index]
        self.update_image()

    def update_image(self):
        self.set_current_frame_string()
        img = image_directory_handler.load_frame(self.image_directory, self.current_frame)
        self.screen.displayImage(img)

    def set_current_frame_string(self):
        # current frame label
        new_string = "FRAME: " + str(self.current_frame)
        self.current_frame_string.set(new_string)

    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args)

        self.master = master
        self.available_frames = kwargs['available_frames']
        self.current_frame = kwargs['start_frame']
        assert self.current_frame in self.available_frames
        self.image_directory = kwargs['image_directory']

        # image displayer
        kwargs['img_width'] = 300
        kwargs['img_height'] = 300
        self.screen = ImageDisplayer(self, *args, **kwargs)
        self.screen.grid(row=0, column=0, columnspan=3)

        # buttons
        self.previous = Button(self, text="PREV", command=self.previous_frame)
        self.previous.grid(row=1,column=0)
        self.next = Button(self, text="NEXT", command=self.next_frame)
        self.next.grid(row=1, column=2)

        # current frame label
        self.current_frame_string = StringVar()
        self.current_frame_label = Label(self, textvariable=self.current_frame_string)
        self.current_frame_label.grid(row=1, column=1)

        self.update_image()


class RadioWidget(Frame):


    def clicked(self):
        print(self.selected.get())

    def __init__(self, master, *args, **kwargs):
        Frame.__init__(self, master, *args)



        common_bg = '#' + ''.join([hex(x)[2:].zfill(2) for x in (181, 26, 18)])  # RGB in dec
        common_fg = '#ffffff'  # pure white

        #rb1 = ttk.Radiobutton(text="works :)", style='Wild.TRadiobutton')  # Linking style with the button

        # or getting the value
        self.selected = IntVar()



        rad1 = Radiobutton(self, text='First', value=1, variable=self.selected, command=self.clicked, borderwidth=10, fg=common_fg, bg=common_bg,
                            activebackground=common_bg, activeforeground=common_fg, selectcolor="green", relief=SOLID, highlightcolor="red") #2, style='Wild.TRadiobutton')
        rad2 = Radiobutton(self, text='Second', value=2, variable=self.selected, command=self.clicked, fg=common_fg, bg="yellow", activebackground="yellow", activeforeground=common_fg, selectcolor="yellow")
        rad3 = Radiobutton(self, text='Third', value=3, variable=self.selected, command=self.clicked)
        rad4 = Radiobutton(self, text='Fourth', value=4, variable=self.selected, command=self.clicked, selectcolor="blue")

        rad1.grid(column=0, row=0)
        rad2.grid(column=1, row=0)
        rad3.grid(column=2, row=0)
        rad4.grid(column=3, row=0)


        s = ttk.Style(master)  # Creating style element
        #s.configure('Wild.TRadiobutton',  # First argument is the name of style. Needs to end with: .TRadiobutton
        #            background="blue",  # Setting background to our specified color above
        #            foreground='black')  # You can define colors like this also
        #    try also the 'clam' theme
        """
        s.theme_use('alt')
        style_name = rad1.winfo_class()
        s.configure(style_name, foreground=common_fg, background=common_bg, indicatorcolor=common_bg)

        s.map(style_name,
                  foreground=[('disabled', common_fg),
                              ('pressed', common_fg),
                              ('active', common_fg)],
                  background=[('disabled', common_bg),
                              ('pressed', '!focus', common_bg),
                              ('active', common_bg)],
                  indicatorcolor=[('selected', common_bg),
                                  ('pressed', common_bg)]

                  )
        """



if __name__ == '__main__':

    window = Tk()
    window.geometry("400x400")
    window.configure(background='black')

    available_frames = list(range(64))

    input_directory = "raw_charuco_board_frames_video_1"


    fs1 = FrameSelector(window, image_directory=input_directory, available_frames=available_frames,
                       start_frame=available_frames[0])
    fs1.grid(row=1, column=0, padx=20, pady=20)


    fs2 = FrameSelector(window, image_directory=input_directory, available_frames=available_frames,
                       start_frame=available_frames[-1])
    fs2.grid(row=1, column=1, padx=20, pady=20)

    radio_selectors = RadioWidget(window)
    radio_selectors.grid(row=2, column=0, columnspan=2)

    # mainloop
    window.mainloop()

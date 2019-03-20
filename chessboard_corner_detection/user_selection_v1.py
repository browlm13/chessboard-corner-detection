#!/usr/bin/env python

"""
    User Selection

        selected_points = chessboard_corner_selection(image)
        chessboard_corners_image1, chessboard_corners_image1 = stereo_images_chessboard_corner_selection(image1, image2)


"""
# just for image format conversions
import cv2
#from PIL import Image
import PIL.Image
from PIL import ImageTk

from tkinter import *

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

class HeaderWidget:

    def get_outer_frame(self):
        return self.header_frame

    def __init__(self, parent):
        # width height
        self.width, self.height = 750, 75

        self.parent = parent

        # create frame
        self.header_frame = Frame(self.parent) #, width=self.width, height=self.height)

        self.text = "Title"

        # create angle text title label
        self.header_label = Label(self.header_frame, text=self.text, font=("Arial Bold", 30), bg="yellow")
        self.header_label.grid(row=0, column=0, columnspan=3, sticky=N+E+W)

        # create canvas
        #self.canvas_width, self.canvas_height = self.width, self.height - self.angle_label.winfo_height()
        self.canvas = Canvas(self.header_frame, bg="blue") #, width=self.width, height=self.height)
        self.canvas.grid(row=1, column=0, columnspan=3, sticky=N+E+W)

def selection_tool():
    print("selection tool clicked")

def magnify_tool():
    print("magnify tool clicked")

def move_tool():
    print("move tool clicked")

class ToolBarWidget:

    def get_outer_frame(self):
        return self.toolbar_frame

    def __init__(self, parent):

        self.parent = parent

        # create toolbar frame
        self.toolbar_frame = Frame(self.parent)

        # select button, TODO: Make class
        self.select_button = Button(self.toolbar_frame, relief=RAISED, text="Select", fg="blue", command=selection_tool)
        self.select_button.grid(row=0, column=0, padx=5, pady=5)

        self.magnify_button = Button(self.toolbar_frame, relief=RAISED, text="Magnify", fg="red", command=magnify_tool)
        self.magnify_button.grid(row=1, column=0, padx=5, pady=5)

        self.move_button = Button(self.toolbar_frame, relief=RAISED, text="Move", fg="green", command=move_tool)
        self.move_button.grid(row=2, column=0, padx=5, pady=5)

class AngleWidget:

    def get_outer_frame(self):
        return self.angle_frame

    def __init__(self, parent, angle_number):
        # width height
        # self.width, self.height = 300, 350
        self.img_width, self.img_height = 300, 300

        self.parent = parent
        self.angle_number = angle_number

        # set text
        self.text = "Angle %s" % self.angle_number

        # create angle frame
        self.angle_frame = Frame(self.parent) #, width=self.width, height=self.height)

        # create angle text title label
        self.angle_label = Label(self.angle_frame, text=self.text, font=("Arial Bold", 14), bg="blue")
        self.angle_label.grid(row=0, column=0, sticky=N)

        # create canvas
        self.canvas = Canvas(self.angle_frame, bg="green") #, width=self.img_width, height=self.img_height)
        self.canvas.grid(row=1, column=0, sticky=E+W)


        # draw on canvas
        # self.canvas.create_line(0, 0, 50, 20, fill="#476042", width=3)

        # put gif photoimage on canvas
        # pic's upper left corner (NW) on the canvas is at x=50 y=10
        # self.canvas.create_image(50, 10, image=gif1, anchor=NW)
        # put gif image on canvas
        # pic's upper left corner (NW) on the canvas is at x=50 y=10
        # self.photo1 = PhotoImage(file='chessboard_corner_selection_animations/camera_1.gif')
        self.original = PIL.Image.open('chessboard_corner_selection_animations/camera_1.gif')
        self.resized = self.original.resize((self.img_width, self.img_height), PIL.Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(self.resized)

        # self.canvas.create_image(0, 0, image=self.image, anchor=NW)

        # slider
        self.slider = Scale(self.angle_frame, bg="red", fg="white", orient=HORIZONTAL)
        self.slider.grid(row=2, column=0)


class RadioWidget:

    def get_outer_frame(self):
        return self.radio_frame

    def clicked(self):
        print(self.selected.get())

    def __init__(self, parent):

        self.parent = parent

        # create frame
        self.radio_frame = Frame(self.parent)

        # or getting the value
        self.selected = IntVar()

        rad1 = Radiobutton(self.radio_frame, text='First', value=1, variable=self.selected, command=self.clicked)
        rad2 = Radiobutton(self.radio_frame, text='Second', value=2, variable=self.selected, command=self.clicked)
        rad3 = Radiobutton(self.radio_frame, text='Third', value=3, variable=self.selected, command=self.clicked)
        rad4 = Radiobutton(self.radio_frame, text='Fourth', value=4, variable=self.selected, command=self.clicked)

        rad1.grid(column=0, row=0)
        rad2.grid(column=1, row=0)
        rad3.grid(column=2, row=0)
        rad4.grid(column=3, row=0)


def clicked():
    rad1.configure(text="Button was clicked !!")


if __name__ == '__main__':
    window = Tk()
    window.title("Selection Window")
    window.configure(background="black")

    # set window size in pixels
    #window.geometry('900x900')

    # test out angle frame class
    angle_widget_1 = AngleWidget(window, 1)
    angle_widget_2 = AngleWidget(window, 2)

    angle_widget_1.get_outer_frame().grid(row=1, column=0, sticky=W, padx=10)
    angle_widget_2.get_outer_frame().grid(row=1, column=2, sticky=E, padx=10)

    # test out toolbar widget
    toolbar_widget = ToolBarWidget(window)
    toolbar_widget.get_outer_frame().grid(row=1, column=1)

    # test out header widget
    header_widget = HeaderWidget(window)
    header_widget.get_outer_frame().grid(row=0, column=0, columnspan=3, sticky=W+E)

    # test out radio widget
    #radio_widget = RadioWidget(window)
    #radio_widget.get_outer_frame().grid(row=2, column=0, columnspan=3, pady=10)

    # mainloop
    window.mainloop()



    # label / photo
    # photo1 = PhotoImage(file='chessboard_corner_selection_animations/camera_1.gif')
    # Label (window, image=photo1, bg='black').grid(row=0, column=0, sticky=E)    # sticky "E-ast"

    # label text
    # lbl = Label(window, text="Hello", font=("Arial Bold", 50))
    # lbl.grid(column=0, row=0)

    #rad1 = Radiobutton(window, text='First', value=1, command=clicked)
    #rad2 = Radiobutton(window, text='Second', value=2)
    #rad3 = Radiobutton(window, text='Third', value=3)
    #rad4 = Radiobutton(window, text='Fourth', value=4)

    """
    # or getting the value
    selected = IntVar()

    rad1 = Radiobutton(window, text='First', value=1, variable=selected)
    rad2 = Radiobutton(window, text='Second', value=2, variable=selected)
    rad3 = Radiobutton(window, text='Third', value=3, variable=selected)
    rad4 = Radiobutton(window, text='Fourth', value=4, variable=selected)


    rad1.grid(column=0, row=0)
    rad2.grid(column=1, row=0)
    rad3.grid(column=2, row=0)
    rad4.grid(column=3, row=0)
 

    # slider
    # w = Scale(master, from_=0, to=42) # not horizontal
    # tickinterval=10 as paramter, length= as parameter
    w = Scale(window, from_=0, to=200, orient=HORIZONTAL, bg="red", fg="white")

    w.pack()
    #w.grid(row=0, column=0, sticky=E)
    # get value
    #w.get()
    # set value
    # w.set(value)

    # button
    # btn = Button(window, text="Click Me", bg="orange", fg="red", command=clicked)
    # btn.grid(column=1, row=0)

    # set window size in pixels
    window.geometry('350x200')


    window.mainloop()
    """
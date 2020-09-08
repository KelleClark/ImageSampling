# -*- coding: utf-8 -*-

from tkinter import *
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import argparse
import cv2
import numpy as np
import os
import sys
import filetype


# set up an a list for files
images = []
# to store the image file types
filetypes = []
# dimensions of each image, stored as strings
columns = []
rows = []
pixels = []
# index for the list of images in the browser
count = 0



# Get and parse the arguments
def get_args():
    parser = argparse.ArgumentParser(description='Image browser v1.0')
    parser.add_argument('path', metavar='dir',
                        help='The root directory to view photos in')
    parser.add_argument('--rows', type=int,  default=720,
                        help='Max number of rows on screen  (Default is 720)')
    parser.add_argument('--cols',  type=int, default=1080,
                        help='Max number of columns on screen (Default is 1080)')

    args = parser.parse_args()
    return(args)

# Check for images in the path and save the exact path to each
#   image in a list.
def load_path(path):
    global images
    rootDir = os.path.join(path)
    for dirName, subdirList, fileList in os.walk(rootDir):
        for fname in fileList:
            pos_img = dirName + "/" + fname
            if cv2.haveImageReader(pos_img): # if it is a readable image
                images.append(pos_img)  #add it to the list of images
                filetypes.append(filetype.guess(pos_img).mime) #Get the file type and save it
                # add placeholders for image column count, row count, and pixel count to lists
                columns.append("")
                rows.append("")
                pixels.append("")
    # If there is a problem with the given path, exit
    if len(images) == 0:
        print("Invalid path or there are no images in path")
        sys.exit(1)


# Load the first image from the directory as opencv
def opencv_img(count):
    # read and convert image
    image = cv2.imread(images[count])
    columns[count] = str(image.shape[1]) # add column count to list
    rows[count] = str(image.shape[0]) # add row count to list
    pixels[count] = str(image.shape[1] * image.shape[0]) # add pixel count to list

    return(image)

# Convert it to ImageTK
# necessary to use cvtColor to correct to expected RGB color
def convert_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To proper format for tk
    im = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=im)
    return(imgtk)

# Wrapper to load the image for display
def load_img(count):
    current_disp = convert_img(opencv_img(count))
    return current_disp


# Update the information about the newest photo and the image itself
#   on the window
def update_window(imgtk, tex):
        label['image'] = imgtk
        label['text'] = tex[1]+"\nfrom "+tex[0]+"\n"+columns[count]+" x "+ \
        rows[count]+" ("+pixels[count]+" pixels)\nImage type: "+ \
        filetypes[count]+ "\nFile size: "+str(os.lstat(images[count]).st_size)\
        +" bytes"
        label.photo = imgtk

# Go to next image
def next_img(event):
    global count
    if count >= len(images) -1:
        count = -1 # -1 to offset regular function
    count = count + 1  # Next image in the list
    imgtk = load_img(count)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

# Go to prior image
def prev_img(event):
    global count
    if count <= 0:
        count = (len(images) - 1) + 1 # +1 to offset regular function
    count = count - 1  # Prior image in the list
    imgtk = load_img(count)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

def intensity_6(event):
    img = opencv_img(count)
    new_img = (img // 4 * 4) + (4 // 2)
    imgtk = convert_img(new_img)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

def intensity_4(event):
    img = opencv_img(count)
    new_img = (img // 16 * 16) + (16 // 2)
    imgtk = convert_img(new_img)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)


def shrink_NN(event):
    nearest_neighbor(0.5)

def shrink_bicubic(event):
    bicubic(0.5)

def shrink_bilinear(event):
    bilinear(0.5)

def increase_NN(event):
    nearest_neighbor(2)

def increase_bicubic(event):
    bicubic(2)

def increase_bilinear(event):
    bilinear(2)

def nearest_neighbor(factor=0.5):
    global count
    global current_disp
    image = opencv_img(count)
    current_disp = cv2.resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)), interpolation=cv2.INTER_NEAREST)
    imgtk = convert_img(current_disp)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

def bicubic(factor = 0.5):
    global count
    global current_disp
    image = opencv_img(count)
    current_disp = cv2.resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)), interpolation=cv2.INTER_CUBIC)
    imgtk = convert_img(current_disp)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

def bilinear(factor=0.5):
    global count
    global current_disp
    image = opencv_img(count)
    current_disp = cv2.resize(image, (int(image.shape[1]*factor), int(image.shape[0]*factor)), interpolation=cv2.INTER_LINEAR)
    imgtk = convert_img(current_disp)
    tex = extract_meta()
    #Update the display
    update_window(imgtk, tex)

#Exit the program
def quit_img(event):
    root.destroy() #Kill the display
    sys.exit(0)

# Gets filename and file location
def extract_meta():
    global count
    ind = images[count].rindex("/")
    ans = ['','']
    if ind != -1:
        ans[0] = images[count][0:ind]
        ans[1]= images[count][ind+1:]

    return ans

# write the image to the path of current image indexed by count
def write_img(event):
    global count
    global current_disp
    currpath = extract_meta()
    newname = currpath[1]+"new"+str(count)
    status = cv2.imwrite(currpath[0]+"/"+ newname+".png", current_disp)
    if status != False:
        print("A new image has been added at "+currpath[0]+newname)
        showinfo("Image saved at "+currpath[0]+newname)
        load_path(args.path)
    else:
       print("The image was not saved")
       showinfo("The image was not saved")


def main():

    #Get the command arguments
    global args
    args = get_args()

    # Root window
    global root
    root = Tk()
    load_path(args.path)
    imgtk = load_img(count)
    tex = extract_meta()

    # keep track of the image currently in window
    global current_disp
    current_disp = imgtk

    # Put everything in the display window
    global label
    label = Label(root, text = tex[1]+"\nfrom "+tex[0]+"\n"+columns[count]+ \
    " x " +rows[count]+" ("+pixels[count]+" pixels)\nImage type: "+ \
    filetypes[count]+"\nFile size: "+str(os.lstat(images[count]).st_size)+ \
    " bytes", compound = RIGHT, image=imgtk)
    label.pack()

    # Frame to display navigation buttons at bottom of window
    frame = Frame()
    frame.pack()

    # Button for prior image
    btn_previous = Button(
        master = frame,
        text = "Previous",
        underline = 0
    )
    btn_previous.grid(row = 0, column = 0)
    btn_previous.bind('<ButtonRelease-1>', prev_img)

    # Button for Save image
    btn_save = Button(
        master = frame,
        text = "Save",
        underline = 0
    )
    btn_save.grid(row = 0, column = 1)
    btn_save.bind('<ButtonRelease-1>', write_img)


    # Button for next image
    btn_next = Button(
        master = frame,
        text = "Next",
        underline = 0
    )
    btn_next.grid(row = 0, column = 2)
    btn_next.bind('<ButtonRelease-1>', next_img)

    # Button for Nearest neighbor
    btn_shrink_NN = Button(
        master = frame,
        text = "Nearest Neighbor, shrink",
        underline = 0
    )
    btn_shrink_NN.grid(row = 1, column = 0)
    btn_shrink_NN.bind('<ButtonRelease-1>', shrink_NN)

    # Button for Bicubic
    btn_shrink_bicubic = Button(
        master = frame,
        text = "Bicubic, shrink",
        underline = 2
    )
    btn_shrink_bicubic.grid(row = 1, column = 1)
    btn_shrink_bicubic.bind('<ButtonRelease-1>', shrink_bicubic)

    # Button for Bilinear
    btn_shrink_bilinear = Button(
        master = frame,
        text = "Bilinear, shrink",
        underline = 2
    )
    btn_shrink_bilinear.grid(row = 1, column = 2)
    btn_shrink_bilinear.bind('<ButtonRelease-1>', shrink_bilinear)

      # Button for Nearest neighbor increase
    btn_increase_NN = Button(
        master = frame,
        text = "Nearest Neighbor, increase",
        underline = 1
    )
    btn_increase_NN.grid(row = 2, column = 0)
    btn_increase_NN.bind('<ButtonRelease-1>', increase_NN)

    # Button for Bicubic increase
    btn_bicubic = Button(
        master = frame,
        text = "Bicubic, increase",
        underline = 2
    )
    btn_bicubic.grid(row = 2, column = 1)
    btn_bicubic.bind('<ButtonRelease-1>', increase_bicubic)

    # Button for Bilinear increase
    btn_bilinear = Button(
        master = frame,
        text = "Bilinear, increase",
        underline = 2
    )
    btn_bilinear.grid(row = 2, column = 2)
    btn_bilinear.bind('<ButtonRelease-1>', increase_bilinear)


    btn_6_bit = Button(
        master = frame,
        text = "k=6",
    )
    btn_6_bit.grid(row = 4, column = 1)
    btn_6_bit.bind('<ButtonRelease-1>', intensity_6)

    btn_4_bit = Button(
        master = frame,
        text = "k=4",
    )
    btn_4_bit.grid(row = 4, column = 0)
    btn_4_bit.bind('<ButtonRelease-1>', intensity_4)




    # Bind all the required keys to functions
    root.bind('<n>', next_img)
    root.bind("<p>", prev_img)
    root.bind("<q>", quit_img)

    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()

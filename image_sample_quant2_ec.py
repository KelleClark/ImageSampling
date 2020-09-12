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

#current image in format for saving
current_img = None

# Current manipulation
manipul = ""

# Get and parse the arguments
def get_args():
    parser = argparse.ArgumentParser(description='Image Manipulation v1.0')
    parser.add_argument('path', metavar='dir',
                        help='The root directory to view photos in')

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
    global current_img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To proper format for tk
    current_img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=current_img)
    return(imgtk)

# Wrapper to load the image for display
def load_img(count):
    current_disp = convert_img(opencv_img(count))
    return current_disp


# Update the information about the newest photo and the image itself
#   on the window
def update_window(imgtk, manipulate=False, info = []):
        tex = extract_meta()
        label['image'] = imgtk
        if manipulate:
              print(tex[1]+"\nfrom MANIPULATE \n"+info[0]+" x "+ \
        info[1]+" ("+info[2]+" pixels)\nFile size: "+info[3]\
        +" bytes\n\n")
        else:
            print(tex[1]+"\nfrom "+tex[0]+"\n"+columns[count]+" x "+ \
        rows[count]+" ("+pixels[count]+" pixels)\nImage type: "+ \
        filetypes[count]+ "\nFile size: "+str(os.lstat(images[count]).st_size)\
        +" bytes\n\n")
        label.photo = imgtk

# Go to next image
def next_img(event):
    global count
    if count >= len(images) -1:
        count = -1 # -1 to offset regular function
    count = count + 1  # Next image in the list
    imgtk = load_img(count)
    #Update the display
    update_window(imgtk)

# Go to prior image
def prev_img(event):
    global count
    if count <= 0:
        count = (len(images) - 1) + 1 # +1 to offset regular function
    count = count - 1  # Prior image in the list
    imgtk = load_img(count)
    #Update the display
    update_window(imgtk)

# Reduce intensity to 6 bit 
def intensity_6(event):
    global manipul 
    img = opencv_img(count)
    new_img = (img // 4 * 4) + (4 // 2)
    imgtk = convert_img(new_img)
    
    manipul  = "k6"
    
    #Update the display
    update_window(imgtk)

# Reduce intensity to 4 bit 
def intensity_4(event):
    global manipul 
    img = opencv_img(count)
    new_img = (img // 16 * 16) + (16 // 2)
    imgtk = convert_img(new_img)
    
    manipul = "k4"
    
    #Update the display
    update_window(imgtk)


# Shrink using nearest neighbor with a factor or 0.5
def shrink_NN(event):
    global manipul 
    nearest_neighbor(0.5)
    manipul = "nn_shrink"

# Shirnk using bicubic with a factor or 0.5
def shrink_bicubic(event):
    global manipul 
    bicubic(0.5)
    manipul = "bilin_shrink"

# Shirnk using bilinear with a factor or 0.5
def shrink_bilinear(event):
    global manipul 
    bilinear(0.5)
    manipul = "bicube_shrink"

# Increase using nearest neighbor with a factor or 2
def increase_NN(event):
    global manipul 
    nearest_neighbor(2)
    manipul = "nn_increase"

# Increase using bicubic with a factor or 2
def increase_bicubic(event):
    global manipul 
    bicubic(2)
    manipul = "bilin_increase"

# Increase using bilinear with a factor or 2
def increase_bilinear(event):
    global manipul 
    bilinear(2)
    manipul = "bicube_increase"

# Nearest neigbor interpolation to the given factor
def nearest_neighbor(factor=0.5):
    global count
    global current_disp
    image = opencv_img(count)
    image = cv2.resize(image, (int(image.shape[1]*factor), 
                                      int(image.shape[0]*factor)), 
                                      interpolation=cv2.INTER_NEAREST)
    imgtk = convert_img(image)
    size = [str(image.shape[1]), str(image.shape[0]), 
            str(image.shape[1] * image.shape[0]), str(sys.getsizeof(image))]
    tex = extract_meta()
    #Update the display
    update_window(imgtk, True, size)

#Bicubic interpolation to the given factor
def bicubic(factor = 0.5):
    global count
    global current_disp
    image = opencv_img(count)
    image = cv2.resize(image, (int(image.shape[1]*factor), 
                                      int(image.shape[0]*factor)), 
                                      interpolation=cv2.INTER_CUBIC)
    imgtk = convert_img(image)
    size = [str(image.shape[1]), str(image.shape[0]),
            str(image.shape[1] * image.shape[0]), str(sys.getsizeof(image))]
    tex = extract_meta()
    #Update the display
    update_window(imgtk, True, size)

# Bilinear interpolation to the given factor
def bilinear(factor=0.5):
    global count
    global current_disp
    image = opencv_img(count)
    image = cv2.resize(image, (int(image.shape[1]*factor), 
                                      int(image.shape[0]*factor)), 
                                      interpolation=cv2.INTER_LINEAR)
    imgtk = convert_img(image)
    size = [str(image.shape[1]), str(image.shape[0]),
            str(image.shape[1] * image.shape[0]), str(sys.getsizeof(image))]
    tex = extract_meta()
    #Update the display
    update_window(imgtk, True, size)

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
    global count, current_img,manipul
    currpath = extract_meta()
    newname = manipul+"_"+currpath[1]
    status = current_img.save(currpath[0]+newname)
    if status != False:
        print("A new image has been added at "+currpath[0]+newname)
        load_path(args.path)
    else:
       print("The image was not saved")


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
    label = Label(root, image=imgtk)
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
    root.bind('4', intensity_4)
    root.bind("6", intensity_6)

    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()

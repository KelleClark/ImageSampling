# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:45:10 2020

@author: sclar
"""


from tkinter import *
from tkinter.messagebox import showinfo
from PIL import Image, ImageTk
import argparse
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
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
intensity = []
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
                intensity.append(os.path.getsize(pos_img)) #at first this is the size of the image in bytes until opencv_img
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
    intensity[count] = intensity[count]/image.shape[0]*image.shape[1]
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
        +" bytes\n with intensity "+str(intensity[count])
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


# Enlarge image by factor of 2 using linear interpolation
def enlarge_linear(event):
    global current_disp
    image = opencv_img(count)

    # print("The shape of the image is " + current_disp.shape[0] + " by " + current_disp.shape[1])
    print("The shape of the image is " + str(image.shape[0]) + " by " + str(image.shape[1]))

    # We make a new matrix (multi-dim array) of all zeros twice the width and height that holds RGB
    # enlarged_img = np.zeros((2 * current_disp.shape[0], 2 * current_disp.shape[1], 3), dtype=np.uint8)
    enlarged_img = np.zeros((2 * image.shape[0], 2 * image.shape[1], 3), dtype=np.uint8)

    # For each row, go to each column position and if the col is divisible by 2, copy old img values
    # otherwise use linear interpolation with left and right spatial coordiante RGB values
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            enlarged_img[i*2, j*2] = image[i, j]
            if j < image.shape[1] - 1:
                enlarged_img[i*2, j*2 + 1] = (1 / 2) * image[i, j] + (1 / 2) * image[i, j + 1]
            else:
                enlarged_img[i*2, j*2 + 1] = image[i, j]

    # For each column, go to each row position and if the row is divisible by 2, copy old img values
    # otherwise use linear interpolation with above and below spatial coordiante RGB values
    # for j in range(current_disp.shape[1] * 2):
    for j in range(0, enlarged_img.shape[1] - 2, 2):
        for i in range(1, enlarged_img.shape[0] - 1, 2):
            enlarged_img[i, j] = (1/2)*enlarged_img[i-1, j] + (1/2)*enlarged_img[i+1, j]
            enlarged_img[i, j+1] = (1/2)*enlarged_img[i+1, j] + (1/2) * enlarged_img[i-1, j+2]
        enlarged_img[enlarged_img.shape[0] - 1, j] = enlarged_img[enlarged_img.shape[0] - 2, j]
        enlarged_img[enlarged_img.shape[0] - 1, j+1] = enlarged_img[enlarged_img.shape[0] - 1, j]

    for i in range(1, enlarged_img.shape[0] - 1, 2):
        enlarged_img[i, enlarged_img.shape[1] - 2] = (1/2)*enlarged_img[i-1, enlarged_img.shape[1] - 2] + (1/2)*enlarged_img[i+1, enlarged_img.shape[1] - 2]
        enlarged_img[i, enlarged_img.shape[1] - 1] = (1/2)*enlarged_img[i-1, enlarged_img.shape[1] - 1] + (1/2)*enlarged_img[i+1, enlarged_img.shape[1] - 1]

    enlarged_img = enlarged_img[0:int(enlarged_img.shape[0]*2), 0:int(enlarged_img.shape[1]*2)]
    imgtk = convert_img(enlarged_img)
    tex = extract_meta()
    update_window(imgtk, tex)

# Shrink image by factor of 2 (or nearly 2 if odd dimension) using linear interpolation
def shrink_linear(event):
    global current_disp
    image = opencv_img(count)

    # print("The shape of the image is " + current_disp.shape[0] + " by " + current_disp.shape[1])
    print("The shape of the image is " + str(image.shape[0]) + " by " + str(image.shape[1]))

    # We make a new matrix (multi-dim array) of all zeros twice the width and height that holds RGB
    # enlarged_img = np.zeros((2 * current_disp.shape[0], 2 * current_disp.shape[1], 3), dtype=np.uint8)
    shrunken_img = np.ones((image.shape[0]//2, image.shape[1]//2, 3), dtype=np.uint8)

    # The new image will have only a new pixel in the center of each rectangle of old pixel positions
    # were two adjacent rows and two adjacent columns create a square.  Taking these squares to be 
    # non-overlapping approximates the half size for the original image.
    # Value of RGB function at the new point is created by using linear interpolation between
    # bottom left and top right oringal pixel values of the square.

            
    for i in range(1,shrunken_img.shape[0]):
        for j in range(shrunken_img.shape[1]-1):
            shrunken_img[i, j] = (1/2)*image[2*i,2*j]+(1/2)*image[2*i-1,2*j]

    shrunken_img = shrunken_img[0:int(shrunken_img.shape[0]*2), 0:int(shrunken_img.shape[1]*2)]
    current_disp = shrunken_img
    imgtk = convert_img(shrunken_img)
    tex = extract_meta()
    update_window(imgtk, tex)

# Change the intensity k
def change_intensity(event):
    global current_disp
    img = opencv_img(count)
    k = 4
    target_level = 2**k
    target_compr_factor = 256/target_level
    
    # a new matrix (multi-dim array) of all ones with the same width and height that holds RGB to be normalized
    normalized_img = np.ones((img.shape[0], img.shape[1], 3))
    
    # a new matrix (multi-dim array) of all ones with the same width and height that holds RGB for intensity change
    changed_img = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    # The new image will have the a new intensity value
    # so we know how many bits are being used by looking at intensity[count] in the current image
    # normalizing the color using a linear transformation from the range of RGB values in the original
    # image to a smaller range from 0 - (2^k -1) for each k
    
    max_red = np.amax(img[:,:,0])
    max_green = np.amax(img[:,:,1])
    max_blue = np.amax(img[:,:,2])
    min_red = np.amin(img[:,:,0])
    min_green = np.amin(img[:,:,1])
    min_blue = np.amin(img[:,:,2])
    print("the max red is " + str(max_red)+" and the min red is "+ str(min_red))
    print("the max green is " + str(max_green)+" and the min red is "+ str(min_green))
    print("the max blue is " + str(max_blue)+" and the min blue is "+ str(min_blue))
   
    # tuple to select colors of each channel line
    colors = ("r", "g", "b")
    channel_ids = (0, 1, 2)

    # create the histogram plot, with three lines, one for
    # each color
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            img[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()
    
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            normalized_img[i, j,0] = (img[i,j,0] - min_red)/(max_red - min_red)
            normalized_img[i, j,1] = (img[i,j,1] - min_green)/(max_green - min_green)
            normalized_img[i, j,2] = (img[i,j,2] - min_blue)/(max_blue - min_blue)
    
    max_normalized_red = np.amax(normalized_img[:,:,0])
    max_normalized_green = np.amax(normalized_img[:,:,1])
    max_normalized_blue = np.amax(normalized_img[:,:,2])
    min_normalized_red = np.amin(normalized_img[:,:,0])
    min_normalized_green = np.amin(normalized_img[:,:,1])
    min_normalized_blue = np.amin(normalized_img[:,:,2])
    print("the normalized_max red is " + str(max_normalized_red)+" and the normalized_min red is "+ str(min_normalized_red))
    print("the normalized_max green is " + str(max_normalized_green)+" and the normalized_min green is "+ str(min_normalized_green))
    print("the normalized_mmx blue is " + str(max_normalized_blue)+" and the normalized_min blue is "+ str(min_normalized_blue))
    
    
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            changed_img[i, j,0] =  np.uint8(min(255,math.floor(np.double(normalized_img[i,j,0]*target_level)) * 0.999 * (2**(8-k))))
            changed_img[i, j,1] =  np.uint8(min(255,math.floor(np.double(normalized_img[i,j,1]*target_level)) * 0.999 * (2**(8-k))))
            changed_img[i, j,2] =  np.uint8(min(255,math.floor(np.double(normalized_img[i,j,2]*target_level)) * 0.999 * (2**(8-k))))
    
    max_changed_red = np.amax(changed_img[:,:,0])
    max_changed_green = np.amax(changed_img[:,:,1])
    max_changed_blue = np.amax(changed_img[:,:,2])
    min_changed_red = np.amin(changed_img[:,:,0])
    min_changed_green = np.amin(changed_img[:,:,1])
    min_changed_blue = np.amin(changed_img[:,:,2])
    print("the max changed_red is " + str(max_changed_red)+" and the min changed_red is "+ str(min_changed_red))
    print("the max changed_green is " + str(max_changed_green)+" and the min changed_green is "+ str(min_changed_green))
    print("the max changed_blue is " + str(max_changed_blue)+" and the min changed_blue is "+ str(min_changed_blue))
    
    changed_img = changed_img[0:int(changed_img.shape[0]), 0:int(changed_img.shape[1])]
   
    plt.xlim([0, 256])
    for channel_id, c in zip(channel_ids, colors):
        histogram, bin_edges = np.histogram(
            changed_img[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=c)

    plt.xlabel("Color value")
    plt.ylabel("Pixels")

    plt.show()
    
    current_disp = changed_img
    imgtk = convert_img(changed_img)
    tex = extract_meta()
    update_window(imgtk, tex)
    
    
# Exit the program
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
    btn_nearest_sh = Button(
        master = frame,
        text = "Nearest Neighbor, shrink",
        underline = 0
    )
    btn_nearest_sh.grid(row = 1, column = 0)
    btn_nearest_sh.bind('<ButtonRelease-1>', shrink_NN)

    # Button for Bicubic
    btn_bicubic_sh = Button(
        master = frame,
        text = "Bicubic, shrink",
        underline = 2
    )
    btn_bicubic_sh.grid(row = 1, column = 1)
    btn_bicubic_sh.bind('<ButtonRelease-1>', shrink_bicubic)

    # Button for Bilinear
    btn_bilinear_sh = Button(
        master = frame,
        text = "Bilinear, shrink",
        underline = 2
    )
    btn_bilinear_sh.grid(row = 1, column = 2)
    btn_bilinear_sh.bind('<ButtonRelease-1>', shrink_bilinear)
    
    # Button for Linear
    btn_linear_sh = Button(
        master = frame,
        text = "Linear, shrink",
        underline = 2
    )
    btn_linear_sh.grid(row = 1, column = 3)
    btn_linear_sh.bind('<ButtonRelease-1>', shrink_linear)

    # Button for Nearest neighbor increase
    btn_nearest_in = Button(
        master = frame,
        text = "Nearest Neighbor, increase",
        underline = 1
    )
    btn_nearest_in.grid(row = 2, column = 0)
    btn_nearest_in.bind('<ButtonRelease-1>', increase_NN)

    # Button for Bicubic increase
    btn_bicubic_in = Button(
        master = frame,
        text = "Bicubic, increase",
        underline = 2
    )
    btn_bicubic_in.grid(row = 2, column = 1)
    btn_bicubic_in.bind('<ButtonRelease-1>', increase_bicubic)

    # Button for Bilinear increase
    btn_bilinear_in = Button(
        master = frame,
        text = "Bilinear, increase",
        underline = 2
    )
    btn_bilinear_in.grid(row = 2, column = 2)
    btn_bilinear_in.bind('<ButtonRelease-1>', increase_bilinear)

    btn_linear_in = Button(
        master = frame,
        text = "Linear, increase",
    )
    btn_linear_in.grid(row = 2, column = 3)
    btn_linear_in.bind('<ButtonRelease-1>', enlarge_linear)
    
    btn_intensity = Button(
        master = frame,
        text = "change intensity",
    )
    btn_intensity.grid(row = 2, column = 4)
    btn_intensity.bind('<ButtonRelease-1>', change_intensity)

    # Bind all the required keys to functions
    root.bind('<n>', next_img)
    root.bind("<p>", prev_img)
    root.bind("<q>", quit_img)

    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()

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


'''might get changed'''

def shrink_NN(event):
    nearest_neighbor(0.5)

def shrink_bicubic(event):
    bicubic(0.5)

def shrink_bilinear(event):
    bilinear(0.5)
    
def shrink_linear(event):
    linear(0.5)
    

def increase_NN(event):
    nearest_neighbor(2)

def increase_bicubic(event):
    bicubic(2)

def increase_bilinear(event):
    bilinear(2)
    
def increase_linear(event):
    linear_inc(2)

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

# Use linear interpolation to fill in missing pixels when shrinking or increasing by factor of 2
def linear_inc(factor  = 2):
    global current_disp
    global columns
    global rows
    global count
    
    # Take current image and make into a np.array
    old_image = np.array(current_disp)
    # We make a new matrix (multi-dim array) of all zeros twice the width and height that holds RGB
    new_img = np.zeros((2*int(rows[count]),2*int(columns[1]), 3),dtype=np.uint8)
    
    # For each row, go to each column position and if the col is divisible by 2, copy old img values
    # otherwise use linear interpolation with left and right spatial coordiante RGB values
    for i in range(2*int(columns[count])):
        if i%2 == 0:
            for j in range(2*int(rows[count])):
                if j%2 == 0:
                    new_img[i,j] = old_image[i,j//2]
                else:
                    new_img[i,j] = (1/2)*current_disp[i,j//2] + (1/2)*current_disp[i,j//2+1]             
                
    # For each column, go to each row position and if the row is divisible by 2, copy old img values
    # otherwise use linear interpolation with above and below spatial coordiante RGB values            
    for j in range(2*int(rows[count])):
        if j%2 == 0:
            for i in range(2*int(columns[count])):
                if i%2 ==0:
                    new_img[i,j] = current_disp[i/2,j]
                else:
                    new_img[i,j] = (1/2)*current_disp[i//2-1,j] + (1/2)*current_disp[i//2+1,j]              
    
    # We now use the diagonal line to fill in the pixels in the gaps left 
    for i in range(2*int(columns[count])):
        if i%2 != 0:
            for j in range(2*int(rows[count])):
                if j%2 != 0:
                    new_img[i,j] = (1/2)*current_disp[i//2 + 1, j//2 - 1] + (1/2)*current_disp[i//2 -1, j//2 + 1]
           

    # To make an image from the array
    img =  ImageTk.PhotoImage(image=Image.fromarray(new_img))                        
    imgtk = convert_img(img)
    tex = extract_meta()
    update_window(imgtk,tex)
   
    
    
    
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
    btn_nearest = Button(
        master = frame,
        text = "Nearest Neighbor, shrink",
        underline = 0
    )
    btn_nearest.grid(row = 1, column = 0)
    btn_nearest.bind('<ButtonRelease-1>', shrink_NN)

    # Button for Bicubic
    btn_bicubic = Button(
        master = frame,
        text = "Bicubic, shrink",
        underline = 2
    )
    btn_bicubic.grid(row = 1, column = 1)
    btn_bicubic.bind('<ButtonRelease-1>', shrink_bicubic)

    # Button for Bilinear
    btn_bilinear = Button(
        master = frame,
        text = "Bilinear, shrink",
        underline = 2
    )
    btn_bilinear.grid(row = 1, column = 2)
    btn_bilinear.bind('<ButtonRelease-1>', shrink_bilinear)
    
    # Button for Linear
    btn_linear_shrink = Button(
        master = frame,
        text = "Linear, shrink",
        underline = 2
    )
    btn_linear_shrink.grid(row = 1, column = 3)
    btn_linear_shrink.bind('<ButtonRelease-1>', shrink_linear)

    # Button for Nearest neighbor increase
    btn_nearest = Button(
        master = frame,
        text = "Nearest Neighbor, increase",
        underline = 0
    )
    btn_nearest.grid(row = 2, column = 0)
    btn_nearest.bind('<ButtonRelease-1>', increase_NN)

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
    
    # Button for Linear increase
    btn_linear_inc = Button(
        master = frame,
        text = "Linear, increase",
        underline = 2
    )
    btn_linear_inc.grid(row = 2, column = 2)
    btn_linear_inc.bind('<ButtonRelease-1>', increase_linear)

    # Bind all the required keys to functions
    root.bind('<n>', next_img)
    root.bind("<p>", prev_img)
    root.bind("<q>", quit_img)

    root.mainloop() # Start the GUI

if __name__ == "__main__":
    main()


import argparse
import cv2
import numpy as np
import os
import sys


# set up an a list for files
images = []

# index for the list of images in the browser
count = 0
# Current manipulation
manipul = ""

# Get and parse the arguments
def get_args():
    parser = argparse.ArgumentParser(description='Image browser v1.0')
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

    # If there is a problem with the given path, exit
    if len(images) == 0:
        print("Invalid path or there are no images in path")
        sys.exit(1)


# Load the first image from the directory as opencv
def opencv_img(count):
    # read and convert image
    image = cv2.imread(images[count])

    return(image)

# Reduce intensity to 6 bit 
def intensity_6(img):
    global manipul 
    new_img = (img // 4 * 4) + (4 // 2)    
    manipul  = "k6"
    #Update the display
    cv2.imshow("result", new_img)
    cv2.imwrite(images[count].rstrip(".jpg")+manipul+".jpg", img)
    cv2.waitKey(0)
    

# Reduce intensity to 4 bit 
def intensity_4(img):
    global manipul 
    img = opencv_img(count)
    new_img = (img // 16 * 16) + (16 // 2)
    manipul = "k4"
    cv2.imshow("result", new_img)
    cv2.imwrite(images[count].rstrip(".jpg")+manipul+".jpg", img)
    cv2.waitKey(0)
    

# Shrink using nearest neighbor with a factor or 0.5
def shrink_NN(img):
    global manipul 
    
    manipul = "nn_shrink"
    nearest_neighbor(img)

# Shirnk using bicubic with a factor or 0.5
def shrink_bicubic(img):
    
    global manipul 
    
    manipul = "bilin_shrink"
    bicubic(img, 0.5)

# Shirnk using bilinear with a factor or 0.5
def shrink_bilinear(img):
    global manipul 
    
    manipul = "bicube_shrink"
    bilinear(img, 0.5)

# Increase using nearest neighbor with a factor or 2
def increase_NN(img):
    global manipul 
    
    manipul = "nn_increase"
    nearest_neighbor(img,2)
    

# Increase using bicubic with a factor or 2
def increase_bicubic(img):
    global manipul 
    
    manipul = "bilin_increase"
    bicubic(img,2)

# Increase using bilinear with a factor or 2
def increase_bilinear(img):
    global manipul 
    
    manipul = "bicube_increase"
    bilinear(img, 2)

# Nearest neigbor interpolation to the given factor
def nearest_neighbor(image, factor=0.5):
    image = cv2.resize(image, (int(image.shape[1]*factor), 
                                      int(image.shape[0]*factor)), 
                                      interpolation=cv2.INTER_NEAREST)
    cv2.imshow("result", image)
    wait_save(image)

#Bicubic interpolation to the given factor
def bicubic(image, factor = 0.5):
    image = cv2.resize(image, (int(image.shape[1]*factor), 
                                      int(image.shape[0]*factor)), 
                                      interpolation=cv2.INTER_CUBIC)
    cv2.imshow("result", image)
    wait_save(image)

# Bilinear interpolation to the given factor
def bilinear(image, factor=0.5):
    image = cv2.resize(image, (int(image.shape[1]*factor), 
                                      int(image.shape[0]*factor)), 
                                      interpolation=cv2.INTER_LINEAR)
    cv2.imshow("result", image)
    wait_save(image)


def wait_save(img):
    cv2.waitKey(0)
    cv2.imwrite(images[count].rstrip(".jpg")+manipul+".jpg", img)
    

def main():

    global count
    global images
    #Get the command arguments
    args = get_args()

    load_path(args.path)
    
    print("First, lower intensity to see the effect of intensity resolution")
    while count < len(images):
        img = opencv_img(count)
        cv2.imshow("result", img)
        print("INITAL IMAGE")
        cv2.waitKey(0)
        print("k=4")
        intensity_4(img)
        print("k=6")
        intensity_6(img)
        count = count + 1
        
    print("Next, edit spacial resolution")
    count = 0
    images = []
    load_path(args.path)
        
    while count < len(images):
        img = opencv_img(count)
        cv2.waitKey(0)
        print("INITAL IMAGE")
        cv2.imshow("result", img)
        shrink_NN(img)
        increase_NN(img)
        shrink_bicubic(img)
        increase_bicubic(img)
        shrink_bilinear(img)
        increase_bilinear(img)
        count = count + 1
    
    print("end")

    

if __name__ == "__main__":
    main()

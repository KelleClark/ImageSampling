import argparse
import cv2
import numpy as np
import os
import sys
import math
from time import time
import matplotlib.pyplot as plt


# set up an a list for files
images = []

# index for the list of images in the browser
count = 0
# Current manipulation
manipul = ""

# Get and parse the arguments
def get_args():
    parser = argparse.ArgumentParser(description='Image Manipulaation v1.0')
    parser.add_argument('path', metavar='dir',
                        help='The root directory to view photos in')

    parser.add_argument('--faster', metavar='bool',  type=int,
                        help='The color algorithm speed: "1" uses a faster implementation')

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
    if args.faster == 1:
        new_img = (img // 4 * 4) + (4 // 2)
    else:
        new_img = change_intensity(img, 6)
    manipul  = "k6"
    #View and save the image
    show_wait_save(new_img)

# Reduce intensity to 4 bit
def intensity_4(img):
    global manipul
    img = opencv_img(count)
    if args.faster == 1:
        new_img = (img // 16 * 16) + (16 // 2)
    else:
        new_img = change_intensity(img, 4)
    manipul = "k4"
    #View and save the image
    show_wait_save(new_img)

def change_intensity(img, k):
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

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            changed_img[i, j,0] =  np.uint8(min(255,math.floor(np.double(normalized_img[i,j,0]*target_level)) * 0.999 * (2**(8-k))))
            changed_img[i, j,1] =  np.uint8(min(255,math.floor(np.double(normalized_img[i,j,1]*target_level)) * 0.999 * (2**(8-k))))
            changed_img[i, j,2] =  np.uint8(min(255,math.floor(np.double(normalized_img[i,j,2]*target_level)) * 0.999 * (2**(8-k))))

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

    return(changed_img)


# Shrink using nearest neighbor with a factor or 0.5
def shrink_NN(img):
    global manipul
    manipul = "nn_shrink"
    start = time()
    nearest_neighbor(img)
    end = time()
    print("Shrinking using Nearest Neighbor took: ", str((end-start)/60), " seconds")

# Shirnk using bicubic with a factor or 0.5
def shrink_bicubic(img):
    global manipul
    manipul = "bilin_shrink"
    start = time()
    bicubic(img, 0.5)
    end = time()
    print("Shrinking using bicubic took: ", str((end-start)/60), " seconds")

# Shirnk using bilinear with a factor or 0.5
def shrink_bilinear(img):
    global manipul
    manipul = "bicube_shrink"
    start = time()
    bilinear(img, 0.5)
    end = time()
    print("Shrinking using bilinear took: ", str((end-start)/60), " seconds")

# Increase using nearest neighbor with a factor or 2
def increase_NN(img):
    global manipul
    manipul = "nn_increase"
    start = time()
    nearest_neighbor(img,2)
    end = time()
    print("Enlarging using Nearest Neighbors took: ", str((end-start)/60),  " seconds")


# Increase using bicubic with a factor or 2
def increase_bicubic(img):
    global manipul
    manipul = "bilin_increase"
    start = time()
    bicubic(img,2)
    end = time()
    print("Enlarging using Bicubic took: ", str((end-start)/60), " seconds")


# Increase using bilinear with a factor or 2
def increase_bilinear(img):
    global manipul
    manipul = "bicube_increase"
    start = time()
    bilinear(img, 2)
    end = time()
    print("Enlarging using Bilinear took: ", str((end-start)/60),  " seconds")

# Nearest neigbor interpolation to the given factor
def nearest_neighbor(image, factor=0.5):
    image = cv2.resize(image, (int(image.shape[1]*factor),
                                      int(image.shape[0]*factor)),
                                      interpolation=cv2.INTER_NEAREST)
    #Show and save the image
    show_wait_save(image)

#Bicubic interpolation to the given factor
def bicubic(image, factor = 0.5):
    image = cv2.resize(image, (int(image.shape[1]*factor),
                                      int(image.shape[0]*factor)),
                                      interpolation=cv2.INTER_CUBIC)
    #Show and save the image
    show_wait_save(image)

# Bilinear interpolation to the given factor
def bilinear(image, factor=0.5):
    image = cv2.resize(image, (int(image.shape[1]*factor),
                                      int(image.shape[0]*factor)),
                                      interpolation=cv2.INTER_LINEAR)
    #Show and save the image
    show_wait_save(image)

# Enlarge image by factor of 2 using linear interpolation
def enlarge_linear(image):
    global manipul
    manipul = "linear_increase"
    start = time()
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
    end = time()
    print("Shrinking using linear took: ", str((end-start)/60),  " seconds")
    show_wait_save(enlarged_img)

# Shrink image by factor of 2 (or nearly 2 if odd dimension) using linear interpolation
def shrink_linear(image):
    global manipul
    manipul = "linear_shrink"
    start = time()
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
    end = time()
    print("Shrinking using linear took: ", str((end-start)),  " seconds")
    show_wait_save(shrunken_img)


# Display the given image, give the user time to view it, and save the image
#    to the main given path
def show_wait_save(img):
    cv2.imshow("result", img)
    cv2.imwrite(images[count].rstrip(".jpg")+manipul+".jpg", img)
    cv2.waitKey(0)

def main():

    global count
    global images
    global args
    #Get the command arguments
    args = get_args()

    load_path(args.path)

    print("First, lower intensity to see the effect of intensity resolution")

    #Change the intensity resolution for every image in the given path. The
    #   k values implemented are:
    #k = 4
    #k = 6
    while count < len(images):
        img = opencv_img(count)
        cv2.imshow("result", img)
        print("INITAL IMAGE:", images[count])
        cv2.waitKey(0)
        print("k=4")
        intensity_4(img)
        print("k=6")
        intensity_6(img)
        count = count + 1

    print("Next, edit spacial resolution")
    # Reset the counter and reload the images to include the new images
    count = 0
    images = []
    load_path(args.path)

    # Save each interpolation on every image in the given path -- including
    #   the intensity changed images. The inetpolations are:
    # Nearest Neighbor, Bicubic, Bilinear
    while count < len(images):  # for each image
        img = opencv_img(count)
        print("INITAL IMAGE:", images[count])
        cv2.waitKey(0)
        cv2.imshow("result", img)
        print("Linear shrink")
        shrink_linear(img)
        print("Linear increase")
        enlarge_linear(img)
        print("NN shrink")
        shrink_NN(img)
        print("NN increase")
        increase_NN(img)
        print("Bicubic shrink")
        shrink_bicubic(img)
        print("Bicubic increase")
        increase_bicubic(img)
        print("Bilinear shrink")
        shrink_bilinear(img)
        print("Bilinear increase")
        increase_bilinear(img)
        count = count + 1

    print("end")



if __name__ == "__main__":
    main()

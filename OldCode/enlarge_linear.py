# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 21:50:30 2020

@author: sclar
"""

# Enlarge image by factor of 2 using linear interpolation
def enlarge_linear(event):
    global current_disp
    
    print("The shape of the image is "+ current_disp.shape[0]+" by "+current_disp.shape[1])
    
    # We make a new matrix (multi-dim array) of all zeros twice the width and height that holds RGB
    enlarged_img = np.zeros((2*current_disp.shape[0],2*current_disp.shape[1], 3),dtype=np.uint8)
    
    # For each row, go to each column position and if the col is divisible by 2, copy old img values
    # otherwise use linear interpolation with left and right spatial coordiante RGB values
    for i in range(current_disp.shape[0]*2):
        for j in range(current_disp.shape[1]*2):
            if j%2 == 0:
                enlarged_img[i,j] = current_disp[i,j]
            else:
                enlarged_img[i,j] = (1/2)*current_disp[i,j-1] + (1/2)*current_disp[i,j+1]
                
                
    # For each column, go to each row position and if the row is divisible by 2, copy old img values
    # otherwise use linear interpolation with above and below spatial coordiante RGB values            
    for j in range(current_disp.shape[1]*2):
        for i in range(current_disp.shape[0]*2):
            if i%2 ==0:
                enlarged_img[i,j] = current_disp[i,j]
            else:
                enlarged_img[i,j] = (1/2)*current_disp[i-1,j] + (1/2)*current_disp[i+1,j]              
                            
    imgtk = convert_img(enlarged_img)
    tex = extract_meta()
    update_window(imgtk,tex)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:47:22 2020

@author: kai
"""


import os
import numpy as np
import numpy.linalg as la
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# global parameter
intensity_mean = 128

def intensity2vec(I):
    return (I - intensity_mean)/intensity_mean

def normalize_image(I):
    
    
    return I/np.sqrt(np.sum(np.power(I,2)))

def get_kernel():
    '''
    This function returns vector image of red signal
    '''
    
    data_path = '../data/RedLights2011_Medium/'
    file_name = 'RL-010.jpg'
    
    n_row = 60
    n_col = 28
    tl_row = 27
    tl_col = 321
    br_row = tl_row + n_row
    br_col = tl_col + n_col
    
    
    bbox = [tl_row,tl_col,br_row,br_col]
    
    # crop image of a Red Light
    I = Image.open(data_path+file_name)
    I = np.asarray(I).astype('float')
    
    MAKE_SMOOTH = True
    target_not_smooth = I[tl_row:br_row,tl_col:br_col,:]
    
    if MAKE_SMOOTH:
        
        radius = 2 # smoothing radius (excluding center pixel)
        # target = I[tl_row:br_row,tl_col:br_col,:]
        target = np.zeros([n_row,n_col,3])
        for r in range(n_row):
            for c in range(n_col):
                rc = r+tl_row
                cc = c+tl_col
                for a in range(3):
                    target[r,c,a] = np.average(I[rc-radius:rc+radius,cc-radius:cc+radius,a])
                
    else:
        target = I[tl_row:br_row,tl_col:br_col,:]

    
    return normalize_image(target)

def heatmatp_threshold_min(im_heat,threshold):
    
    im_heat_new = im_heat
    idx = im_heat > threshold
    im_heat_new[im_heat < threshold] = threshold
    return im_heat_new, idx

def convolve(I,target):
    '''
    This function returns a heat map of same size as image. It reports float 
    value between 0 and 1 where 1 corresopnds to highest correlation.
    '''
    
    # get image size
    im_row = I.shape[0]
    im_col   = I.shape[1]
    
    # kernal size
    box_row = target.shape[0]
    box_col  = target.shape[1]
    
    row_range = im_row - box_row + 1
    col_range = im_col - box_col + 1
    
    # pre-allocate space for heatmap
    im_heat = np.zeros([im_row,im_col])
    
    # for each pixel, get dot product
    for r in range(row_range):
        for c in range(col_range):
            
            thisPatch = I[r:(r+box_row),c:(c+box_col)]
            thisPatch = normalize_image(thisPatch)
            val = np.sum(np.multiply(target,thisPatch))
            
            x_cord = int(r+(box_row/2))
            y_cord = int(c+(box_col/2))
            
            # store center coordinate of kernal location
            im_heat[x_cord,y_cord] = val
            

    # # normalization
    # maxval = np.amax(im_heat)
    # minval = np.amin(im_heat)    
    # im_heat = (im_heat-minval)/maxval
    
    return im_heat

def classify(I,target):
    
    
    # get heatmap
    im_heat = convolve(I,target)
    
    # threshold 
    threshold = 0.91
    im_heat_new, flagMatch = heatmatp_threshold_min(im_heat,threshold)
    
    
    nrow = target.shape[0]
    ncol = target.shape[1]
    
    r = 10 # number of pixel radius inside which is ONE red light
    sqr_width = 2*r +1
    ITR_MAX =1000
    
    bounding_boxes = []
    for i in range(ITR_MAX):
        
        # find detected indices
        idx = np.argwhere(flagMatch)
        
        if idx.size == 0: # if empty
            break
        
        # center row/column
        rc = idx[0,0]
        cc = idx[0,1]
        
        # find bounding box coordinates
        tl_row = rc-nrow/2
        tl_col = cc-ncol/2
        br_row = rc+nrow/2
        br_col = cc+ncol/2
        
        # add boundinig box
        bounding_boxes.append([tl_row,tl_col,br_row,br_col])
        
        # set neighbor values to False
        flagMatch[(rc-r):(rc+r),(cc-r):(cc+r)] = False
    
    return bounding_boxes, im_heat
    
def draw_bounding_box(bb):
    
    tl_row = bb[0]
    tl_col = bb[1]
    br_row = bb[2]
    br_col = bb[3]
    
    nrow = br_row-tl_row
    ncol = br_col-tl_col
    
    # Create a Rectangle patch
    rect = patches.Rectangle((tl_col,tl_row),ncol,nrow,linewidth=1,edgecolor='r',facecolor='none')

    return rect

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium/'
# file_name = 'RL-010.jpg'
# file_name = 'RL-011.jpg'
# file_name = 'RL-012.jpg'
# file_name = 'RL-092.jpg'
# file_name = 'RL-014.jpg'
# file_name = 'RL-177.jpg'
# file_name = 'RL-110.jpg'
file_name = 'RL-025.jpg'


# 10th image read image using PIL:
I_raw = Image.open(data_path+file_name)
I = np.asarray(I_raw).astype('float')

# get cropped image of signal
target = get_kernel()

# classify 
bounding_boxes, im_heat = classify(I,target)



fig,ax = plt.subplots(1)

ax.imshow(I_raw)

for bb in bounding_boxes:
    rect = draw_bounding_box(bb)
    ax.add_patch(rect)

plt.show()




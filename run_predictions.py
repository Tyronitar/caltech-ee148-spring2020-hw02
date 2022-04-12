import os
import json
from tkinter import W

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pyinstrument import Profiler

from utils import convolve, downsample, make_kernels, visualize

CENTER_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.6

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    heatmap = convolve(I, T)
    return heatmap


def predict_boxes(heatmap, T):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    h, w, _ = T.shape
    loc = np.where(heatmap >= CENTER_THRESHOLD)

    for pt in zip(*loc):
        toplr = pt[0] - h // 2
        toplc = pt[1] - w // 2
        botrr = pt[0] + h // 2
        botrc = pt[1] + w // 2
        conf = heatmap[toplr:botrr, toplc:botrc].mean()
        if conf > CONFIDENCE_THRESHOLD:
            output.append([pt[0] - h // 2, pt[1] - w // 2, pt[0] + h // 2, pt[1] + w // 2, conf])

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    kernels = make_kernels()
    output = []

    heatmap = np.zeros(I.shape[:2])
    for k in kernels:
        heatmap = compute_convolution(I, k)
        output.extend(predict_boxes(heatmap, k))
    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = 'data/RedLights2011_Medium'

# load splits: 
split_path = 'data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = 'data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
# for i in range(len(file_names_train)):
for i in range(1):

    # read image using PIL:
    # I = Image.open(os.path.join(data_path,file_names_train[i]))
    I = Image.open(os.path.join(data_path,"RL-010.jpg")).convert('HSV')
    small = downsample(I, 2)

    # convert to numpy array:
    small_arr = np.asarray(small)
    bounding_boxes = detect_red_light_mf(small_arr)
    visualize(small, bounding_boxes)

    # preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)

import os
import json

import numpy as np
from PIL import Image
import tqdm

from utils import convolve, downsample, make_kernels, merge_boxes, score_clustering

CENTER_THRESHOLD = 0.9
CONFIDENCE_THRESHOLD = 0.15
MIN_WIDTH = 5
MIN_HEIGHT = 5

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    heatmap = convolve(I, T)
    return heatmap


def predict_boxes(heatmap, T, max_size):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    # h, w, _ = T.shape
    max_h, max_w = max_size
    loc = np.where(heatmap >= CENTER_THRESHOLD)
    centers = zip(*loc)

    return score_clustering(heatmap, threshold=CENTER_THRESHOLD)


def detect_red_light_mf(I, sampling_factor=2):
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

    output = []

    heatmap = np.zeros(I.shape[:2])
    for k in kernels:
        heatmap = np.maximum(compute_convolution(I, k), heatmap)
    output = predict_boxes(heatmap, None, I.shape[:2])
    output  = merge_boxes(output)
    output = np.array(output)
    if len(output) > 0:
        output[:, :-1] *= sampling_factor
    return output.tolist()

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
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
SAMPLING_FACTOR = 3

kernels = make_kernels(sampling_factor=SAMPLING_FACTOR)

print("Generating Predicitons...\n")
for i in tqdm.tqdm(range(len(file_names_train))):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i])).convert('HSV')
    small = downsample(I, SAMPLING_FACTOR)

    # convert to numpy array:
    small_arr = np.asarray(small)

    bounding_boxes = detect_red_light_mf(small_arr, SAMPLING_FACTOR)

    preds_train[file_names_train[i]] = bounding_boxes

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    print("Generating Test Set Predictions...\n")
    for i in tqdm.tqdm(range(len(file_names_test))):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))
        small = downsample(I, SAMPLING_FACTOR)

        # convert to numpy array:
        small_arr = np.asarray(small)

        bounding_boxes = detect_red_light_mf(small_arr, SAMPLING_FACTOR)

        preds_test[file_names_test[i]] = bounding_boxes

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)

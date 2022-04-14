import os
import json
from tkinter import W

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import tqdm

from utils import fuse_boxes, merge_boxes, visualize

def compute_iou(box_1, box_2, debug=False):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    b1 = np.array(box_1)
    b2 = np.array(box_2)

    xl = max(b1[0], b2[0])
    yt = max(b1[1], b2[1])
    xr = min(b1[2], b2[2])
    yb = min(b1[3], b2[3])

    if xr < xl or yb < yt:
        return 0.0

    int_area = (xr - xl) * (yb - yt)

    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])

    iou = int_area / (a1 + a2 - int_area)
    if debug:
        print(int_area)
        print(a1)
        print(a2)
        print(iou)

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        # if len(gt) > 2:
        #     gt = fuse_boxes(gt)
        # if pred_file == 'RL-010.jpg':
        #     I = Image.open(os.path.join(data_path,"RL-010.jpg")).convert('HSV')
        #     visualize(I, pred)
        #     visualize(I, gt)
        pred = np.array(pred)
        pred = pred[pred[:, 4] > conf_thr, :].tolist()
        for i in range(len(gt)):
            if len(pred) == 0:
                FN += 1
                continue
            iou_func = lambda box: compute_iou(gt[i], box[:4])
            ious = np.array([*map(iou_func, pred)])
            best_pred = ious.argmax()
            best_iou = ious[best_pred]
            # best_pred = 0
            # best_iou = 0
            # for j in range(len(pred)):
            #     iou = compute_iou(pred[j][:4], gt[i])
            #     if iou > best_iou:
            #         best_iou = iou
            #         best_pred = j
            # if pred_file == 'RL-010.jpg':
            #     compute_iou(pred[best_pred][:4], gt[i], debug=True)
            #     print(f"gt: {gt[i]}\n--best: {pred[best_pred]}, id: {best_pred} \n--iou: {best_iou}\n")
            #     print(gt[i])
            #     print(pred[best_pred])
            if best_iou > iou_thr:
                TP += 1
                del pred[best_pred]
            else:
                FN += 1
        FP += len(pred)


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

if __name__ == '__main__':
    # set a path for predictions and annotations:
    preds_path = 'data/hw02_preds'
    data_path = 'data/RedLights2011_Medium'
    gts_path = 'data/hw02_annotations'

    # load splits:
    split_path = 'data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

    # Set this parameter to True when you're done with algorithm development:
    done_tweaking = False

    '''
    Load training data. 
    '''
    with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
        preds_train = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)

    if done_tweaking:
        
        '''
        Load test data.
        '''
        
        with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
            preds_test = json.load(f)
            
        with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
            gts_test = json.load(f)


    # For a fixed IoU threshold, vary the confidence thresholds.
    # The code below gives an example on the training set for one IoU threshold. 


    # confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)[:, 4]) # using (ascending) list of confidence scores as thresholds
    confidence_thrs = []
    for fname in preds_train:
        for box in preds_train[fname]:
            confidence_thrs.append(box[4])
    confidence_thrs = np.sort(np.array(confidence_thrs))
    confidence_thrs = confidence_thrs[:-len(confidence_thrs) // 75:len(confidence_thrs) // 75]
    print(confidence_thrs)
    # confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(tqdm.tqdm(confidence_thrs, total=len(confidence_thrs))):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

    # Plot training set PR curves


    total_preds = tp_train + fp_train + fn_train
    total_gts = 0
    for l in gts_train.values():
        total_gts += len(l)
    precision_train = tp_train / total_preds
    recall_train = tp_train / total_gts

    print(recall_train)
    print(precision_train)

    plt.plot(recall_train, precision_train)
    plt.show()

    if done_tweaking:
        print('Code for plotting test set PR curves.')

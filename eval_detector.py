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

        if len(pred) == 0:
            FN += len(gt)
            continue
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
    done_tweaking = True
    iou_thrs = [0.25, 0.5, 0.75]
    # iou_thrs = [0.05, 0.10, 0.15]  # Alternative Thresholds

    '''
    Load training data. 
    '''
    with open(os.path.join(preds_path,'preds_train_weak.json'),'r') as f:
        preds_train = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)

    if done_tweaking:
        
        '''
        Load test data.
        '''
        
        with open(os.path.join(preds_path,'preds_test_weak.json'),'r') as f:
            preds_test = json.load(f)
            
        with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
            gts_test = json.load(f)

    confidence_thrs = []
    for fname in preds_train:
        for box in preds_train[fname]:
            confidence_thrs.append(box[4])
    confidence_thrs = np.sort(np.array(confidence_thrs))
    confidence_thrs = confidence_thrs[:-len(confidence_thrs) // 75:len(confidence_thrs) // 75]

    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for iou_thr in iou_thrs:
        print(f"Computing Counts for IoU = {iou_thr}...")
        for i, conf_thr in enumerate(tqdm.tqdm(confidence_thrs, total=len(confidence_thrs))):
            tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr, conf_thr=conf_thr)

        # Plot training set PR curves

        total_preds = tp_train + fp_train + fn_train
        total_gts = 0
        for l in gts_train.values():
            total_gts += len(l)
        precision_train = tp_train / total_preds
        recall_train = tp_train / total_gts

        ids = np.where((precision_train != 0) | (recall_train != 0))[0]

        plt.plot(recall_train[ids], precision_train[ids], label=f"{iou_thr}")

    plt.title("PR Curve for Various IoU Thresholds")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"PR_train_weak_alt.png", bbox_inches='tight')
    plt.show()

    if done_tweaking:
        plt.clf()

        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for iou_thr in iou_thrs:
            print(f"Computing Test Set Counts for IoU = {iou_thr}...")
            for i, conf_thr in enumerate(tqdm.tqdm(confidence_thrs, total=len(confidence_thrs))):
                tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr, conf_thr=conf_thr)

            # Plot training set PR curves

            total_test_preds = tp_test + fp_test + fn_test
            total_gts = 0
            for l in gts_test.values():
                total_gts += len(l)
            precision_test = tp_test / total_test_preds
            recall_test = tp_test / total_gts

            print(precision_test)
            print(recall_test)
            ids = np.where((precision_test != 0) | (recall_test != 0))[0]

            plt.plot(recall_test[ids], precision_test[ids], label=f"{iou_thr}")

        plt.title("PR Curve for Various IoU Thresholds")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(f"PR_test_weak_alt.png", bbox_inches='tight')
        plt.show()

import os
import json

import numpy as np
from PIL import Image, ImageDraw

def show_pred_and_gt(fname: str):
    I = Image.open(os.path.join(data_path,fname))
    if fname in file_names_train:
        pred = preds_train[fname]
        gt = gts_train[fname]
    else:
        pred = preds_test[fname]
        gt = gts_test[fname]

    img = ImageDraw.Draw(I)
    for box in pred:
        draw_box = (box[1], box[0], box[3], box[2])
        img.rectangle(draw_box, outline='red', width=5)
    for box in gt:
        draw_box = (box[1], box[0], box[3], box[2])
        img.rectangle(draw_box, outline='green', width=5)
    I.show()
    I.save(f"{fname[:-4]}_viz.jpg")


if __name__ == '__main__':
    # set a path for predictions and annotations:
    preds_path = 'data/hw02_preds'
    data_path = 'data/RedLights2011_Medium'
    gts_path = 'data/hw02_annotations'

    # load splits:
    split_path = 'data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))


    '''
    Load training data. 
    '''
    with open(os.path.join(preds_path,'preds_train_weak.json'),'r') as f:
        preds_train = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
        gts_train = json.load(f)

    with open(os.path.join(preds_path,'preds_test_weak.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)
    
    imgs = [4, 9, 10, 26]
    for i in imgs:
        show_pred_and_gt(f"RL-{i:03}.jpg")
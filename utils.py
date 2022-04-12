from operator import mod
import os
from tkinter import W

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

DATA_PATH = 'data/RedLights2011_Medium'

KERNEL_BOXES = {
        "RL-010.jpg": [[122, 13, 173, 84], [320, 26, 349, 92]],
        "RL-036.jpg": [[216, 149, 232, 171], [296, 163, 305, 183]],
        "RL-050.jpg": [[335, 123, 348, 155]],
        # "RL-248.jpg": [[498, 130, 518, 172]],
        # "RL-274.jpg": [[315, 232, 322, 248]],
}

def normalize_arr(arr: np.ndarray) -> np.ndarray:
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')

    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr.astype("uint8")


def normalize_img(I: Image.Image) -> Image.Image:
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = np.asarray(I)
    return Image.fromarray(normalize_arr(arr).astype("uint8"), "RGB")


def downsample(I: Image.Image, s: float = 1.0) -> Image.Image:
    w, h, = I.size
    newsize = (int(w // s), int(h // s))
    return I.resize(newsize)


def convolve(A: np.ndarray, B: np.ndarray, mode='cosine') -> np.ndarray:
    assert mode in ['cosine', 'dot']

    rows, cols, n_chan = A.shape
    h, w, _ = B.shape
    result = np.zeros(A.shape[:2])
    b_norm = np.linalg.norm(B)

    # Zero pad A
    left_pad = (w - 1) // 2
    right_pad = w - 1 - left_pad
    top_pad = (h - 1) // 2
    bot_pad = h - 1 - top_pad
    padded = np.zeros((rows + top_pad + bot_pad, cols + left_pad + right_pad, n_chan))
    padded[top_pad:top_pad + rows, left_pad:left_pad + cols, :] = A

    for i in range(rows):
        for j in range(cols):
            window = padded[i:i + h, j:j + w, :]
            if mode == 'cosine':
                result[i, j] = np.einsum('ijk,ijk', B, window) / (b_norm * np.linalg.norm(window))
            else:
                result[i, j] = np.einsum('ijk,ijk', B, window)
    
    return result
        

def make_kernels(sampling_factor=2) -> list[np.ndarray]:
    kernels = []
    for img, boxes in KERNEL_BOXES.items():
        I = Image.open(os.path.join(DATA_PATH, img))
        for i, box in enumerate(boxes):
            light_image = I.crop(tuple(box)).convert('HSV')
            if i == 4:
                k_img = downsample(light_image, 0.5 * sampling_factor)
                kernels.append(np.asarray(light_image))
            if i < 2:
                k_img = downsample(light_image, 1.5 * sampling_factor)
                kernels.append(np.asarray(k_img))
            k_img = downsample(light_image, sampling_factor)
            kernels.append(np.asarray(k_img))
    
    # kernels.append(np.asarray(downsample(Image.fromarray(kernels[0], mode='HSV'), 2)))
    # kernels.append(np.asarray(downsample(Image.fromarray(kernels[1], mode='HSV'), 2)))
    # kernels.append(np.asarray(downsample(Image.fromarray(kernels[4], mode='HSV'), 0.5)))

    return kernels


def visualize(I: Image.Image,
bounding_boxes: list[list[int]],
outline: str = "red",
save=None) -> None:
    """Visualize the bounding boxes in the image"""
    I = I.convert('RGB')
    img = ImageDraw.Draw(I)
    for box in bounding_boxes:
        draw_box = (box[1], box[0], box[3], box[2])
        img.rectangle(draw_box, outline=outline)
    I.show()
    if save is not None:
        I.save(save)


def nsm(boxes: list[list[int]], threshold: float=0.4) -> list[list[int]]:
    """Merge overlapping boxes with non-maximum suppression"""
    out = []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    conf = boxes[:, 4]  # confidence scores

    pick = []  # The indices of the boxes to choose

    # Compute the area of the bounding boxes and sort the bounding boxes by their
    # confidence levels.
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # The indices of all boxes at start
    indices = np.argsort(conf)

    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        pick.append(i)

        # Find the coordinates of the intersection box
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        # Find the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (w * h) / areas[indices[:last]]

        # Delete redundant boxes
        indices = np.delete(indices, last)
        indices = np.delete(indices, np.where(overlap > threshold)[0])
    return boxes[pick].astype(int).tolist()

def fuse_boxes(boxes: list[list[int]], threshold: float=0.4) -> list[list[int]]:
    # Merge boxes that are completely contianed in one another
    out = []
    boxes = np.array(boxes)
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    conf = boxes[:, 4]  # confidence scores

    pick = []

    # The indices of all boxes at start
    indices = np.argsort(y2)

    for i in indices:
        temp_indices = indices[indices != i]
        contained = np.where((x1[temp_indices] <= x1[i]) & (y1[temp_indices] <= y1[i]) & \
                             (x2[temp_indices] >= x2[i]) & (y2[temp_indices] >= y2[i]))
        if not contained[0].any():
            pick.append(i)
    
    return boxes[pick].astype(int)


def merge_boxes(boxes: list[list[int]], threshold: float=0.4) -> list[list[int]]:
    if len(boxes) < 2:
        return boxes

    fused = fuse_boxes(boxes)
    suppressed = nsm(fused)

    return suppressed

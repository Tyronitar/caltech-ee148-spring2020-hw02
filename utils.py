import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

DATA_PATH = 'data/RedLights2011_Medium'

KERNEL_BOXES = {
        "RL-010.jpg": [[122, 13, 173, 84], [320, 26, 349, 92]],
        "RL-036.jpg": [[216, 149, 232, 171], [296, 163, 305, 183]],
        "RL-050.jpg": [[335, 123, 348, 155]],
        "RL-248.jpg": [[498, 130, 518, 172]],
        "RL-274.jpg": [[315, 232, 322, 248]],
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


def downsample(I: Image.Image, s: int = 1) -> Image.Image:
    w, h, = I.size
    newsize = (w // s, h // s)
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
        

def make_kernels() -> list[np.ndarray]:
    kernels = []
    for img, boxes in KERNEL_BOXES.items():
        I = Image.open(os.path.join(DATA_PATH, img))
        for box in boxes:
            k_img = I.crop(tuple(box)).convert('HSV')
            k_img = downsample(k_img, 2)
            kernels.append(np.asarray(k_img))
    
    kernels.append(np.asarray(downsample(Image.fromarray(kernels[0], mode='HSV'), 2)))
    kernels.append(np.asarray(downsample(Image.fromarray(kernels[1], mode='HSV'), 2)))

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


def merge_boxes(boxes: list[list[int]]) -> list[list[int]]:
    out = []

    return out

import numpy as np

from utils import convolve

ERROR_MARGIN = 1e-4


def is_close(a, b):
    return abs(a - b) < ERROR_MARGIN


def test_convolve_dot():
    k = np.array([[1, 2, 3],
                  [-4, 7, 4],
                  [2, -5, 1]
                ])
    k = k[:, :, np.newaxis]
    I = np.array([
        [2, 4, 9, 1, 4],
        [2, 1, 4, 4, 6],
        [1, 1, 2, 9, 2],
        [7, 3, 5, 1, 3],
        [2, 3, 4, 8, 5]
    ])
    I = I[:, :, np.newaxis]
    res = convolve(I, k, mode='dot')
    correct = np.array([
        [21, 59, 37, -19, 2],
        [30, 51, 66, 20, 43],
        [-14, 31, 49, 101, -19],
        [59, 15, 53, -2, 21],
        [49, 57, 64, 76, 10],
    ])
    assert (res == correct).all()

def test_convolve_cosine():
    k = np.array([[1, 2, 3],
                  [4, 7, 4],
                  [2, 5, 1]
                ])
    k = k[:, :, np.newaxis]
    I = k.copy()
    res = convolve(I, k, mode='cosine')

    assert is_close(res[1, 1], 1.0)
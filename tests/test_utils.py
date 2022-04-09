import numpy as np

from utils import convolve


# def test_convolve():
#     k = np.array([[1, 2, 3],
#                   [-4, 7, 4],
#                   [2, -5, 1]
#                 ])
#     k = k[:, :, np.newaxis]
#     I = np.array([
#         [2, 4, 9, 1, 4],
#         [2, 1, 4, 4, 6],
#         [1, 1, 2, 9, 2],
#         [7, 3, 5, 1, 3],
#         [2, 3, 4, 8, 5]
#     ])
#     I = I[:, :, np.newaxis]
#     res = convolve(I, k)
#     correct = np.array([
#         [21, 59, 37, -19, 2],
#         [30, 51, 66, 20, 43],
#         [-14, 31, 49, 101, -19],
#         [59, 15, 53, -2, 21],
#         [49, 57, 64, 76, 10],
#     ])
#     assert (res == correct).all()
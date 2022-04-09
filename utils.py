import numpy as np

def convolve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    rows, cols, _ = A.shape
    h, w, _ = B.shape
    window_indices = np.indices(B.shape[:2])
    result = np.zeros(A.shape[:2])
    b = B.ravel()
    b_norm = np.linalg.norm(b)

    # Zero pad A
    left_pad = (w - 1) // 2
    right_pad = w - 1 - left_pad
    top_pad = (h - 1) // 2
    bot_pad = h - 1 - top_pad
    padded = np.pad(A, ((top_pad, bot_pad), (left_pad, right_pad), (0, 0)), 'constant')

    for i in range(rows):
        for j in range(cols):
            row_ids = window_indices[0] + i
            col_ids = window_indices[1] + j
            
            window = padded[row_ids, col_ids, :]
            result[i, j] = np.dot(b, window.ravel()) / (b_norm * np.linalg.norm(window))
    
    return result
        
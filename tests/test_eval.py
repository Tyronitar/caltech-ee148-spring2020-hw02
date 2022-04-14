import numpy as np
import pytest

from eval_detector import compute_counts, compute_iou
from test_utils import is_close

@pytest.mark.parametrize(
    "box1, box2, iou",
    [
        ([1, 10, 20, 30], [1, 10, 20, 30], 1.0),
        ([1, 10, 20, 30], [1, 20, 20, 30], 0.5),
    ]
)
def test_iou(box1, box2, iou):

    assert is_close(compute_iou(box1, box2), iou)
import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

BoxArray = npt.NDArray[np.floating] | Sequence[Sequence[float]]


def compute_kappa(
    boxes_a: BoxArray,
    boxes_b: BoxArray,
    image_shape: tuple[int, int],
) -> float:
    """Compute Cohen's kappa between two sets of bounding boxes via pixel masks.

    Args:
        boxes_a: Ground-truth boxes as an (n, 4) array (or sequence) of
            [x1, y1, x2, y2].
        boxes_b: Predicted boxes as an (n, 4) array (or sequence) of
            [x1, y1, x2, y2].
        image_shape: (width, height) of the image.

    Returns:
        Cohen's kappa score, or 0.0 if undefined.
    """
    # Masks are (height, width) so the [y, x] slicing below indexes the right
    # axes; building them as (width, height) would transpose the boxes and clip
    # them on the wrong axis for non-square images.
    width, height = image_shape
    mask_gt = np.zeros((height, width), dtype=np.uint8)
    mask_pred = np.zeros((height, width), dtype=np.uint8)

    for bbox in boxes_a:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        mask_gt[y1:y2, x1:x2] = 1

    for bbox in boxes_b:
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        mask_pred[y1:y2, x1:x2] = 1

    y_true = mask_pred.ravel()
    y_pred = mask_gt.ravel()

    # Bug fix: len(y_true > 1) always equals len(y_true) (length of boolean array).
    # Correct check: whether there are more than 1 element in each array.
    if len(y_true) > 1 and len(y_pred) > 1:
        score = cohen_kappa_score(y_true, y_pred, labels=[0, 1])
    else:
        score = 0.0

    return float(score) if not np.isnan(score) else 0.0

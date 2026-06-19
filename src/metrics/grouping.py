"""Per-image row grouping shared by matching, scoring and the external backends.

The detection pipelines all need the same thing: split a DataFrame's rows by
``image_name`` and hand each image's boxes/labels to a numeric kernel. Doing this
with ``dict(tuple(df.groupby(...)))`` materialises a fresh sub-DataFrame per image
(pandas' ``_chop``) and then each per-image ``df[cols]`` access pays the indexer
cost again — together the dominant runtime on tens of thousands of images.

:func:`image_row_indices` does the grouping once and returns only *positional*
row indices, so callers extract each column to numpy a single time over the whole
frame and slice those arrays per image with plain fancy-indexing — no pandas in
the loop.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def image_row_indices(df: pd.DataFrame) -> dict[str, npt.NDArray[np.intp]]:
    """Map each ``image_name`` to its positional row indices (single pass).

    Returns ``{image_name: ndarray_of_int_positions}`` in first-appearance order.
    The positions index into the column arrays obtained via ``df[col].to_numpy()``
    / ``df[cols].to_numpy()``, letting callers slice per image without any
    per-group DataFrame construction.
    """
    return {
        str(name): np.asarray(idx, dtype=np.intp)
        for name, idx in df.groupby("image_name", sort=False).indices.items()
    }

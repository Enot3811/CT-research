"""Implementation of resizing of the N-dim ``numpy.ndarray``."""

from typing import Tuple

import numpy as np
import skimage


def resize(
    *data: np.ndarray,
    resize_shape: Tuple[int, ...],
    interpolation_order: int = 1,
    aspect_ratio: bool = False
) -> Tuple[np.ndarray, ...]:
    """
    Resize image to specified shape.
    
    If need to maintain aspect ratio then pad with zeros.

    Parameters
    ----------
    data : np.ndarray
        Original data array.
    resize_shape : Tuple[int]
        Shape for resizing.
    interpolation_order : int
        The order of interpolation. The order has to be in the range 0-5:
        0: Nearest-neighbor
        1: Bi-linear
        2: Bi-quadratic
        3: Bi-cubic
        4: Bi-quartic
        5: Bi-quintic
    aspect_ratio : bool
        Whether resize function maintains aspect ration.

    Returns
    -------
    np.ndarray
        Resized data.

    Raises
    ------
    ValueError
        Number of resizing shapes does not equal image shapes.
    """
    if len(resize_shape) != len(data[0].shape):
        raise ValueError(f'Number of resizing shapes {resize_shape} '
                         f'does not equal image shapes {data[0].shape}')

    resized_data = []
    for case in data:

        if aspect_ratio:
            scale = np.min(np.array(resize_shape) / np.array(case.shape))
            new_shape = tuple([int(shape * scale) for shape in case.shape])
        else:
            new_shape = resize_shape

        if case.dtype == bool:
            interpolation_order = 0
        if interpolation_order == 0:
            anti_aliasing = False
        else:
            anti_aliasing = True
        d_type = case.dtype

        resized_case = skimage.transform.resize(
            image=case, output_shape=new_shape, order=interpolation_order,
            preserve_range=True, anti_aliasing=anti_aliasing)
        resized_case = np.array(resized_case, dtype=d_type)

        # Expand crop if necessary
        shape_diff = np.array(resize_shape) - np.array(new_shape)
        assert np.all(shape_diff >= 0)
        before = (shape_diff // 2)
        after = (shape_diff - before)
        pad_width = list(zip(before.tolist(), after.tolist()))
        padded_data = np.pad(resized_case, pad_width,
                             mode='constant',
                             constant_values=np.min(case))
        assert padded_data.shape == resize_shape
        resized_data.append(padded_data)

    return tuple(resized_data)

"""Implementation of the random crop of the N-dim ``numpy.ndarray``."""

from typing import Tuple, Optional

import numpy as np


def crop_random(
    *data: np.ndarray,
    crop_shape: Tuple[int, ...],
    anchors: Optional[np.ndarray] = None,
    anchor_prob: float = 0.0
) -> Tuple[np.ndarray, ...]:
    """
    Crop given data sequence to given shapes randomly (same random for all).

    Algorithm randomly selects base point from given `anchors` or randomly.
    Then it samples crop around this point with small random perturbations.
    Padding is applied if got crop shape doesn't fit requested.
    It's performed via ``constant = np.min(data)``.

    Default behavior: sample random points across all data space and
    crop around them.

    Parameters
    ----------
    data : np.ndarray
        Volumes that will be cropped.
    crop_shape : Tuple[int]
        Shape for cropped data.
    anchors: np.ndarray
        Array of points which is requested to be inside crop (at least one)
        with specified probability.
    anchor_prob: float
        Probability with which at least one requested anchor will be inside
        the crop.

    Returns
    -------
    np.ndarray
        Cropped and expanded (if need) data to specified shape.

    Raises
    ------
    ValueError
        The dimension of the crop does not match the dimension of the data.
    ValueError
        One of dimensions of crop shape is bigger than the original one.
    """
    init_shape = data[0].shape
    for case in data[1:]:
        if case.shape != init_shape:
            raise ValueError(
                f'One of the data cases passed to crop has different shape: '
                f'{init_shape=}, {case.shape=}'
            )

    if len(init_shape) != len(crop_shape):
        raise ValueError(
            f'The dimension of the {crop_shape=} does not match' +
            f'the dimension of the {init_shape=}'
        )

    if anchors is not None:
        if anchors.dtype != np.int32:
            raise ValueError(f'Anchors have unexpected (not np.int32) type: '
                             f'{anchors.dtype}.')
        if np.min(anchors) < 0:
            raise ValueError('At least one anchor value is less than 0.')
        if not np.all(anchors < np.array(init_shape)):
            raise ValueError(f'At least one anchor value more than '
                             f'{init_shape=}.')
        if np.max(anchors) <= 1:
            raise ValueError('All anchors are less or equal than 1.')

    select_prob = np.random.uniform()
    if anchors is not None and select_prob < anchor_prob:
        rpoint_id = np.random.choice(range(anchors.shape[0]))
        rpoint = anchors[rpoint_id]
    else:
        rpoint = np.random.randint(
            np.zeros_like(init_shape),
            np.array(init_shape)
        )

    slices = []
    for i in range(len(crop_shape)):
        # do small shifts to make anchor sampling more random
        random_offset = np.random.randint(-crop_shape[i] // 10,
                                          crop_shape[i] // 10)
        crop_center = rpoint[i] + random_offset
        st_coord = max(0, crop_center - crop_shape[i] // 2)
        end_coord = min(init_shape[i], crop_center + crop_shape[i] // 2)
        slices.append(slice(st_coord, end_coord))

    crops = []
    for case in data:
        crop: np.ndarray = case[tuple(slices)]
        # Expand crop if necessary
        shape_diff = np.array(crop_shape) - np.array(crop.shape)
        assert np.all(shape_diff >= 0)
        before = (shape_diff // 2)
        after = (shape_diff - before)
        pad_width = list(zip(before.tolist(), after.tolist()))
        crop = np.pad(crop, pad_width,
                      mode='constant',
                      constant_values=np.min(case))
        assert crop.shape == crop_shape, f'{crop.shape=} : {crop_shape}'
        crops.append(crop)

    return tuple(crops)

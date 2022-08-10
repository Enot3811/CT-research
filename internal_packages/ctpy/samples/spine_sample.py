"""
Spine sample data storing structure.

This structure has only CT data, resolution data and semantic mask,
no annotation or other additional information.
"""

from typing import Dict, Tuple

import numpy as np

from ndarray_ext.ops import resize

from .ctsample import CTSample


class SpineSample(CTSample):
    """
    Structure for storing one Spine-sample.
    """

    mask: np.ndarray
    """
    Mask with the spine IA-segmentation mask.
    """

    def __init__(self,
                 ct: np.ndarray,
                 mask: np.ndarray,
                 voxel_size: Dict[str, float],
                 name: str):
        super(SpineSample, self).__init__(ct, voxel_size, name)
        self.mask = mask

    def get_resized_with_mask(
        self,
        voxel_size: Tuple[float, float, float]
    ) -> 'SpineSample':
        """
        Change voxel size and get new SpineSample with resized CT and mask.

        Parameters
        ----------
        voxel_size : Tuple[float, float, float]
            New voxel size.

        Returns
        -------
        SpineSample
            SpineSample with new voxel size and resized CT and mask.
        """
        coefficients = tuple(map(lambda vs_old, vs_new: vs_old / vs_new,
                                 self.voxel_size_arr, voxel_size))
        new_shape = tuple(map(lambda sh, coef: int(sh * coef),
                              self.shape, coefficients))
        resized_ct = resize(self.ct, resize_shape=new_shape)[0]
        resized_mask = resize(self.mask, resize_shape=new_shape)[0]
        dict_voxel_size = {
            'i': voxel_size[0],
            'j': voxel_size[1],
            'k': voxel_size[2]
        }
        return SpineSample(resized_ct, resized_mask,
                           dict_voxel_size, self.name)

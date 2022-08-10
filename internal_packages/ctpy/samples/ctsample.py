"""
CT data storing structure.

This structure has only CT data and resolution data,
no annotation or other additional information.
"""

from typing import Dict, Tuple

import numpy as np

from ndarray_ext.ops import resize


class CTSample:
    """
    Structure for storing one CT-sample.
    """

    ct: np.ndarray
    """
    CT pixel data.
    """

    name: str
    """
    Name of CT file.
    """

    voxel_size: Dict[str, float]
    """
    Physical size of one voxel along each axis.
    Has three keys: ``i``, ``j``, ``k``.
    """

    @property
    def voxel_size_arr(self):
        """Get voxel-size of the sample as ``np.ndarray``."""
        return np.array([self.voxel_size['i'],
                         self.voxel_size['j'],
                         self.voxel_size['k']], dtype=np.float32)

    @property
    def shape(self):
        """Return shape of the CT-sample."""
        return self.ct.shape

    def __init__(self,
                 ct: np.ndarray,
                 voxel_size: Dict[str, float],
                 name: str):
        self.ct = ct
        self.voxel_size = voxel_size
        self.name = name

    def __repr__(self):
        """Get str-representation of the object."""
        return f'{self.name}: {self.voxel_size_arr.round(decimals=2)}'

    def get_resized(
        self,
        voxel_size: Tuple[float, float, float]
    ) -> 'CTSample':
        """
        Change voxel size and get new CTSample with resized CT.

        Parameters
        ----------
        voxel_size : Tuple[float, float, float]
            New voxel size.

        Returns
        -------
        CTSample
            CTSample with new voxel size and resized CT.
        """
        coefficients = tuple(map(lambda vs_old, vs_new: vs_old / vs_new,
                                 self.voxel_size_arr, voxel_size))
        new_shape = tuple(map(lambda sh, coef: int(sh * coef),
                              self.shape, coefficients))
        resized_ct = resize(self.ct, resize_shape=new_shape)[0]
        dict_voxel_size = {
            'i': voxel_size[0],
            'j': voxel_size[1],
            'k': voxel_size[2]
        }
        return CTSample(resized_ct, dict_voxel_size, self.name)

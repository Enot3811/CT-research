"""
CT sampler with augmentation implementation.

Uses simulation of the real augmentations to waste the time. Just for tests.
"""

from typing import Tuple, Generator

import numpy as np

from ctpy.h5_samplers import CTSampler


class CTAugmentedSampler(CTSampler):
    """
    Functor-generator that performs 3D/2D-CT-Scans random crops sampling.

    The main difference from base class: applying augmentations.

    Function inits with static params of generation.
    Each call creates python-generator object that yields necessary data in
    specified range.
    """

    def __init__(self, crop_shape: Tuple[int, int, int],
                 min_v: int = -1024, max_v: int = 4096):
        """
        Create instance of the functor-generator.

        Parameters
        ----------
        crop_shape : VolumeShapeType
            Shape of the volume crop.
        """
        super().__init__(crop_shape, min_v, max_v)

    def __call__(
            self, h5_path: str,
            start: int, end: int
    ) -> Generator[Tuple[np.ndarray, str], None, None]:
        """
        Create python-generator that yields random-crop of CT augmented data.

        Sampling is performed in ``[start, end)``.

        Parameters
        ----------
        h5_path : str
            The path to hdf5 file with data for the dataset.
        start : int
            The Index of sample from which generator starts reading.
        end : int
            The index of sample to which the generator reads.
            Not included.

        Yields
        ------
        Generator[Tuple[np.ndarray, str], None, None]:
            Loads the next CT-3D-sample: CT-scan & CT-name.
        """
        gen = super().__call__(h5_path, start, end)
        for read_data in gen:
            crop, name = read_data
            crop = self._do_augmentation(crop)
            yield crop, name

    @staticmethod
    def _do_augmentation(data: np.ndarray) -> np.ndarray:
        """
        Do some augmentation. Hard-written code just for tests right now.

        Parameters
        ----------
        data : np.ndarray
            Data that will be augmented.

        Returns
        -------
        np.ndarray
            Augmented data.
        """
        # Some hard calculations just for time
        for _ in range(2):
            np.unique(data)
        return data

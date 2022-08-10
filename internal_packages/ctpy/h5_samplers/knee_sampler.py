"""
2D Classification sampler for knee X-ray dataset.

It can be used as a generator that yields 2D image of X-ray and its label.
"""

import random
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf

from ndarray_ext.ops import resize
from tf_ext import TFH5Generator

from ..serializers import KneeSampleSerialization


class KneeSampler(TFH5Generator):
    """
    Classification sampler for knee X-ray dataset.

    This sampler does the following:
    1) Iterates over h5 dataset in specified range from start to end index;
    2) Serializes each iterated sample to KneeSample;
    3) Yields 2D X-ray with its label.
    """

    @property
    def output_signature(self) -> Tuple[tf.TensorSpec, ...]:
        """Return output signature of Knee-ds."""
        return (
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int16)
        )

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        aspect_ratio: bool = False,
        binary_classification: bool = False,
        shuffle: bool = False
    ):
        """
        Create instance of the functor-generator.

        Parameters
        ----------
        data_shape : Tuple[int, ...]
            Shape for resize data.
        aspect_ratio : bool
            Whether resize function maintains aspect ration.
        binary_classification : bool, optional
            Whether the classification is binary.
            If set True all labels except '0' will be specified as '1'.
        shuffle : bool
            Need to shuffle samples order or not.
        """
        self.binary_classification = binary_classification
        if binary_classification:
            self.num_classes = 2
        else:
            self.num_classes = 5
        self.shuffle = shuffle
        self.data_shape = data_shape
        self.aspect_ratio = aspect_ratio
        self.serialization_manager = KneeSampleSerialization()

    def __call__(
            self, h5_path: Path, start: int, end: int
    ) -> TFH5Generator.GeneratorType[Tuple[np.ndarray, int]]:
        """
        Create generator that yields 2D X-ray image with label.

        Parameters
        ----------
        h5_path : Path
            The path to hdf5 file with data for the dataset.
        start : int
            The Index of sample from which generator starts reading.
        end : int
            The index of sample to which the generator reads.
            Not included.

        Yields
        ------
        TFH5Generator.GeneratorType[Tuple[np.ndarray, int]]:
            Sampler generator that yields X-ray images with their labels.
        """
        ds = h5py.File(h5_path, 'r')

        sample_inds = list(range(start, end))
        if self.shuffle:
            random.shuffle(sample_inds)

        for i in sample_inds:
            knee_sample = self.serialization_manager.from_h5(ds, i)
            if self.binary_classification:
                label = min(1, knee_sample.kl_label)
            else:
                label = knee_sample.kl_label
            xray = knee_sample.xray
            if xray.shape != self.data_shape:
                xray = resize(xray,
                              resize_shape=self.data_shape,
                              aspect_ratio=self.aspect_ratio)[0]
            xray = xray.astype(np.float32)
            yield xray, label

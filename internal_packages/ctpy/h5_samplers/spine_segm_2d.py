"""
Sampler that creates generator of 2D spine segmentation data.

2D spine segmentation data = CT-slice & its binary mask (where 1 is a spine).
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import tensorflow as tf

from ndarray_ext.ops import resize

from .sampler import Sampler
from .types import SliceShape
from ..serializers import SpineSampleSerialization


class SpineSGMSampler2D(Sampler):
    """
    2D segmentation sampler for CT Spine dataset.

    This sampler does the following:
    | 1) Iterates over h5 dataset in specified range from start to end index;
    | 2) Serializes each iterated sample to ``SpineSample``;
    | 3) Selects data_per_sample slices from current ``SpineSample``;
    | 4) Defines what slice is select using probs: empty, target or random.
    If it is not possible to select slice of chosen type (may be that some
    cases won't have empty or target at all), then selects what exists.
    | 5) Selected slice and mask will be resized to specified
    ``resize_shape`` size. Resize will be done with saving sides relation.
    Slice and mask will be padded with zeros if needed.
    | 6) Yields resized slice and mask.
    """

    @property
    def output_signature(self) -> Tuple[tf.TensorSpec, ...]:
        """Return output signature of this CT-reader."""
        return (
            tf.TensorSpec(shape=(None, None), dtype=tf.int16),
            tf.TensorSpec(shape=(None, None), dtype=tf.bool)
        )

    def __init__(self, resize_shape: SliceShape,
                 probs: Tuple[float, float, float],
                 n: int, shuffle: bool = False):
        """
        Create instance of the functor-generator.

        Parameters
        ----------
        resize_shape : SliceShapeType
            Shape for resizing selected slices.
        probs : Tuple[float, ...]
            Tuple containing probabilities for slice types that may be
            selected:
            | 1) Probability to select empty slice;
            | 2) Probability to select slice which contains true label;
            | 3) Probability to select slice randomly;
        n : int
            Count of samples to generate from each raw sample.
        shuffle : bool, optional
            Shuffle samples range before sampling start.

        Raises
        ------
        ValueError
            Sum of specified probs does not equal 1.0.
        """
        super().__init__(n, shuffle)
        self.resize_shape = resize_shape
        eps = 0.0001
        if np.abs(np.sum(probs) - 1.0) > eps:
            raise ValueError(f'Sum of probs "{probs}" does not equal 1.0.')
        self.probs = probs
        self.serialize_manager = SpineSampleSerialization()

    def __call__(
            self, h5_path: Path, start: int, end: int
    ) -> Sampler.GeneratorType[Tuple[np.ndarray, np.ndarray]]:
        """
        Create generator that yields 2D slices with binary label.

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
        Sampler.GeneratorType[Tuple[np.ndarray, np.ndarray]]:
            Sampler generator that yields pairs: CT slice & its spine mask.

        Raises
        ------
        RuntimeError
            Slice selection error.
        """
        ds = h5py.File(h5_path, 'r')
        if ds.attrs['space'] != 'RAS':
            raise IOError(
                f'Unexpected patient space: {ds.attrs["space"]}'
            )

        for i in self._get_sampler_range(start, end):
            spine_sample = self.serialize_manager.from_h5(ds, i)
            ct = spine_sample.ct
            mask = spine_sample.mask > 0

            target_slices_mask = np.any(mask, axis=(0, 1))
            targets = np.argwhere(target_slices_mask)
            #
            empty_slices_mask = np.logical_not(target_slices_mask)
            empty = np.argwhere(empty_slices_mask)
            #
            random = np.arange(mask.shape[0])

            # Define in-functions for slice selections

            def random_slice_selector():
                _indx = np.random.randint(0, len(random))
                return _indx

            def target_slice_selector():
                if len(targets) > 0:
                    _indx = np.random.randint(0, len(targets))
                    return _indx
                else:
                    return random_slice_selector()

            def empty_slice_selector():
                if len(empty) > 0:
                    _indx = np.random.randint(0, len(empty))
                    return _indx
                else:
                    return random_slice_selector()

            # Sample while sampled slices from case less then required
            for _ in range(self.n):
                sample_variant = np.random.choice(
                    ['empty', 'target', 'random'], p=self.probs)
                if sample_variant == 'random':
                    indx = random_slice_selector()
                elif sample_variant == 'target':
                    indx = target_slice_selector()
                elif sample_variant == 'empty':
                    indx = empty_slice_selector()
                else:
                    raise RuntimeError('Slice selection error.')
                ct_slice = ct[..., indx]
                mask_slice = mask[..., indx]
                assert mask_slice.dtype == np.bool_
                assert ct_slice.dtype == np.int16
                yield resize(
                    ct_slice, mask_slice,
                    resize_shape=self.resize_shape,
                    aspect_ratio=True
                )

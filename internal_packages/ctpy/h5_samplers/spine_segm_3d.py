"""
3D segmentation sampler for CT Spine dataset.

This sampler may be used for binary segmentation task:
mark vertebra pixels on CT volume.

It can be used as a generator that yields 3D tile from CT and its 3D mask.
"""

from pathlib import Path
from typing import Tuple, Optional

import h5py
import numpy as np
import tensorflow as tf

from ndarray_ext.ops import crop_random, resize

from .sampler import Sampler
from .types import VolumeShape
from ..serializers import SpineSampleSerialization


class SpineSGMSampler3D(Sampler):
    """
    3D segmentation sampler for CT Spine dataset.

    This sampler does the following:
    | 1) Iterates over h5 dataset in specified range from start to end index;
    | 2) Serializes each iterated sample to `SpineSample`;
    | 3) Resizes CT volume from `SpineSample` according to specified
    voxel size. Does nothing if there is no set voxel size specification.
    | 4) Selects ``data_per_sample`` tiles from current `SpineSample`;
    | 5) Yields tile and mask.
    """

    @property
    def output_signature(self) -> Tuple[tf.TensorSpec, ...]:
        """Return output signature of this CT-reader."""
        return (
            tf.TensorSpec(shape=(None, None, None), dtype=tf.int16),
            tf.TensorSpec(shape=(None, None, None), dtype=tf.bool)
        )

    def __init__(
        self,
        crop_shape: VolumeShape,
        probs: Tuple[float, float, float],
        n: int, shuffle: bool = False,
        voxel_size: Optional[Tuple[float, float, float]] = None,
        attempts_number: int = 10
    ):
        """
        Create instance of the functor-generator.

        Parameters
        ----------
        crop_shape : VolumeShape
            Returned 3D tile shape.
        probs : Tuple[float, float, float]
            Tuple containing probabilities for slice types that may be
            selected:
            | 1) Probability to select empty tile;
            | 2) Probability to select tile which contains true label;
            | 3) Probability to select tile randomly;
        n : int
            Count of samples to generate from each raw sample.
        shuffle : bool, optional
            Shuffle samples range before sampling start.
        voxel_size: Tuple[float, float, float], optional
            Voxel size of sampled data. No resize if ``None``.
        attempts_number : int
            Number of attempts to find an element to requested select.

        Raises
        ------
        ValueError
            Sum of specified probs does not equal 1.0.
        """
        super().__init__(n, shuffle)
        self.crop_shape = crop_shape
        self.voxel_size = voxel_size
        self.attempts_number = attempts_number
        eps = 0.0001
        if np.abs(np.sum(probs) - 1.0) > eps:
            raise ValueError(f'Sum of probs "{probs}" does not equal 1.0.')
        self.probs = probs
        self.serialize_manager = SpineSampleSerialization()

    def __call__(
            self, h5_path: Path, start: int, end: int
    ) -> Sampler.GeneratorType[Tuple[np.ndarray, np.ndarray]]:
        """
        Create instance of the functor-generator.

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
            Sampler generator that yields CT tiles with their masks.

        Raises
        ------
        RuntimeError
            Tile selection error.
        """
        ds = h5py.File(h5_path, 'r')
        if ds.attrs['space'] != 'RAS':
            raise IOError(
                f'Unexpected patient space: {ds.attrs["space"]}'
            )

        for i in self._get_sampler_range(start, end):
            spine_sample = self.serialize_manager.from_h5(ds, i)
            ct = spine_sample.ct
            mask = spine_sample.mask

            # Resize data to requested voxel size if need
            if self.voxel_size is not None:
                new_shape = np.round(
                    np.array(ct.shape) * np.array(spine_sample.voxel_size) /
                    np.array(self.voxel_size)
                ).astype(np.int32)
                ct = resize(
                    ct, resize_shape=new_shape, interpolation_order=1
                )
                mask = resize(
                    mask, resize_shape=new_shape, interpolation_order=0
                )

            # Make mask binary
            mask = mask > 0

            # If 'target' sampling prob is not zero-close
            # Then find all positions where mask is True & use for sampling
            if self.probs[1] > 1e-4:
                targets = np.argwhere(mask).astype(np.int32)
            else:
                targets = None
            
            for _ in range(self.n):
                sample_variant = np.random.choice(
                    ['empty', 'target', 'random'], p=self.probs)
                if sample_variant == 'random':
                    yield crop_random(ct, mask, crop_shape=self.crop_shape)
                elif sample_variant == 'target':
                    yield crop_random(ct, mask, crop_shape=self.crop_shape,
                                      anchors=targets, anchor_prob=1.0)
                elif sample_variant == 'empty':
                    # Try to find empty tile for several iterations
                    # Yield random in case of failed search
                    failed_attempts = 0
                    while True:
                        ct_tile, mask_tile = crop_random(
                            ct, mask, crop_shape=self.crop_shape
                        )
                        is_empty = np.all(mask_tile == 0)
                        if is_empty:
                            break
                        else:
                            failed_attempts += 1
                        if failed_attempts >= self.attempts_number:
                            break
                    yield ct_tile, mask_tile

                else:
                    raise RuntimeError('Tile selection error.')

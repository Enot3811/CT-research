"""Base sampler implementation to generalize some common logic."""

import random
from abc import ABC

from tf_ext import TFH5Generator


class Sampler(TFH5Generator, ABC):
    """Base abstract sampler."""

    GeneratorType = TFH5Generator.GeneratorType

    def __init__(self, n: int, shuffle: bool = False):
        """
        Base sampler constructor.

        Parameters
        ----------
        n : int
            Count of samples to generate from each raw sample.
        shuffle : bool, optional
            Shuffle samples range before sampling start.
        """
        self.n = n
        self.shuffle = shuffle

    def _get_sampler_range(self, st: int, end: int):
        """Get range for sampler with/without shuffling."""
        ids_range = list(range(st, end))
        if self.shuffle:
            random.shuffle(ids_range)
        return ids_range

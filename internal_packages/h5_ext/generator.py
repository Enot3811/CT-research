"""General definition of the generator from h5-dataset in specified indices."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, TypeVar


class H5Generator(ABC):
    """
    Data generator from some h5 dataset in specified data borders.

    The key-difference from typical generator is that this class is a
    generator creator functor: its instance is created from some static
    parameters and each call of this function creates new generator from
    specified data.
    """

    T = TypeVar('T')
    GeneratorType = Generator[T, None, None]

    @abstractmethod
    def __call__(self, h5_path: Path, start: int, end: int) -> GeneratorType:
        """
        Create Python-generator object that yields data from h5-dataset.

        Each call creates new generator to read data with specified logic
        from passed h5-dataset with passed interval (end not included).

        Parameters
        ----------
        h5_path : str
            The path to hdf5 dataset.
        start : int
            Start reading index.
        end : int
            End reading index.
        """
        pass

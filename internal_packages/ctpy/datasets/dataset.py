"""
Base class for each DS to generate data from it.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, List, Generator, TypeVar

from ..samples import CTSample

Split = Literal['train', 'test', 'val']


class Dataset(ABC):
    """
    CT-Dataset wrapper.

    This class allows to get all cases names and create generator of cases.
    Data of each case is defined by the method, each DS has ``CTSample``
    generator at least. Other generators can be implemented in derived classes.

    It's required to have split files for each derived dataset.
    Splits are defined by split-files: ``train.txt``, ``val.txt`` &
    ``test.txt``.
    """

    T = TypeVar('T')
    GeneratorType = Generator[T, None, None]

    def __init__(self, ds_root: Path):
        if not ds_root.exists():
            raise IOError(f'Passed {ds_root=} does not exist.')
        self.ds_root = ds_root

    @abstractmethod
    def get_names(
            self, split: Optional[Split] = None
    ) -> List[str]:
        """
        Get list of dataset sample names depending of the requested subset.

        Return all sample names if requested subset is ``None``.

        Parameters
        ----------
        split : Split, optional
            Requested split to get cases from the dataset.

        Returns
        -------
        List[str]:
            List of case's sample names of corresponding DS. List will contain
            all sample names that correspond to requested split or
            whole sample names list if subset wasn't passed.
        """
        pass

    @abstractmethod
    def get_ctsample_gen(
            self, split: Optional[Split] = None
    ) -> GeneratorType[CTSample]:
        """
        Create generator of dataset that yields `CTSample` from it.

        If subset name is specified generator yields `CTSample` only from
        this subset.
        If requested subset is ``None`` generate `CTSample` for all dataset.

        Parameters
        ----------
        split : Optional[Split], optional
            Requested split to yield ``CTSample``-s from DS.
            If not specified generator will yield ``CTSample``-s
            from all dataset.

        Returns
        -------
        DSGenerator[CTSample]:
            Python-generator that yields `CTSample` objects from DS.
        """
        pass

    @property
    def train_names(self) -> List[str]:
        """Get names of the train cases of the dataset."""
        return self.get_names('train')

    @property
    def test_names(self) -> List[str]:
        """Get names of the test cases of the dataset."""
        return self.get_names('test')

    @property
    def val_names(self) -> List[str]:
        """Get names of the val cases of the dataset."""
        return self.get_names('val')

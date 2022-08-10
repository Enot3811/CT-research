"""
Module with KneeKL X-ray dataset wrapper implementation.

This view allows to get necessary samples from DS: `KneeSample`, names, e.t.c
"""


from pathlib import Path
from typing import List, Optional, Dict, Generator

import imageio
import numpy as np
from loguru import logger

from .dataset import Split
from ..samples.knee_sample import KneeSample


class KneeKL:
    """
    X-ray dataset wrapper for KneeKL.

    This class allows to get all cases names and create generator of cases.
    """

    def __init__(self, ds_root: Path):
        """
        Initialize KneeKL dataset object with specified parameters.

        Parameters
        ----------
        ds_root : Path
            Dataset root directory.

        Raises
        ------
        IOError
            Passed root dir does not exist.
        """
        if not ds_root.exists():
            raise IOError(f'Passed {ds_root=} does not exist.')
        self.ds_root = ds_root
        self.splits: Dict[str, List[str]] = {}
        self.sample_paths: Dict[str, Path] = {}
        for split_name in ('train', 'val', 'test'):
            split_paths = list(self.ds_root.glob(f'{split_name}/*/*.png'))
            split_names = ['_'.join(path.parts[-2:])
                           for path in split_paths]
            self.splits[split_name] = split_names
            self.sample_paths.update(
                {k: v for k, v in zip(split_names, split_paths)}
            )

    def get_names(self, split: Optional[Split] = None) -> List[str]:
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
            List of case's sample names of knee X-ray DS. List will contain
            all sample names that correspond to requested split or
            whole sample names list if subset wasn't passed.
        """
        if split is None:
            samples = (list(self.splits['train']) +
                       list(self.splits['val']) +
                       list(self.splits['test']))
        else:
            samples = list(self.splits[f'{split}'])
        return samples
        
    def get_knee_sample_gen(
        self, split: Optional[Split] = None
    ) -> Generator[KneeSample, None, None]:
        """
        Create generator of Knee-KL dataset that yields `KneeSample` from it.

        If subset name is specified generator yields `KneeSample` only from
        this subset.
        If requested subset is ``None`` generate `KneeSample` for all dataset.

        Parameters
        ----------
        split : Optional[Split], optional
            Requested split to yield ``KneeSample``-s from DS.
            If not specified generator will yield ``KneeSample``-s
            from all dataset.

        Returns
        -------
        Generator[KneeSample, None, None]:
            Python-generator that yields `KneeSample` objects from DS.
        """
        xray_names = self.get_names(split)
        for xray_name in xray_names:
            xray_path = Path(self.sample_paths[xray_name])
            try:
                xray = imageio.v2.imread(xray_path)
                if xray.dtype != np.uint8:
                    raise ValueError(f'Unexpected {xray.dtype=}')
                if len(xray.shape) != 2:
                    raise ValueError(f'Unexpected {xray.shape=}')
                name = xray_name
                label = int(xray_name.split('_')[0])
                knee_sample = KneeSample(xray, label, name)
            except np.core._exceptions._ArrayMemoryError as e:  # type: ignore
                logger.warning(f'Error occurred during reading {xray_path}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            except Exception as e:
                logger.warning(f'Error occurred during reading {xray_path}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            yield knee_sample

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

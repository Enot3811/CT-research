"""Factory class for creating DSGenerators with different types."""

from pathlib import Path
from typing import Dict, Type, cast
from functools import partial

from .ctspine1k import CTSpine1K
from .mosmed import MosmedDS
from .msd import MSD
from .dataset import Dataset


class DatasetFactory:
    """Factory for `Dataset` objects."""

    __DS_MAP: Dict[str, Type[Dataset]] = {
        'ctspine_1k': CTSpine1K,
        'mosmed': MosmedDS,
        'msd_brats_2016_2017': cast(Type[Dataset], partial(MSD, not_3d=True)),
        'msd_prostate': cast(Type[Dataset], partial(MSD, not_3d=True)),
        'msd_colon': MSD,
        'msd_hepatic_vessels': MSD,
        'msd_hippocampus': MSD,
        'msd_left_atrium': MSD,
        'msd_lits_2017': MSD,
        'msd_lung': MSD,
        'msd_pancreas': MSD,
        'msd_spleen': MSD
    }
    """Mapping from ds-name to the ds-class."""

    @staticmethod
    def create_dataset(ds_root: Path) -> Dataset:
        """
        Create `Dataset` object of specified type.

        Parameters
        ----------
        ds_root : Path
            Path to dir that contain dataset.
            Name of the root is a key identifier for created object.

        Returns
        -------
        Dataset
            Created `Dataset` object that actually is a view of data.

        Raises
        ------
        KeyError
            Specified Root path has not corresponding generator.
        """
        if ds_root.name in DatasetFactory.__DS_MAP:
            return DatasetFactory.__DS_MAP[ds_root.name](ds_root)
        else:
            raise KeyError(f'Root path "{ds_root}" '
                           f'has no corresponding dataset-wrapper.')

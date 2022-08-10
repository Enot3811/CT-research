"""
Module with CTGenerator for all datasets included to MSD.

MSD returns path relative to root directory.
Its CTSample generator reads NIfTI.gz files from any split of MSD.
"""

import copy
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ..samples import CTSample
from ..serializers import CTSampleSerialization
from .dataset import Dataset, Split
from ..nifti.reader import NIFTIReader


class MSD(Dataset):
    """
    CT-Dataset wrapper for MSD datasets.

    This class allows to get all cases names and create generator of cases.
    """
    
    def __init__(self, ds_root: Path, not_3d: bool = False):
        super().__init__(ds_root)
        self.not_3d = not_3d

    def get_names(self, split: Optional[Split] = None) -> List[str]:
        """
        Get list of dataset relative paths depending of the requested split.

        Return all relative paths if requested split is ``None``.

        Parameters
        ----------
        split : split, optional
            Requested split to get cases of the dataset.

        Returns
        -------
        List[str]:
            List of case's relative paths of MSD DS. List will contain
            all relative paths that correspond to requested split or
            whole relative paths list if split wasn't passed.
        """
        if split is not None:
            path = Path(self.ds_root, f'{split}.txt')
            with open(path, 'r') as split_file:
                samples = split_file.read().split('\n')
            samples.remove('')  # Empty str is got after read and split file
        else:
            samples = [str(Path(abs_path.parts[-2], abs_path.parts[-1]))
                       for abs_path in
                       self.ds_root.glob('images*/[!.]*.nii.gz')]
        return samples

    def get_ctsample_gen(
        self, split: Optional[Split] = None
    ) -> Dataset.GeneratorType[CTSample]:
        """
        Create generator fro MSD datasets that yields ``CTSample`` from them.

        If split name is specified generator yields ``CTSample`` only from
        this split.
        If requested split is ``None`` generate CTSample for all dataset.

        Parameters
        ----------
        split : Optional[split], optional
            Requested data-split to yields ``CTSample``-s from.
            If not specified generator will yield ``CTSample``-s
            from all dataset.
            By default None.

        Returns
        -------
        DSGenerator.GeneratorType[CTSample]:
            Python-generator that yields CTSample objects from DS.
        """
        samples = self.get_names(split)
        ct_reader = NIFTIReader()
        serializer = CTSampleSerialization()
        for sample in samples:
            try:
                ct_path = Path(self.ds_root, sample)
                volume = ct_reader(ct_path, self.not_3d)
                if self.not_3d:
                    raw_4d = volume.raw_data
                    ct_sample = None
                    for i in range(raw_4d.shape[-1]):
                        # Get only one CT from 4D
                        raw_3d = copy.copy(raw_4d[:, :, :, i])
                        # Copy volume with all its metainfo
                        sub_volume = copy.deepcopy(volume)
                        # Replace 4D by one 3D CT
                        sub_volume.raw_data = raw_3d
                        ct_sample = serializer.from_volume(sub_volume, sample)
                    if ct_sample is None:
                        raise IOError(f'No 3D volumes in {sample}.')
                else:
                    ct_sample = serializer.from_volume(volume, sample)
            except Exception as e:
                logger.warning(f'Error occurred during reading {sample}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            yield ct_sample

"""
Module with CTSpine_1k dataset wrapper implementation.

This view allows to get necessary samples from DS: `CTSample` or `SpineSample`.
"""


from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from loguru import logger

from .dataset import Dataset, Split
from ..nifti.reader import NIFTIReader
from ..samples import CTSample, SpineSample
from ..serializers import SpineSampleSerialization, CTSampleSerialization


class CTSpine1K(Dataset):
    """
    CT-Dataset wrapper for CTSpine_1k.

    This class allows to get all cases names and create generator of cases.
    """

    def __init__(self, ds_root):
        super().__init__(ds_root=ds_root)

        ct_dir = self.ds_root.joinpath('volumes')
        masks_dir = self.ds_root.joinpath('masks')
        ct_files: List[Path] = list(ct_dir.glob('*.nii.gz'))
        masks_files: List[Path] = list(masks_dir.glob('*'))

        ct_paths: Dict[str, Path] = {
            c.name.replace('_ct', '').replace('.nii.gz', ''): c
            for c in ct_files
        }

        # collect masks for ct-paths only because their suffix is bullshit
        mask_paths: Dict[str, Path] = {}
        for sample_name in ct_paths.keys():
            appropriate_ids = []
            appropriate_names = []
            for i, m in enumerate(masks_files):
                if m.name.startswith(sample_name):
                    if m.is_dir() or ''.join(m.suffixes[-2:]) == '.nii.gz':
                        left_part = m.name.replace(sample_name, '')
                        if left_part[0] in ['_', '.']:
                            appropriate_ids.append(i)
                            appropriate_names.append(m.name)
            if len(appropriate_ids) > 1:
                logger.warning(
                    f'More than one mask was found for {sample_name}'
                )
                continue
            elif len(appropriate_ids) == 0:
                logger.warning(
                    f'No mask was found for {sample_name}: {appropriate_names}'
                )
                continue
            else:
                mask_paths[sample_name] = masks_files[appropriate_ids[0]]
                masks_files.pop(appropriate_ids[0])

        self.sample_paths: Dict[str, Dict[str, Path]] = {}
        for name in ct_paths.keys():
            if name not in mask_paths.keys():
                logger.warning(f'No mask path for {name}.')
                continue
            s_path: Dict[str, Path] = {
                'ct': ct_paths[name],
                'mask': mask_paths[name]
            }
            self.sample_paths[name] = s_path

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
            List of case's sample names of CTSpine_1k DS. List will contain
            all sample names that correspond to requested split or
            whole sample names list if subset wasn't passed.
        """
        if split is not None:
            path = Path(self.ds_root, f'{split}.txt')

            if not path.is_file():
                raise FileNotFoundError(f'Split file "{path}" does not exist.')

            with open(path, 'r') as split_file:
                samples = split_file.read().split('\n')
            samples.remove('')  # Empty str is got after read and split file
        else:
            samples = list(self.sample_paths.keys())
        return samples

    def get_ctsample_gen(
        self, split: Optional[Split] = None
    ) -> Dataset.GeneratorType[CTSample]:
        """
        Create CTSpine_1K dataset generator that yields `CTSample` from it.

        If subset name is specified generator yields `CTSample` only from
        this subset.
        If requested subset is ``None`` generate `CTSample` for a
        whole dataset.

        Parameters
        ----------
        split : Optional[Split], optional
            Requested split to yield ``CTSample``-s from DS.

        Returns
        -------
        Dataset.GeneratorType[CTSample]:
            Python-generator that yields `CTSample` objects from DS.
        """
        samples = self.get_names(split)
        ct_reader = NIFTIReader()
        serializer = CTSampleSerialization()
        for sample in samples:
            ct_path = self.sample_paths[sample]['ct']
            try:
                ct_volume = ct_reader(ct_path)
                ct_sample = serializer.from_volume(ct_volume, sample)
            except np.core._exceptions._ArrayMemoryError as e:  # type: ignore
                logger.warning(f'Error occurred during reading {ct_path}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            except Exception as e:
                logger.warning(f'Error occurred during reading {ct_path}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            yield ct_sample

    def get_spinesample_gen(
        self, split: Optional[Split] = None
    ) -> Dataset.GeneratorType[SpineSample]:
        """
        Create CTSpine_1K dataset generator that yields `SpineSample` from it.

        If subset name is specified generator yields `SpineSample` only from
        this subset.
        If requested subset is ``None`` generate `SpineSample` for all dataset.

        Parameters
        ----------
        split : Optional[Split], optional
            Requested split to yield ``SpineSample``-s from DS.

        Returns
        -------
        Dataset.GeneratorType[SpineSample]:
            Python-generator that yields `SpineSample` objects from DS.
        """
        samples = self.get_names(split)
        ct_reader = NIFTIReader()
        serializer = SpineSampleSerialization()
        for sample in samples:
            ct_path = self.sample_paths[sample]['ct']
            mask_path = self.sample_paths[sample]['mask']
            try:
                ct_volume = ct_reader(ct_path)
                mask_volume = ct_reader(mask_path)
                spine_sample = serializer.from_volume(
                    ct_volume, mask_volume, sample
                )
            except np.core._exceptions._ArrayMemoryError as e:  # type: ignore
                logger.warning(f'Error occurred during reading {ct_path}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            except Exception as e:
                logger.warning(f'Error occurred during reading {ct_path}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            yield spine_sample

"""
Module with CTGenerator for Mosmed dataset.

`MosmedDS` returns sample names.
Its CTSample-generator extracts dicom files from mosmed tars to temp folder
and then reads them.
"""


import tarfile
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger

from ..samples import CTSample
from ..serializers import CTSampleSerialization
from .dataset import Split, Dataset
from ..dicom.reader import DicomReader


class MosmedDS(Dataset):
    """
    CT-Dataset wrapper for Mosmed-data.

    This class allows to get all cases names and create generator of cases.
    """

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
            List of case's sample names of MosMed DS. List will contain
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
            sample_paths: List[Path] = list(
                self.ds_root.glob('Week*/studies/*/*/*.tar')
            )
            
            # Create sample names from paths
            samples = [f'{s_path.parts[-5]};'
                       f'{s_path.parts[-3]};'
                       f'{s_path.parts[-2]}'
                       for s_path in sample_paths]

        return samples

    def get_ctsample_gen(
            self, split: Optional[Split] = None
    ) -> Dataset.GeneratorType[CTSample]:
        """
        Create generator of MosMed dataset that yields `CTSample` from it.

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
        samples = self.get_names(split)
        ct_reader = DicomReader()
        serializer = CTSampleSerialization()
        for sample in samples:
            path_pieces = sample.split(';')
            tar_path = Path(self.ds_root, path_pieces[0], 'studies',
                            path_pieces[1], path_pieces[2], 'study_in.tar')

            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    with tarfile.open(tar_path, 'r') as tar:
                        tar.extractall(tmpdir, members=tar.getmembers())
                    volumes = ct_reader(Path(tmpdir))
                    if not volumes:
                        raise IOError(f'No valid volumes at: {tar_path}')
                    else:
                        if len(volumes) > 1:
                            logger.warning(
                                f'More than one volume is stored at: '
                                f'{tar_path}. Selected first only.'
                            )
                        volume = volumes[0]
                    ct_sample = serializer.from_volume(volume, sample)
            except np.core._exceptions._ArrayMemoryError as e:  # type: ignore  # noqa
                logger.warning(f'Error occurred during reading {sample}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            except Exception as e:
                logger.warning(f'Error occurred during reading {sample}:\n'
                               f'{e}\n'
                               f'Skip case.')
                continue
            yield ct_sample

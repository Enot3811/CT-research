"""
HDF5 Split manager: divide h5 datasets into blocks with assigned generators.

The main goal of this manager is to control sampling from many different h5
datasets and allow to split single dataset into parts.

This functionality allows to parallel processing as along different files as
inside one file (typically huge). Important to note that h5-dataset provides
parallel reading.

Typical scenario of usage:
1.1 Create huge h5 file with regular data
1.2 Create small h5 with some other data
1.n <...>
2. Create configuration instance for each dataset:
    * how many splits to do
    * what generator to use (implement if need)
3. Create manager from this configurations list
4. Then `manager[i]` - return i-th generator of the data. Total amount of
generator are equal to the splits number sum from all datasets. If first DS
was split into 5 parts and second DS was split into 3 parts, then
`manager[3]` - first DS, 4th split block and
`manager[7]` - second DS, 3rd block.

HDF-5 datasets that are supported: any h5-datasets that have 'size' attribute.
Splitting indices will be generated basing on this.
"""

from math import ceil
from pathlib import Path
from typing import Tuple, List

import h5py
from loguru import logger

from .generator import H5Generator


class H5SplitManager:
    """
    Split manager for h5-datasets.

    It allows to divide input h5-file into blocks with specified
    data generator. Manger presents sequence public API. It's i-th element is
    a generator of the data from specified h5 file in specified borders.

    Manager allows to use any datasets which has ``size`` attribute. Splitting
    blocks will be created basing on this. It's assumed that all keys, all
    read ``h5py.Dataset`` objects will have amount of elements equal to this
    ``size`` attribute. It's also recommended to use the same chunk size.

    Important: each i-th generator is new generator object.
    """

    class DSConfig:
        """Single h5 dataset meta-configuration necessary for split."""

        path: Path
        """Path to the h5 dataset."""

        split_n: int
        """Number of blocks to split data into."""

        generator: H5Generator
        """Data generator from h5-dataset that need to use for this DS."""

        def __init__(self, path: Path, split_n: int, generator: H5Generator):
            """Init structure with meta-configuration."""
            self.path = path
            self.split_n = split_n
            self.generator = generator

    class SingleReader:
        """
        Reader class for single dataset.
        
        Reader is needed for referring to a certain part of data
        in hdf5 files (now this part is separate dataset)
        and returning generator that will load this data.
        """

        def __init__(
            self, h5_path: Path, start: int, end: int, generator: H5Generator
        ):
            """
            Initialize SingleReader object
            
            During the initialization object saves given parameters in self.

            Parameters
            ----------
            h5_path : Path
                The path to hdf5 file with data.
            start : int
                The index of sample in file from which loading will start.
            end : int
                The index of sample in file before which loading will end.
            generator : Callable
                The generator functor for loading data from file.
            """
            self.h5_path = h5_path
            self.start = start
            self.end = end
            self.generator = generator

        def create_generator(self) -> H5Generator.GeneratorType:
            """
            Create generator for reading dataset.
            
            Calls the saved generator functor with put
            other saved parameters into him as a return.

            Returns
            -------
            H5Generator.GeneratorType:
                New Python-generator for loading data in passed interval.
            """
            return self.generator(self.h5_path, self.start, self.end)

    def __init__(self, dataset_configs: List[DSConfig]):
        """
        Create split-manager instance from a list of datasets meta-configs.

        During the initialization each file from given config is split
        into specify number of splits and then each resulted part
        of data from file is put to `SingleReader` as a path to ``hdf5`` file,
        start/end indexes and a generator for loading.

        At the end of the initialization, the split-manager object will store
        a list of `SingleReader`, each of which will refer to a certain part
        of the data from the specified file.

        Parameters
        ----------
        dataset_configs : List[DSConfig]
            List with datasets to manage. Each item is a single h5-dataset
            meta-configuration. Each dataset can be split into several blocks.
            Each DS has its own generator. Final count of data readers will
            be equal to split number sum of all datasets.
        """
        self._readers = []

        for ds_config in dataset_configs:
            with h5py.File(ds_config.path, 'r') as f:
                dataset_len = f.attrs['size']
                if ds_config.split_n > dataset_len:
                    raise ValueError(
                        f'It is not possible to split ds with '
                        f'length={dataset_len} on splits={ds_config.split_n}.'
                    )
                chunks = set()
                for k in f.keys():
                    chunks.add(f[k].chunks[0])
                split_size = dataset_len // ds_config.split_n
                logger.debug(f'Expected one-split size: {split_size}.')
                chunks_list = list(chunks)
                if len(chunks_list) > 1:
                    logger.warning(f'Found more than one chunk size: '
                                   f'{chunks}.')
                for chunk in chunks:
                    is_many_chunks_in_ds = dataset_len > chunk
                    is_split_has_many_chunks = split_size > chunk
                    is_split_multiple_of_chunks = split_size % chunk == 0
                    if is_many_chunks_in_ds:
                        if not is_split_has_many_chunks:
                            logger.warning(
                                f'It is recommended to use split number to '
                                f'have many chunks in one split: '
                                f'{chunk=} {split_size=}. The reason for that '
                                f'is because when any element in a chunk is '
                                f'accessed, the entire chunk is read '
                                f'from disk.'
                            )
                        if not is_split_multiple_of_chunks:
                            logger.warning(
                                f'It is recommended to use split that '
                                f'will contain amount of cases which is '
                                f'multiple of chunk size: '
                                f'{chunk=} {split_size=}. The reason for that '
                                f'is because when any element in a chunk is '
                                f'accessed, the entire chunk is read '
                                f'from disk.'
                            )
                    else:
                        logger.warning(
                            'Dataset has chunk size more than count of '
                            'samples.'
                        )

            # Get list of start/end indexes
            boundaries = H5SplitManager._split_dataset(
                dataset_len, ds_config.split_n)

            # Create reader for each split
            for (start, end) in boundaries:
                self._readers.append(
                    H5SplitManager.SingleReader(ds_config.path,
                                                start,
                                                end,
                                                ds_config.generator))
    
    def __getitem__(self, i: int) -> H5Generator.GeneratorType:
        """
        Get generator for i-th dataset block.

        Each new call creates new generator object.

        Parameters
        ----------
        i : int
            Index of requested dataset.

        Returns
        -------
        H5Generator.GeneratorType:
            Python-generator object that yields data from i-th split.
        """
        return self._readers[i].create_generator()

    def __len__(self) -> int:
        """
        Get number of dataset blocks (number of all splits from all ds-files).

        Returns
        -------
        int
            Number of dataset splits.
        """
        return len(self._readers)

    @staticmethod
    def _split_dataset(dataset_length: int,
                       split_number: int) -> List[Tuple[int, int]]:
        """
        Split dataset into several intervals ``[start, end)``.

        Parameters
        ----------
        dataset_length : int
            Length of dataset that will split.
        split_number : int
            Number of dataset splits needed.

        Returns
        -------
        List[Tuple[int, int]]
            List with tuples contained ``[start, end)`` indexes.
        """
        split_step = ceil(dataset_length / split_number)
        if split_step < 10:
            logger.warning(
                'It is recommended to use split which gives each worker '
                'at least 10 cases to read/process. It will lead to '
                'get trade off between parallelism and I/O prefetching '
                f'benefits. Current {split_step=} {split_number=}.'
            )
        start_positions = list(range(0, dataset_length, split_step))
        end_positions = [min(dataset_length, x + split_step)
                         for x in start_positions]

        return list(zip(start_positions, end_positions))

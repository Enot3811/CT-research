"""
Class-wrapper that dispatches all serialization functionality for `KneeSample`.
"""


from typing import Union, Dict

import h5py
import numpy as np
from cvl.hdf5 import store_data, DatasetDescription

from h5_ext import H5DSCreationManager

from ..samples.knee_sample import KneeSample


class KneeSampleSerialization:
    """
    Serialization manager for `KneeSample` instance.

    Can read & write corresponding objects from h5.
    """

    def __init__(self,
                 chunksize: int = None,
                 compression: str = None, compression_opts: int = None):
        """
        Create Serializer.

        Parameters
        ----------
        chunksize: int, optional
            HDF5 dataset chunk size (default is unset).
        compression: str, optional
            HDF5 dataset compression mode.
            Default use ``h5py`` default behavior.
        compression_opts: int, optional
            Sets the compression level and may be an integer from 0 to 9,
            default is 4. Ignores if `compression` is ``None``.
        """
        self.__common_params = {
            'chunksize': chunksize,
            'compression': compression,
            'compression_opts': compression_opts
        }
        self.__ds_descriptions: Dict[str, DatasetDescription] = {
            'xray': DatasetDescription(
                shape=(), dtype=h5py.vlen_dtype('int16'),
                **self.__common_params
            ),
            'label': DatasetDescription(
                shape=(), dtype='int16',
                **self.__common_params
            ),
            'shape': DatasetDescription(
                shape=(2,), dtype='uint16',
                **self.__common_params
            ),
            'name': DatasetDescription(
                shape=(), dtype=h5py.string_dtype(),
                **self.__common_params
            )
        }

    def get_h5_structure(self) -> Dict[str, DatasetDescription]:
        """Get expected h5-dataset structure (copy)."""
        return self.__ds_descriptions.copy()

    def __check_dataset(self, ds: h5py.File):
        """Check that passed h5.File correspond to the descriptions."""
        err_prefix = 'Passed h5-dataset to accumulate into '
        for ds_k, ds_descr in self.__ds_descriptions.items():
            if ds_k not in ds:
                raise KeyError(
                    f'{err_prefix} has no required {ds_k=}'
                )
            stored_shape = ds[ds_k].shape[1:]
            if ds_descr.shape != stored_shape:
                raise ValueError(
                    f'{err_prefix} has wrong shape for {ds_k}: '
                    f'has - {stored_shape}, expected - {ds_descr.shape}'
                )
            stored_dtype = ds[ds_k].dtype
            if ds_descr.dtype != stored_dtype:
                raise ValueError(
                    f'{err_prefix} has wrong dtype for {ds_k}: '
                    f'has - {stored_dtype}, expected - {ds_descr.dtype}'
                )

    def from_h5(self, ds: h5py.File, i: int) -> KneeSample:
        """
        Create sample from data in h5 file.

        It's expected strict structure of the ``h5`` file. All necessary fields
        must have `h5.Dataset` object inside passed ``h5``,
        X-ray data is expected to be flatten.

        Parameters
        ----------
        ds : h5py.File
            Description of the file.
        i : int
            Index of the case to read.

        Returns
        -------
        KneeSample:
            Created sample with all necessary data.
        """
        self.__check_dataset(ds)

        if 'shape' in ds:
            shape = tuple(ds['shape'][i])
            xray = np.array(ds['xray'][i]).reshape(shape)
        else:
            xray = np.empty(ds['xray'].shape[1:], ds['xray'].dtype)
            ds['xray'].read_direct(
                xray, slice(i, i + 1), slice(0, None)
            )
        
        label = ds['label'][i]
        name = ds['name'][i].decode()
        sample = KneeSample(xray, label, name)
        return sample

    def to_h5(self, knee_sample: KneeSample,
              manager: Union[h5py.File, H5DSCreationManager]):
        """
        Dump passed sample in h5-dataset with fixed structure.

        It's assumed that handling h5-file will have all necessary fields
        with the same names.

        Parameters
        ----------
        knee_sample: KneeSample
            Knee X-ray sample object to dump into h5.
        manager :
            Manager-object that handles dumping into h5-file.
            It can be h5-descriptor or h5-creation-manager from extensions.
        """
        kwargs = {
            'xray': np.reshape(knee_sample.xray, (-1,)),
            'label': knee_sample.kl_label,
            'shape': knee_sample.shape,
            'name': knee_sample.name
        }
        if isinstance(manager, h5py.File):
            self.__check_dataset(manager)
            store_data(manager, **kwargs)
        elif isinstance(manager, H5DSCreationManager):
            self.__check_dataset(manager.handle_ds)
            manager.accumulate_data(**kwargs)
        else:
            raise RuntimeError(f'Unknown manager type for KneeSample dump '
                               f'into h5: {manager=}.')

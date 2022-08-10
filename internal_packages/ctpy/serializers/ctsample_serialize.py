"""
Class-wrapper that dispatches all serialization functionality for `CTSample`.
"""


from typing import Union, Dict

import h5py
import numpy as np
from cvl.hdf5 import store_data, DatasetDescription

from h5_ext import H5DSCreationManager

from ..samples import CTSample
from ..volume import Volume


class CTSampleSerialization:
    """
    Serialization manager for `CTSample` instance.

    Can read & write corresponding objects from h5 or volume-dtypes.
    It requires from data to be in ``RAS`` space.
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
            'ct': DatasetDescription(
                shape=(), dtype=h5py.vlen_dtype('int16'),
                **self.__common_params
            ),
            'shape': DatasetDescription(
                shape=(3,), dtype='uint16',
                **self.__common_params
            ),
            'voxel_size': DatasetDescription(
                shape=(3,), dtype='float32',
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

    def from_volume(self, volume: Volume, name: str):
        """
        Create `CTSample` from `Volume`. Auto-warp into ``RAS``.

        Parameters
        ----------
        volume : Volume
            Arbitrary volume.
        name : str
            Name of passed volume

        Returns
        -------
        CTSample:
            Created sample in ``RAS`` space with only necessary data.
        """
        ras_vol = volume.get_warped(space='RAS')
        ct = ras_vol.pixel_data
        voxel_size = ras_vol.voxel_size
        sample = CTSample(ct, voxel_size, name)
        return sample

    def from_h5(self, ds: h5py.File, i: int):
        """
        Create sample from data in h5 file.

        It's expected strict structure of the ``h5`` file. All necessary fields
        must have `h5.Dataset` object inside passed ``h5``, ct-data is expected
        to be flatten.

        Parameters
        ----------
        ds : h5py.File
            Description of the file.
        i : int
            Index of the case to read.

        Returns
        -------
        CTSample:
            Created sample with all necessary data.
        """
        self.__check_dataset(ds)

        if ds.attrs['space'] != 'RAS':
            raise NotImplementedError(f'Creation of CTSample from unsupported '
                                      f'space = {ds.attrs["space"]}')

        if 'shape' in ds:
            shape = tuple(ds['shape'][i])
            ct = np.array(ds['ct'][i]).reshape(shape)
        else:
            ct = np.empty(ds['ct'].shape[1:], ds['ct'].dtype)
            ds['ct'].read_direct(
                ct, slice(i, i + 1), slice(0, None)
            )
        voxel_size_arr = ds['voxel_size'][i]
        voxel_size: Dict[str, float] = {
            'i': float(voxel_size_arr[0]),
            'j': float(voxel_size_arr[1]),
            'k': float(voxel_size_arr[2])
        }
        name = ds['name'][i].decode()
        sample = CTSample(ct, voxel_size, name)
        return sample

    def to_h5(self, ct_sample: CTSample,
              manager: Union[h5py.File, H5DSCreationManager]):
        """
        Dump passed sample in h5-dataset with fixed structure.

        It's assumed that handling h5-file will have all necessary fields
        with the same names (look `CTSample` API).

        Parameters
        ----------
        ct_sample: CTSample
            CT sample object to dump into h5.
        manager :
            Manager-object that handles dumping into h5-file.
            It can be h5-descriptor or h5-creation-manager from extensions.
        """
        kwargs = {
            'ct': np.reshape(ct_sample.ct, (-1,)),
            'voxel_size': ct_sample.voxel_size_arr,
            'shape': ct_sample.shape,
            'name': ct_sample.name
        }
        if isinstance(manager, h5py.File):
            self.__check_dataset(manager)
            store_data(manager, **kwargs)
        elif isinstance(manager, H5DSCreationManager):
            self.__check_dataset(manager.handle_ds)
            manager.accumulate_data(**kwargs)
        else:
            raise RuntimeError(f'Unknown manager type for CTSample dump '
                               f'into h5: {manager=}.')

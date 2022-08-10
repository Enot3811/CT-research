"""Script that converts spine CT dataset to the only one fixed voxel size."""


import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Tuple

import h5py
from tqdm import tqdm

from ctpy.serializers import SpineSampleSerialization
from ctpy.samples import SpineSample
from h5_ext import H5DSCreationManager
from pynative_ext.argparse.feeders import feed_log
from pynative_ext.argparse.types import non_negative_float, natural_int
from pynative_ext.loguru import init_loguru
from pynative_ext.os import get_now, get_repo_info
from pynative_ext.os import get_uuid


def parse_args():
    """Create & parse CLI-args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--ds', type=Path,
        help='Path to spine CT h5 dataset.',
        required=True
    )
    parser.add_argument(
        '--out', type=Path,
        help='Path to save converted h5 dataset.',
        required=True
    )
    parser.add_argument(
        '--voxel-size', type=non_negative_float,
        nargs=3,
        help='New voxel size for dataset.',
        required=True
    )
    parser.add_argument(
        '--num-workers', type=natural_int,
        help='Number of parallel workers for processing.',
        default=1
    )
    parser.add_argument(
        '--buffer-size', type=natural_int,
        help='How many cases to process before dump.',
        default=10
    )

    feed_log(parser)

    args = parser.parse_args()

    init_loguru(args)

    if not args.ds.is_file():
        raise FileNotFoundError(f'Dataset file "{args.ds}" does not exist.')

    return args


def read_and_resize_case(
        idx: int, dataset: Path,
        new_voxel_size: Tuple[float, float, float]
) -> SpineSample:
    """
    Function for read data from h5 dataset, resize and return result.

    Parameters
    ----------
    idx: int
        Idx of process case in source h5 dataset.
    dataset: Path
        Path to the h5 dataset to read data from.
    new_voxel_size: VoxelSize
        Wished voxel size.

    Returns
    -------
    SpineSample:
        Resized sample from ``ds[idx]`` position with new voxel-size.
    """
    ds = h5py.File(dataset, 'r')
    serialize_manager = SpineSampleSerialization()
    sample = serialize_manager.from_h5(ds, idx)
    resized_sample = sample.get_resized_with_mask(new_voxel_size)
    return resized_sample


def main():
    """Application entry point."""
    args = parse_args()
    new_voxel_size = tuple(args.voxel_size)

    serialize_manager = SpineSampleSerialization()
    ds_descriptions = serialize_manager.get_h5_structure()
    ds_attrs = {
        'date': get_now(),
        'commit_info': get_repo_info(),
        'uuid': get_uuid(),
        'root': args.out.name,
        'space': 'RAS'
    }

    with H5DSCreationManager(
        ds=args.out,
        attrs=ds_attrs,
        descriptions=ds_descriptions
    ) as h5_manager, h5py.File(args.ds, 'r') as h5_ds:
        idxes = list(range(h5_ds.attrs['size']))

        # open ds & prepare fn for each worker
        worker_fn = partial(read_and_resize_case,
                            dataset=args.ds,
                            new_voxel_size=new_voxel_size)

        progress_bar = tqdm(idxes)
        progress_bar.set_description(
            f'Generating {args.buffer_size=} {args.num_workers=}...'
        )
        with mp.Pool(processes=args.num_workers) as pool:
            for i in range(0, len(idxes), args.buffer_size):
                st = i
                end = min(len(idxes), i + args.buffer_size)
                samples = pool.map(worker_fn, idxes[st:end])
                progress_bar.update(end - st)
                for sample in samples:
                    serialize_manager.to_h5(sample, h5_manager)


if __name__ == '__main__':
    main()

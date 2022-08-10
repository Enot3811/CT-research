"""
Debug script that create ``KneeSampler`` object for specified
h5 file and and iterate over them visualizing read samples.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from ctpy.h5_samplers import KneeSampler
from ndarray_ext.visualization import get_images_grid
from pynative_ext.argparse.feeders import feed_log
from pynative_ext.loguru import init_loguru


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ds',
        help='Path to h5 dataset file.',
        required=True,
        type=Path)

    feed_log(parser, default_level='DEBUG')

    args = parser.parse_args()

    init_loguru(args)

    if not args.ds.is_file():
        raise FileNotFoundError(f'H5 file "{args.ds}" does not exist.')

    return args


def main():
    """Application entry point."""
    args = parse_args()
    sampler = KneeSampler(
        data_shape=(128, 128), aspect_ratio=True, shuffle=True
    )

    with h5py.File(args.ds, 'r') as f:
        if 'size' in f.attrs.keys():
            length = f.attrs['size']
        else:
            length = len(f['xray'])
    start_indx = 0
    end_indx = length

    gen = sampler(args.ds, start=start_indx, end=end_indx)
    for xray, label in gen:
        get_images_grid(
            xray,
            names=[f'{label=}, shape={xray.shape}'])
        plt.show()


if __name__ == '__main__':
    main()

"""
Debug script that create ``SegmentationSampler3D`` object for specified
h5 file and and iterate over them visualizing read samples.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ctpy.h5_samplers import SpineSGMSampler3D
from ndarray_ext.visualization import get_volumes_grid
from pynative_ext.argparse.feeders import feed_log
from pynative_ext.argparse.types import unit_interval, natural_int
from pynative_ext.loguru import init_loguru


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--h5-path',
        help='Path to h5 dataset file.',
        required=True,
        type=Path)
    parser.add_argument(
        '--empty-prob',
        help='Float between 0 and 1, probability of choosing an empty tile.',
        default=0.4,
        type=unit_interval)
    parser.add_argument(
        '--target-prob',
        help='Float between 0 and 1, probability of choosing a target tile.',
        default=0.4,
        type=unit_interval)
    parser.add_argument(
        '--random-prob',
        help='Float between 0 and 1, probability of choosing a random tile.',
        default=0.2,
        type=unit_interval)
    parser.add_argument(
        '--crop-shape',
        help='Output tile shape.',
        nargs=3,
        default=(128, 128, 128),
        type=int)
    parser.add_argument(
        '--data-per-sample',
        help='Number of selected slices per one CT.',
        default=10,
        type=natural_int)

    feed_log(parser, default_level='DEBUG')

    args = parser.parse_args()

    init_loguru(args)

    if not args.h5_path.is_file():
        raise FileNotFoundError(f'H5 file "{args.h5_path}" does not exist.')
    eps = 0.0001
    if np.abs(args.empty_prob + args.target_prob +
              args.random_prob - 1.0) > eps:
        raise ValueError(f'Sum of probs "{args.empty_prob}, {args.target_prob}'
                         f', {args.random_prob}" does not equal 1.0.')

    return args


def main():
    """Application entry point."""
    args = parse_args()
    sampler = SpineSGMSampler3D(
        crop_shape=tuple(args.crop_shape),
        n=args.data_per_sample,
        probs=(args.empty_prob, args.target_prob, args.random_prob)
    )
    with h5py.File(args.h5_path, 'r') as f:
        if 'size' in f.attrs.keys():
            length = f.attrs['size']
        else:
            length = len(f['ct'])
    start_indx = 0
    end_indx = length
    gen = sampler(args.h5_path, start=start_indx, end=end_indx)
    for ct_tile, mask_tile in gen:
        fig = get_volumes_grid(ct_tile, mask_tile, names=['CT', 'Spine'])
        fig.show()
        plt.show()


if __name__ == '__main__':
    main()

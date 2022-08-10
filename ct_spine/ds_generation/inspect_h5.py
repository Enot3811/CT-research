"""
Inspect passed HDF5 dataset with `SpineSample` data.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from ctpy.serializers import SpineSampleSerialization
from ndarray_ext.visualization import get_volumes_grid


def parse_args():
    """Create & parse CLI args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--ds', type=Path,
                        help='Path to the h5 dataset with CTSample data.')

    args = parser.parse_args()
    return args


def main():
    """Application entry point."""
    args = parse_args()
    ds = h5py.File(str(args.ds), 'r')
    serialize_manager = SpineSampleSerialization()
    for i in range(ds.attrs['size']):
        spine_sample = serialize_manager.from_h5(ds, i)
        get_volumes_grid(spine_sample.ct, spine_sample.mask,
                         title=spine_sample.name,
                         names=['Volume', 'Spine'])
        plt.show()


if __name__ == '__main__':
    main()

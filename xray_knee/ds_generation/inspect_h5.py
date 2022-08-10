"""
Inspect passed HDF5 dataset with `KneeSample` data.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

from ctpy.serializers import KneeSampleSerialization
from ndarray_ext.visualization import get_images_grid


def parse_args():
    """Create & parse CLI args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--ds', type=Path, required=True,
                        help='Path to the h5 dataset with `KneeSample` data.')

    args = parser.parse_args()
    return args


def main():
    """Application entry point."""
    args = parse_args()

    ds = h5py.File(args.ds, 'r')
    serialize_manager = KneeSampleSerialization()

    for i in range(ds.attrs['size']):
        xray_sample = serialize_manager.from_h5(ds, i)
        get_images_grid(
            xray_sample.xray,
            names=[f'label={xray_sample.kl_label}, shape={xray_sample.shape}'],
            title=str(xray_sample.name))
        plt.show()


if __name__ == '__main__':
    main()

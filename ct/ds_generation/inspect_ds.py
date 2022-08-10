"""
Debug script that create ``DSGenerator`` objects for specified dataset roots
and iterate over them visualizing read ``CTSample`` objects.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from ctpy.datasets import DatasetFactory
from ndarray_ext.visualization import get_volumes_grid
from pynative_ext.argparse.feeders import feed_log
from pynative_ext.loguru import init_loguru


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--ds-roots',
        help='Paths to dataset root directories.',
        nargs='+',
        required=True,
        type=Path)

    feed_log(parser, default_level='DEBUG')

    args = parser.parse_args()

    init_loguru(args)

    for path in args.ds_roots:
        if not path.is_dir:
            raise FileNotFoundError(f'Root directory "{path}" does not exist.')

    return args


def main():
    """Application entry point."""
    args = parse_args()
    ds_roots = args.ds_roots

    ds_factory = DatasetFactory()

    # Iterate over datasets
    for ds_root in ds_roots:
        logger.info(f'Iterating over {ds_root} dataset:')

        ds = ds_factory.create_dataset(ds_root)
        ctsample_gen = ds.get_ctsample_gen()

        # Iterate over dataset volumes
        for ct_sample in ctsample_gen:
            fig = get_volumes_grid(
                ct_sample.ct,
                title=f'{ct_sample.name}-{ct_sample.voxel_size=}',
                names=['CTSample(MAX)']
            )
            fig.show()
            plt.show()


if __name__ == '__main__':
    main()

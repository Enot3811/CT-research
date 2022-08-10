"""
Script for creating splits for CT datasets.

Script gets all sample names and split them according to specified fractions.
Default dataset splits are ``train.txt``, ``val.txt`` and ``test.txt``.

If there are some new samples that were not written in existing splits,
they are distributed according to specified fractions.
"""


import argparse
from pathlib import Path
from random import shuffle
from functools import partial
from typing import List

from pynative_ext.argparse.types import unit_interval
from pynative_ext.argparse.feeders import feed_log
from pynative_ext.loguru import init_loguru

from loguru import logger

from ctpy.datasets.dataset import Split
from ctpy.datasets.ds_factory import DatasetFactory


SPLITS: List[Split] = ['train', 'val', 'test']


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    unit_float = partial(unit_interval)

    parser.add_argument(
        '--ds-roots',
        help='Paths to dataset root directories.',
        nargs='+',
        required=True,
        type=Path)
    parser.add_argument(
        '--train-part',
        help='Float between 0 and 1, fraction of data to reserve for train.',
        default=0.7,
        type=unit_float
    )
    parser.add_argument(
        '--val-part',
        help=('Float between 0 and 1, fraction of data'
              'to reserve for validation.'),
        default=0.2,
        type=unit_float
    )
    parser.add_argument(
        '--test-part',
        help='Float between 0 and 1, fraction of data to reserve for test.',
        default=0.1,
        type=unit_float
    )

    feed_log(parser)

    args = parser.parse_args()

    init_loguru(args)

    if args.train_part + args.val_part + args.test_part > 1.0:
        raise ValueError(f'Sum of fractions [{args.train_part}, '
                         f'{args.val_part}, {args.test_part}] '
                         f'bigger than 1.0')

    for path in args.ds_roots:
        if not path.is_dir:
            raise FileNotFoundError(f'Root directory "{path}" does not exist.')

    return args


def main():
    """Application entry point."""
    args = parse_args()
    ds_roots = args.ds_roots
    
    for ds_root in ds_roots:
    
        logger.info(f'Splitting for {ds_root}')
        dataset = DatasetFactory.create_dataset(ds_root)

        samples = dataset.get_names()
        logger.info(f'All samples: {len(samples)}')

        # Get samples from splits if they are
        old_samples = set()
        for split in SPLITS:
            if ds_root.joinpath(f'{split}.txt').exists():
                old_samples = (old_samples | set(dataset.get_names(split)))
            else:
                logger.info(f'{split=} does not exist.')
        logger.info(f'Samples from existing splits: {len(old_samples)}')

        # Keep only new
        new_samples = list(set(samples) - old_samples)
        shuffle(new_samples)

        # Distribute splits to txt files
        logger.info(f'Distributing new {len(new_samples)} samples to splits.')

        samples_n = len(new_samples)
        train_samples_n = int(args.train_part * samples_n)
        val_samples_n = int(args.val_part * samples_n)

        train_samples = new_samples[:train_samples_n]
        val_samples = new_samples[
            train_samples_n:(train_samples_n + val_samples_n)]
        test_samples = new_samples[(train_samples_n + val_samples_n):]

        for split_name, split_content in zip(
                SPLITS, [train_samples, val_samples, test_samples]
        ):
            with open(Path(ds_root, f'{split_name}.txt'), 'a') as split_file:
                for sample in split_content:
                    split_file.write(f'{sample}\n')


if __name__ == '__main__':
    main()

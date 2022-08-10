"""
Generate h5-dataset from CTSpine_1k folder with `SpineSample` objects.

Script takes ds-root as input path, creates CTDataset object and generates
h5-dataset for each: train, test & val.
"""

import argparse
import shutil
from pathlib import Path
from uuid import uuid1

from loguru import logger
from tqdm import tqdm

from ctpy.datasets import CTSpine1K
from ctpy.serializers import SpineSampleSerialization
from h5_ext import H5DSCreationManager
from pynative_ext.argparse.feeders import feed_log
from pynative_ext.argparse.types import natural_int
from pynative_ext.loguru import init_loguru
from pynative_ext.os import get_now, get_repo_info


def parse_args():
    """Create & parse CLI-args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    inputs = parser.add_argument_group('Inputs')
    inputs.add_argument(
        '--ds-root', type=Path,
        help='Dataset root to generate SpineSample instances from. '
             'DS must be supported by `ctpy` package.',
        required=True
    )

    outputs = parser.add_argument_group('Outputs')
    outputs.add_argument('--out', type=Path,
                         help='Path where out h5-files will be '
                              'saved: train.h5, test.h5, val.h5. '
                              'It is expected that out-path does not exist '
                              'to prevent data from shuffling & corruption.',
                         required=True)

    # Optional generation control-args
    h5_opts = parser.add_argument_group('Generation Options')
    h5_opts.add_argument('--buffer-size', type=natural_int, default=1,
                         help='How many cases to accumulate before dump.')
    h5_opts.add_argument('--chunk-size', type=natural_int, default=1,
                         help='Chunk-size for each dataset object in '
                              'h5-ds.')
    h5_opts.add_argument('--compression', choices=['gzip', 'lzf', 'szip'],
                         help='Optional h5-chunks compression to apply. '
                              'Only h5py tools are supported. No compression '
                              'by default.')
    h5_opts.add_argument('--compression-level', type=int,
                         choices=list(range(10)),
                         help='Compression level to set. More means better '
                              'compression and slower process.')

    # Log args
    feed_log(parser)

    args = parser.parse_args()

    init_loguru(args)

    # Check out folder & recreate if need
    if args.out.exists() and len(list(args.out.glob('*'))) > 0:
        logger.warning(
            f'Out folder({args.out}) already exists and not empty.\n'
            f'Press "Enter" to overwrite or "Ctrl+C" to stop.')
        input()
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    # Save log-file in generation out folder
    logger.add(args.out.joinpath('log.txt'), level='DEBUG')
    logger.debug(f'Run arguments:\n{args}\n')

    return args


def main():
    """Application entry point."""
    args = parse_args()

    dataset = CTSpine1K(args.ds_root)

    serialize_manager = SpineSampleSerialization(
        chunksize=args.chunk_size,
        compression=args.compression,
        compression_opts=args.compression_level
    )
    ds_attrs = {
        'date': get_now(),
        'commit_info': get_repo_info(),
        'uuid': uuid1().hex,
        'root': args.ds_root.name,
        'space': 'RAS'
    }

    for split in ['train', 'test', 'val']:
        logger.info(f'Generating {split}...')
        with H5DSCreationManager(
            ds=args.out.joinpath(f'{split}.h5'),
            attrs=ds_attrs,
            descriptions=serialize_manager.get_h5_structure(),
            buffer_size=args.buffer_size
        ) as h5_manager:
            spine_sample_gen = dataset.get_spinesample_gen(split)
            progress_bar = tqdm(dataset.__getattribute__(f'{split}_names'))
            progress_bar.set_description(f'Generating {split}...')
            for spine_sample in spine_sample_gen:
                serialize_manager.to_h5(spine_sample, h5_manager)
                progress_bar.update(1)
                del spine_sample
            progress_bar.close()
        logger.info('OK.')


if __name__ == '__main__':
    main()

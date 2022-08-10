# !/usr/bin/env python

"""
Script for testing TF augmentations best place.

This script checks for each test configuration:
1) augmentations in generator performance
2) augmentations in map performance

"""

import argparse
import csv
import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import tensorflow as tf
from loguru import logger

from ct_sampler_aug import CTAugmentedSampler
from ctpy.h5_samplers import CTSampler
from tf_ext import TFH5Dataset, TFH5SplitManager
from tf_ext.misc.resources import tf_control
from tf_ext.profile import profile_dataset


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    in_gr = parser.add_argument_group('Inputs')
    in_gr.add_argument('--config',
                       help='Path to json config for test',
                       type=Path,
                       required=True)

    out_gr = parser.add_argument_group('Outputs')
    out_gr.add_argument(
        '--log-file',
        help='Path to file with logs',
        type=Path)
    out_gr.add_argument(
        '--out',
        help='Path to csv file with the results.',
        type=Path)

    args = parser.parse_args()

    if not args.config.exists():
        logger.warning(
            f'Config file: "{args.config}" does not exist.')

    for out_p in [args.log_file, args.out]:
        if out_p and out_p.exists():
            logger.warning(
                f'Out file "{out_p}" already exists & '
                f'will be overwritten.')

    return args


@tf.function
def tf_augmentation(volumes: tf.Tensor, names: tf.Tensor):
    """TF augmentation func for map."""

    def map_f(_vols):
        assert len(_vols.shape) == 4
        r = np.zeros_like(_vols)
        for i in range(_vols.shape[0]):
            r[i] = CTAugmentedSampler._do_augmentation(_vols[i])
        return r

    volume = tf.numpy_function(map_f, [volumes], tf.float32)
    return volume, names


def run_test(config_path: Path, num_iter: int = 1) -> List[Dict[str, Any]]:
    """
    Run augmentations place test.

    During this test:
    1) Create DS with augmentations in generator & run interleave
    2) Create DS without augmentations in generator & run interleave & map.

    Parameters
    ----------
    config_path : str
        Path to json test configuration file.
    num_iter : int, optional
        The number of iterations over dataset to calculate the average
        time, by default 1.

    Returns
    -------
    List[Dict[str, Any]]:
        List with results that consist of manager configuration & reading time.
    """
    results = []

    with open(config_path, 'r') as read_file:
        test_config = json.load(read_file)

    logger.debug('Iterate through tests...')
    for _, test_case in enumerate(test_config['tests']):
        ds_list = test_case['datasets']
        assert len(ds_list) == 1
        ds = ds_list[0]
        params = {k: v for k, v in ds.items()}
        title = '; '.join([f'{k} - {v}' for k, v in params.items()])

        if len(params['crop_shape']) != 3:
            raise IOError(
                f'Config {params["crop_shape"]=} is undefined.'
            )
        crop_shape = (params['crop_shape'][0], params['crop_shape'][1],
                      params['crop_shape'][2])

        # Augmentation in interleave variant
        logger.debug('Create manager with generator-variant augs...')
        ds_config = TFH5SplitManager.DSConfig(
            path=ds['h5_path'], split_n=ds['split_number'],
            generator=CTAugmentedSampler(crop_shape=crop_shape)
        )
        h5_splits = TFH5SplitManager([ds_config])
        logger.debug(f'Amount of different managers is: {len(h5_splits)}')
        dset = TFH5Dataset(h5_splits, num_workers=cpu_count(),
                           block_length=test_case['block_length'])
        dset = dset.batch(16).prefetch(1)
        avg_time = profile_dataset(dset, title=title, num_iter=num_iter)
        results.append({**params, 'aug': 'gen', 'avg_time': avg_time})

        # Augmentation in the map variant
        logger.debug('Create manager with map-variant augs...')
        ds_config.generator = CTSampler(crop_shape=crop_shape)
        manager = TFH5SplitManager([ds_config])
        logger.debug(f'Amount of different managers is: {len(manager)}')
        dset = TFH5Dataset(h5_splits, num_workers=cpu_count(),
                           block_length=test_case['block_length'])
        dset = dset.batch(test_case['block_length']).map(
            tf_augmentation, num_parallel_calls=cpu_count())
        dset = dset.unbatch().batch(16).prefetch(1)
        avg_time = profile_dataset(dset, title=title, num_iter=num_iter)
        results.append({**params, 'aug': 'map', 'avg_time': avg_time})

    logger.debug('OK.')

    return results


@tf_control
def main():
    """Application entry point."""
    args = parse_args()

    if args.log_file:
        os.makedirs(args.log_file.parent, exist_ok=True)
        logger.add(args.log_file, format='{message}', level='DEBUG')

    results = run_test(args.config)

    logger.info(f'Results are:\n{results}\n')

    if args.out:
        with open(args.out, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile,
                                    fieldnames=list(results[0].keys()),
                                    delimiter=';')
            writer.writeheader()
            for result in results:
                writer.writerow(result)


if __name__ == '__main__':
    main()

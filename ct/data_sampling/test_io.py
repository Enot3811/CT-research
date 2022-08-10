# !/usr/bin/env python

"""
Script for testing TF I/O default tools.

It allows to vary
1) Amount of datasets to load from
1) Each dataset split
2) Size of loading batch

"""

import argparse
import csv
import json
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger

from ct_sampler_aug import CTAugmentedSampler
from ctpy.h5_samplers import CTSampler
from tf_ext import TFH5SplitManager, TFH5Dataset
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


def run_test(config_path: Path, num_iter: int = 1) -> List[Dict[str, Any]]:
    """
    Run I/O test.

    Iterate over dataset with the specified parameters and check average time.
    Test allows to vary:
      * datasets amount
      * dataset split
      * size of loading buffer

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

    logger.debug('Create generators...')
    generator_configs = test_config['generators']
    generators = {}
    for g_name, g_params in generator_configs.items():
        g_class = g_params['class']
        if len(g_params['crop_shape']) != 3:
            raise IOError(
                f'Config {g_params["crop_shape"]=} is undefined.'
            )
        crop_shape = (g_params['crop_shape'][0], g_params['crop_shape'][1],
                      g_params['crop_shape'][2])
        if g_class == 'CTSampler':
            generators[g_name] = CTSampler(crop_shape=crop_shape)
        elif g_class == 'CTAugmentedSampler':
            generators[g_name] = CTAugmentedSampler(crop_shape=crop_shape)
    logger.debug('OK.')

    logger.debug('Iterate through tests...')
    for _, test_case in enumerate(test_config['tests']):
        ds_list = test_case['datasets']
        ds_configs: List[TFH5SplitManager.DSConfig] = []
        for ds in ds_list:
            ds_configs.append(
                TFH5SplitManager.DSConfig(
                    path=ds['h5_path'], split_n=ds['split_number'],
                    generator=generators[ds['generator_name']])
            )
        h5_splits = TFH5SplitManager(ds_configs)
        logger.debug(f'Amount of different managers is: {len(h5_splits)}')
        dset = TFH5Dataset(h5_splits, num_workers=cpu_count(),
                           block_length=test_case['block_length'])
        dset = dset.batch(16).prefetch(1)

        params = {}
        for j, ds in enumerate(ds_list):
            for k, v in ds.items():
                params[f'{j}_{k}'] = v
        title = '; '.join([f'{k} - {v}' for k, v in params.items()])
        avg_time = profile_dataset(dset, title=title, num_iter=num_iter)
        results.append({**params, 'avg_time': avg_time})
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

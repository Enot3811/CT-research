"""Script for profiling tf.data.Dataset for CT with TF tools."""

import argparse
import json
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers

from ct_sampler_aug import CTAugmentedSampler
from ctpy.h5_samplers import CTSampler
from pynative_ext.argparse.types import natural_int
from tf_ext import TFH5Dataset, TFH5SplitManager
from tf_ext.misc.resources import tf_control


def parse_args():
    """Create & parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--batch-size', type=natural_int, default=16,
                        help='Count of cases to use during one train '
                             'iteration.')
    parser.add_argument('--workers', type=natural_int, default=cpu_count(),
                        help='Count of parallel processes to use during data '
                             'sampling.')
    parser.add_argument('--epochs', type=natural_int, default=10,
                        help='Count of epochs to test.')

    in_args = parser.add_argument_group('Inputs')
    in_args.add_argument('--config', required=True, type=Path,
                         help='Path to the config to use during '
                              'TF performance collection.')

    out_args = parser.add_argument_group('Outputs')
    out_args.add_argument('--log-dir', type=Path, required=True,
                          help='Path to the folder where tensorboard '
                               'will write.')

    args = parser.parse_args()
    return args


@tf_control
def main():
    """Application entry point."""
    args = parse_args()
    n_classes = 10

    with open(args.config, 'r') as read_file:
        config = json.load(read_file)

    logger.debug('Create generators...')
    generator_configs = config['generators']
    generators = {}
    for g_name, g_params in generator_configs.items():
        g_class = g_params['class']
        del g_params['class']
        if g_class == 'CTReader':
            generators[g_name] = CTSampler(**g_params)
        elif g_class == 'CTAugmentedReader':
            generators[g_name] = CTAugmentedSampler(**g_params)
    logger.debug('OK.')

    ds_list = config['datasets']
    for ds in ds_list:
        ds['generator'] = generators[ds['generator_name']]
    h5_splits = TFH5SplitManager(ds_list)
    logger.debug(f'Amount of different managers is: {len(h5_splits)}')

    dset = TFH5Dataset(h5_splits, num_workers=args.workers)
    dset = dset.map(lambda v, n: (v[..., None],
                                  np.random.randint(n_classes)))
    dset = dset.batch(args.batch_size).prefetch(1)

    model = keras.Sequential(
        [
            # 96 x 96 x 96
            keras.Input(shape=(96, 96, 96)),
            layers.Conv2D(32, 3, activation='relu', padding='SAME'),
            layers.MaxPool2D(strides=(2, 2)),  # 48 x 48 x 32
            layers.Conv2D(16, 3, activation='relu', padding='SAME'),
            layers.MaxPool2D(),  # 24 x 24 x 16
            layers.Conv2D(8, 3, activation='relu', padding='SAME'),
            layers.MaxPool2D(),  # 12 x 12 x 8
            layers.Flatten(),  # 12 * 12 * 8
            layers.Dense(64, activation='relu'),
            layers.Dense(n_classes, activation='softmax')
        ]
    )

    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics='SparseCategoricalAccuracy')
    tbc = tf.keras.callbacks.TensorBoard(
        log_dir=args.log_dir,
        histogram_freq=1,
        update_freq='epoch',
        profile_batch=(1, args.epochs * args.batch_size)
    )
    model.fit(dset, epochs=args.epochs, callbacks=[tbc])


if __name__ == '__main__':
    main()

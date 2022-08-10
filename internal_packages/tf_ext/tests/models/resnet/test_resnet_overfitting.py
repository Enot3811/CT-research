"""
Tests checking that model can overfit on small data.
"""

from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers

from tf_ext.models import ResNet


def test_overfit_resnet2d(resnet2d_fe_fn: Callable[..., ResNet]):
    """Check that model can overfit on 2D synth data."""
    tf.keras.backend.clear_session()

    num_epochs = 8
    n_classes = 5
    ds_size = 80
    b_size = 8
    in_shape = (ds_size, 224, 224, 3)
    data = tf.unstack(tf.random.uniform(in_shape))
    labels = tf.unstack(tf.random.uniform((ds_size,), 0, n_classes,
                                          dtype=tf.int32))
    dset = tf.data.Dataset.from_tensor_slices((data, labels))
    dset = dset.batch(b_size)

    resnet2d_classifier = Sequential([
        resnet2d_fe_fn(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(n_classes)
    ])
    resnet2d_classifier.compile(
        optimizer=tf.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    results = resnet2d_classifier.fit(
        dset, epochs=num_epochs, verbose=0
    )

    diffs = np.diff(results.history['loss'])
    decrease_steps = np.count_nonzero(diffs < 0)
    increase_steps = diffs.size - decrease_steps
    assert decrease_steps > increase_steps, f'Diffs: {diffs}'

"""Implementation of simple multi-layer perceptron."""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout


class MLP(Layer):
    """Multi-layer perceptron."""

    def __init__(self, units, drop_rate, **kwargs):
        """
        Create multi-layer perceptron layer.

        Parameters
        ----------
        units : List[int]
            List of units to create.
        drop_rate : float
            DP rate after each dense layer to apply.
        kwargs : Dict
            Key-word arguments passed into parent class - tf.keras.Model.
        """
        super(MLP, self).__init__(**kwargs)

        # save all input arguments in some layer config
        self._config = {
            'units': units,
            'drop_rate': drop_rate,
            **kwargs
        }

        self._blocks = [
            (Dense(cur_units, activation=tf.nn.gelu),
             Dropout(drop_rate)) for cur_units in units
        ]

    def call(self, inputs, *args, **kwargs):
        """
        Run MLP layer on passed inputs.

        MLP is a sequence of Dense + Dropout.

        Parameters
        ----------
        inputs : tf.Tensor
            Input data for an MLP layer.
        args : List
            Not used.
        kwargs : Dict
            Not used.

        Returns
        -------
        tf.Tensor
            Output tensor of the MLP layer.
        """
        x = inputs
        for dense, dp in self._blocks:
            x = dense(x)
            x = dp(x)
        return x

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create layer from the config."""
        return cls(**config)

"""SWIN window shifter layer."""

import tensorflow as tf
from tensorflow.keras import layers


class SwinCyclicShifter(layers.Layer):
    """SWIN layer performing shift of input tensor for windows shifting."""

    def __init__(self, roll_size: int, dimensionality: int, **kwargs):
        """
        Initialize SwinCyclicShifter layer.

        Parameters
        ----------
        roll_size : int
            Size of shift along x and y. Usually it equal window size // 2.
        dimensionality : int
            Dimensionality of layer.

        Raises
        ------
        ValueError
            Dimensionality of data can be either 2 or 3.
        """
        super().__init__(**kwargs)

        self.layer_config = {
            'roll_size': roll_size,
            **kwargs
        }

        if dimensionality in {2, 3}:
            self.dimensionality = dimensionality
        else:
            raise ValueError(
                'Dimensionality of data can be either 2 or 3 but gotten '
                f'{dimensionality}.')

        self.roll_size = roll_size

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """Shift data in input tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Tensor with image/volume, shape is [B, H, W, C] or [B, H, W, D, C].
        args : List
            Not used.
        kwargs : Dict
            Not used.

        Returns
        -------
        tf.Tensor
            Tensor with shifted data, shape is [B, H, W, C] or [B, H, W, D, C].
        """
        x = inputs
        for i in range(1, self.dimensionality + 1):
            x = tf.roll(x, self.roll_size, axis=i)
        return x

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self.layer_config

    @classmethod
    def from_config(cls, config):
        """Create layer from the config."""
        return cls(**config)

"""
Bottleneck residual block implementation.
"""


import tensorflow as tf
from tensorflow.keras import layers


class BottleneckResidualBlock(layers.Layer):
    """
    The bottleneck residual block. Reduces channels before heavy convolution.
    More in the paper https://arxiv.org/pdf/1512.03385.pdf.

    Total idea can be defined as follows:
    * Linear projection into smaller channels (bottleneck)
    * Heave convolution
    * Linear projection into target channels
    * Shortcut linear projection in case when input not equal to the target
    * Sum of target and shortcut (or input)
    """

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        out_channels: int,
        dimensionality: int,
        stride: int,
        **kwargs
    ):
        """
        Initialize bottleneck residual block with specified parameters.

        Parameters
        ----------
        in_channels : int
            Block input channels.
        bottleneck_channels : int
            Number of channels in heavy convolution.
        out_channels : int
            Block output channels.
        dimensionality: int
            Dimensionality of input data.
        stride : int
            Convolution stride on this block.

        Raises
        ------
        ValueError
            Specified dimensionality does not supported.
        """
        super().__init__()

        self._config = {
            'in_channels': in_channels,
            'dimensionality': dimensionality,
            'bottleneck_channels': bottleneck_channels,
            'out_channels': out_channels,
            'stride': stride,
            **kwargs
        }

        if dimensionality == 2:
            conv = layers.Conv2D
        elif dimensionality == 3:
            conv = layers.Conv3D
        else:
            raise ValueError(
                f'Specified {dimensionality=} does not supported. '
                'It should be 2 or 3.')

        self.conv1 = conv(
            filters=bottleneck_channels,
            kernel_size=1)
        self.batchNorm1 = layers.BatchNormalization()
        self.activation1 = layers.ReLU()
        self.conv2 = conv(
            filters=bottleneck_channels,
            kernel_size=3,
            strides=(stride,) * dimensionality,
            padding='same')
        self.batchNorm2 = layers.BatchNormalization()
        self.activation2 = layers.ReLU()
        self.conv3 = conv(
            filters=out_channels,
            kernel_size=1)
        self.batchNorm3 = layers.BatchNormalization()
        self.activation3 = layers.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut_proj = tf.keras.Sequential([
                conv(filters=out_channels,
                     kernel_size=1,
                     strides=(stride,) * dimensionality),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut_proj = None

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Call bottleneck residual block of passed inputs.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        args: Any
            Not used.
        kwargs: Any
            Not used.

        Returns
        -------
        tf.Tensor
            Output tensor.
        """
        shortcut = inputs

        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.activation2(x)
        x = self.conv3(x)
        x = self.batchNorm3(x)

        if self.shortcut_proj is not None:
            shortcut = self.shortcut_proj(shortcut)

        x += shortcut
        x = self.activation3(x)

        return x

    def get_config(self):
        """Get config from which model can be created."""
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create model instance from its config."""
        return cls(**config)

"""Implementation of swin transformer layer."""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers

from .swin_cyclic_shifter import SwinCyclicShifter
from .patch_cutter import PatchCutter
from .mlp import MLP


class SwinWindowTransformLayer(layers.Layer):
    """
    SWIN window transformer layer.
    
    Performs self-attention on the windows into which data tensor is divided.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        mlp_ratio: int,
        drop_rate: float,
        output_dim: int,
        window_size: int,
        dimensionality: int,
        shift_size: Optional[int] = None,
        eps: float = 1e-6,
        **kwargs
    ):
        """
        Initialize SwinWindowTransformLayer.

        Parameters
        ----------
        num_heads : int
            Number of attention heads that used in this layer.
        key_dim : int
            Dimension of key in self-attention on this layer.
        mlp_ratio : int
            Coefficient of bottleneck expansion for hidden units of MLP
            that used after self-attention.
        drop_rate : float
            Dropout probability for MLP layers.
        output_dim : int
            Size of last dimension to be obtained after this layer.
            Usually this dimension equals input dimension
            and have to stay the same.
        window_size : int
            Size of window side, how many patches does the height/width
            of the window consist of.
        dimensionality : int
            Dimensionality of layer.
        shift_size : Optional[int]
            Size of shift step that is performed before windowing.
            Shift is needed for shifted window based self-attention.
            If not specified, this layer will not use shift.
        eps : float, optional
            Small value for stability when dividing, by default 1e-6

        Raises
        ------
        ValueError
            Dimensionality of data can be either 2 or 3.
        """
        super().__init__(**kwargs)

        # save all input arguments in some layer config
        self.layer_config = {
            'num_heads': num_heads,
            'key_dim': key_dim,
            'mlp_ratio': mlp_ratio,
            'drop_rate': drop_rate,
            'output_dim': output_dim,
            'window_size': window_size,
            'dimensionality': dimensionality,
            'shift_size': shift_size,
            'eps': eps,
            **kwargs
        }

        if dimensionality in {2, 3}:
            self.dimensionality = dimensionality
        else:
            raise ValueError(
                'Dimensionality of data can be either 2 or 3 but gotten '
                f'{dimensionality}.')

        self.window_size = window_size
        self.first_norm = layers.LayerNormalization(epsilon=eps)
        if shift_size is not None:
            self.shifter = SwinCyclicShifter(shift_size, dimensionality)
        else:
            self.shifter = tf.identity
        self.window_cutter = PatchCutter(window_size)
        self.attention = layers.MultiHeadAttention(num_heads,
                                                   key_dim,
                                                   dropout=drop_rate,
                                                   output_shape=output_dim)
        
        # Do expand dim in MLP and then return to initial dim
        mlp_units = [output_dim * mlp_ratio, output_dim]
        
        self.second_norm = layers.LayerNormalization(epsilon=eps)
        self.mlp = MLP(mlp_units, drop_rate)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        """
        Call window transformer layer of passed inputs.

        Transformer layer is complex layer of operations:
        * norm input
        * shift if shift_size was specified
        * divide into windows
        * pass through self-attention
        * reconstruct original shape from windows
        * sum inputs & attention result
        * norm sum
        * pass through MLP
        * norm MLP out
        * sum previous summation and normed MLP out

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor with patches.
            Shape is [batch_size x num_patches x channels_in_patch]
        args : List
            Not used.
        kwargs : Dict
            Not used.

        Returns
        -------
        tf.Tensor
            Output tensor with shape [batch_size x num_patches x output_dim]
        """
        x = inputs
        original_shape = tf.shape(x)

        # normed = self.first_norm(x)
        normed = x

        # Do shift
        shifted = self.shifter(normed)

        # Divide into windows with PatchCutter
        windowed = self.window_cutter(shifted)  # Здесь теряется размерность с количеством окон

        # Reshape PatchCutter result
        # bs x num_windows x window_dim ->
        # bs * num_windows x window_size ** dimensionality x patch_dim
        patch_cutter_shape = tf.shape(windowed)
        windowed = tf.reshape(
            windowed,
            (-1, self.window_size ** self.dimensionality, original_shape[-1]))

        # Windowed self-attention
        # self_att = self.attention(windowed, windowed)
        self_att = windowed

        # Return shape to original
        unwindowed = tf.reshape(self_att, patch_cutter_shape)

        num_patches = patch_cutter_shape[1]
        # TODO
        # n is 0-rank tf.Tensor and tf.split cannot receive it as argument
        # 1) Calculating with numpy does not work when using layer.Input.
        # 2) I tried to change the way this argument is specified
        # and tried specified n like "1-D integer Tensor or Python list
        # containing the sizes of each output tensor along axis"
        # but during using layer.Input I got an exception:
        # "ValueError: Exception encountered when calling layer
        # "swin_window_transform_layer" (type SwinWindowTransformLayer)."
        # 3) I implemented second realization of recovering that replaces
        # `split` with `unstack`, but unstack also requires numeric dimension.
        # It have to calculate number of unstacked tensors dynamically.

        n = tf.cast(tf.math.round(
            tf.math.pow(
                tf.cast(num_patches, tf.float32), 1 / self.dimensionality)),
            tf.int32)

        unsqueezed_shape = (original_shape[0],
                            num_patches,
                            *(self.window_size,) * self.dimensionality,
                            original_shape[-1])
        unwindowed = tf.reshape(unwindowed, unsqueezed_shape)
        # split_size = tf.cast(tf.shape(unwindowed)[1] / n, tf.int32)
        # splits = tf.ones((n,), tf.int32) * split_size
        # rows = tf.split(unwindowed, splits, axis=1, num=n)

        # First realization (old, only for 2D)
        # rows = tf.split(unwindowed, n, axis=1)
        # rows = [tf.concat(tf.unstack(x, axis=1), axis=2) for x in rows]

        # Second realization
        num_rows = tf.shape(unwindowed)[1]
        rows = tf.unstack(unwindowed, num_rows, axis=1)
        for concat_axis in range(self.dimensionality, 1, -1):
            split_indexes = tf.range(0, num_rows + 1, n)
            rows = [tf.concat(rows[split_indexes[i]:split_indexes[i + 1]],
                              axis=concat_axis)
                    for i in range(len(split_indexes) - 1)]
            num_rows = len(rows)
        unwindowed = tf.concat(rows, axis=1)

        # Sum, mlp, norm with original shape
        x = unwindowed + x
        normed = self.second_norm(x)
        mlp_out = self.mlp(normed)
        out = mlp_out + x
        return out

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self.layer_config

    @classmethod
    def from_config(cls, config):
        """Create layer from the config."""
        return cls(**config)

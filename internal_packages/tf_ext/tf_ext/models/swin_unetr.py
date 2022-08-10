"""Implementation of SWIN transformer model."""

import sys
import os
from typing import Tuple, List, Union

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow_addons.layers import InstanceNormalization

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tf_ext.layers import (
    SwinInputProjector, SwinPatchMerger, SwinWindowTransformLayer, PatchCutter)


class SwinUnetr(Model):
    """
    SWIN transformer.

    This model described in https://arxiv.org/pdf/2103.14030.pdf
    """

    class ResBlock(layers.Layer):
        """
        Residual block in SWIN UNETR.

        It consists of two postnormalized 3x3x3 convolutional layers
        with instance normalization.
        """

        def __init__(self, num_channels: int, dimensionality: int, **kwargs):
            """
            Initialize SWIN UNETR residual block.

            Parameters
            ----------
            num_channels : int
                Output channels of block.
            dimensionality : int
                Dimensionality of block. Must be either 2 or 3.
            """
            super().__init__(**kwargs)

            if dimensionality == 2:
                conv = layers.Conv2D
            elif dimensionality == 3:
                conv = layers.Conv3D
            else:
                raise ValueError(
                    'Dimensionality must be 2 or 3, '
                    f'but gotten {dimensionality=}')
            self.first_conv = conv(filters=num_channels,
                                   kernel_size=3,
                                   padding='same')
            self.first_norm = InstanceNormalization()
            self.second_conv = conv(filters=num_channels,
                                    kernel_size=3,
                                    padding='same')
            self.second_norm = InstanceNormalization()

        def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
            """
            Call SWIN UNETR residual block of passed inputs.

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
            x = self.first_conv(inputs)
            x = self.first_norm(x)
            x = self.second_conv(x)
            out = self.second_norm(x)
            return out

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        patch_size: int = 2,
        window_size: int = 4,
        proj_dim: int = 48,
        downsampling: int = 2,
        stage_num_blocks: Tuple[int, ...] = (2, 2, 2, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        key_dim: int = 32,
        mlp_ratio: int = 4,
        drop_rate: float = 0.1,
        eps: float = 1e-6,
        **kwargs
    ):
        """
        Initialize SWIN UNETR with specified parameters.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of input data.
        patch_size : int, optional
            Size of patch side in pixels.
        window_size : int, optional
            Size of window side in patches.
        proj_dim : int, optional
            Dimension for projection after patch cutting
            and patch merging. In paper defined as C.
        downsampling : int, optional
            Resolution downsampling coefficient for patch merging.
            If patch resolution was n x n and downsampling equals 2
            then patch resolution will be n/2 x n/2.
        stage_num_blocks : Tuple[int, ...], optional
            Numbers of SWIN transformer blocks on each stage.
        num_heads : Tuple[int, ...], optional
            Number of attention heads on each stage.
        key_dim : int, optional
            Key dimension in self-attention.
        mlp_ratio : int, optional
            Coefficient of expansion for hidden units of MLP
            that used after self-attention.
        drop_rate : float, optional
            Dropout probability for MLP layers.
        eps : float, optional
            Small value for stability when dividing.

        Raises
        ------
        ValueError
            num_heads and stage_num_blocks must have the same length.
        """
        super().__init__(**kwargs)

        if len(stage_num_blocks) != len(num_heads):
            raise ValueError(
                'num_heads and stage_num_blocks must have the same length.')

        self._config = {
            'input_shape': input_shape,
            'patch_size': patch_size,
            'window_size': window_size,
            'proj_dim': proj_dim,
            'downsampling': downsampling,
            'stage_num_blocks': stage_num_blocks,
            'num_heads': num_heads,
            'key_dim': key_dim,
            'mlp_ratio': mlp_ratio,
            'drop_rate': drop_rate,
            'eps': eps,
            **kwargs
        }

        if len(input_shape) == 3:
            self.dimensionality = 2
            deconv = layers.Conv2DTranspose
        elif len(input_shape) == 4:
            self.dimensionality = 3
            deconv = layers.Conv3DTranspose
        else:
            raise NotImplementedError(
                'SWIN can work only with 2D and 3D data. '
                f'but gotten {input_shape=} does not fit neither 2D nor 3D.')
        
        self.input_patch_partition = PatchCutter(patch_size)
        patch_count = (input_shape[0] // patch_size) ** self.dimensionality
        self.input_projector = SwinInputProjector(
            project_dim=proj_dim,
            patch_count=patch_count,
            dimensionality=self.dimensionality)

        self.swin_stages: List[List[layers.Layer]] = []
        self.upsampler_resblocks: List[SwinUnetr.ResBlock] = []
        self.skip_resblocks: List[SwinUnetr.ResBlock] = []
        self.deconv_layers: List[Union[layers.Conv2DTranspose,
                                       layers.Conv3DTranspose]] = []

        # Add higher ResBlocks
        self.upsampler_resblocks.append(
            SwinUnetr.ResBlock(num_channels=proj_dim,
                               dimensionality=self.dimensionality,
                               name=f'Upsampler_res {proj_dim}'))
        self.skip_resblocks.append(
            SwinUnetr.ResBlock(num_channels=proj_dim,
                               dimensionality=self.dimensionality,
                               name=f'Skip_res {proj_dim}'))
        self.deconv_layers.append(
            deconv(filters=proj_dim, kernel_size=3, strides=2, padding='same'))

        # Initialize stages
        for i, num_blocks in enumerate(stage_num_blocks):
            num_channels = 2 ** i * proj_dim
            stage_layers = []

            for j in range(num_blocks):
                shift_sizes = [None, int(window_size / 2)]

                stage_layers.append(SwinWindowTransformLayer(
                    num_heads=num_heads[i],
                    key_dim=key_dim,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    output_dim=num_channels,
                    window_size=window_size,
                    dimensionality=self.dimensionality,
                    shift_size=shift_sizes[j % 2]))

            stage_layers.append(SwinPatchMerger(
                downsampling=downsampling,
                num_channels=num_channels * 2,
                dimensionality=self.dimensionality))
            self.swin_stages.append(stage_layers)

            self.deconv_layers.append(deconv(filters=num_channels,
                                             kernel_size=3,
                                             strides=2,
                                             padding='same',
                                             name=f'Deconv {num_channels}'))
            self.upsampler_resblocks.append(
                SwinUnetr.ResBlock(num_channels=num_channels,
                                   dimensionality=self.dimensionality,
                                   name=f'Upsampler_res {num_channels}'))
            self.skip_resblocks.append(
                SwinUnetr.ResBlock(num_channels=num_channels,
                                   dimensionality=self.dimensionality,
                                   name=f'Skip_res {num_channels}'))

        # Add bottom ResBlock
        num_channels = 2 ** len(stage_num_blocks) * proj_dim
        self.upsampler_resblocks.append(
            SwinUnetr.ResBlock(num_channels=num_channels,
                               dimensionality=self.dimensionality,
                               name=f'Upsampler_res {num_channels}'))

        # Reverse upsampler and deconvs
        self.upsampler_resblocks.reverse()
        self.deconv_layers.reverse()
        
    def call(
        self,
        inputs: tf.Tensor,
        training: tf.Tensor = None,
        mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Call SWIN UNETR model on batch of images.

        Parameters
        ----------
        inputs : tf.Tensor
            Images/volumes tensor.
            It's assumed that it's shape is [B, H, W, C] or [B, H, W, D, C].
        training : tf.Tensor
            Training control flag. Not used.
        mask : tf.Tensor
            Mask for data. Not used.

        Returns
        -------
        tf.Tensor
            Output tensor with extracted features.
        """
        skips = []

        # Do first skip from input
        skips.append(self.skip_resblocks[0](inputs))

        # Patch partition and second skip
        x = self.input_patch_partition(inputs)
        # Reshape from [B x num_patches x C] to [B x H x W (x D) x C] for skip
        shape = tf.shape(x)
        n = tf.cast(tf.math.round(
            tf.math.pow(
                tf.cast(shape[1], tf.float32), 1 / self.dimensionality)),
            tf.int32)
        new_shape = (shape[0], *(n,) * self.dimensionality, shape[-1])
        reshaped_x = tf.reshape(x, new_shape)
        skips.append(self.skip_resblocks[1](reshaped_x))

        # SWIN stages and saving skips
        x = self.input_projector(x)
        for i, stage in enumerate(self.swin_stages):
            for layer in stage:
                x = layer(x)
            # Do skip after stages except last stage
            if i != len(self.swin_stages) - 1:
                skips.append(self.skip_resblocks[i + 2](x))

        # Upsampling
        skips.reverse()
        for i, upsampler_resblock in enumerate(self.upsampler_resblocks):

            # Do concatenation with skips
            if i != 0:
                x = tf.concat([skips[i - 1], x], axis=-1)
            
            x = upsampler_resblock(x)

            # Do deconvolution to expand resolution
            if i != len(self.upsampler_resblocks) - 1:
                x = self.deconv_layers[i](x)

        return x

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create layer from the config."""
        return cls(**config)

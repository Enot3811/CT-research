"""Implementation of SWIN transformer model."""

from typing import Tuple, Dict, Callable
import sys
import os

import tensorflow as tf
from tensorflow.keras import Model

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tf_ext.layers import (
    SwinInputProjector, SwinPatchMerger, SwinWindowTransformLayer, PatchCutter)


class SwinTransformer(Model):
    """
    SWIN transformer.

    This model described in paper https://arxiv.org/pdf/2103.14030.pdf
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        patch_size: int = 4,
        window_size: int = 7,
        proj_dim: int = 96,
        downsampling: int = 2,
        stage_num_blocks: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        key_dim: int = 32,
        mlp_ratio: int = 4,
        drop_rate: float = 0.1,
        eps: float = 1e-6,
        **kwargs
    ):
        """
        Initialize SWIN transformer with specified parameters.

        All default values for parameters gotten from
        https://arxiv.org/pdf/2103.14030.pdf

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

        if len(input_shape) not in {3, 4}:
            raise NotImplementedError(
                'SWIN can work only with 2D and 3D data. '
                f'but gotten {input_shape=} does not fit neither 2D nor 3D.')
        else:
            dimensionality = len(input_shape) - 1
        
        self.input_patch_partition = PatchCutter(patch_size)
        self.model_layers = []

        for i, num_blocks in enumerate(stage_num_blocks):
            num_channels = 2 ** i * proj_dim

            # The first stage is different from the other
            if i == 0:
                patch_count = (input_shape[0] // patch_size) ** dimensionality
                self.model_layers.append(
                    SwinInputProjector(project_dim=proj_dim,
                                       patch_count=patch_count,
                                       dimensionality=dimensionality))
            else:
                self.model_layers.append(
                    SwinPatchMerger(downsampling=downsampling,
                                    num_channels=num_channels,
                                    dimensionality=dimensionality))

            for j in range(num_blocks):
                shift_sizes = [None, int(window_size / 2)]

                self.model_layers.append(SwinWindowTransformLayer(
                    num_heads=num_heads[i],
                    key_dim=key_dim,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    output_dim=num_channels,
                    window_size=window_size,
                    dimensionality=dimensionality,
                    shift_size=shift_sizes[j % 2]))

    def call(
        self,
        inputs: tf.Tensor,
        training: tf.Tensor = None,
        mask: tf.Tensor = None
    ) -> tf.Tensor:
        """
        Call SWIN transformer model on batch of images.

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
        x = self.input_patch_partition(inputs)
        for layer in self.model_layers:
            x = layer(x)
        return x

    def get_config(self):
        """Get layer config from which this layer can be created again."""
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Create layer from the config."""
        return cls(**config)


class SwinFactory:
    """
    Factory for SWIN transformer models.
    """

    @staticmethod
    def _create_swin_t(
        input_shape: Tuple[int, ...]
    ) -> SwinTransformer:
        """
        Create SWIN-T transformer.

        Returns
        -------
        ResNet
            SWIN-T transformer model.
        """
        return SwinTransformer(input_shape=input_shape)

    @staticmethod
    def _create_swin_s(
        input_shape: Tuple[int, ...]
    ) -> SwinTransformer:
        """
        Create SWIN-S transformer.

        Returns
        -------
        ResNet
            SWIN-S transformer model.
        """
        return SwinTransformer(input_shape=input_shape,
                               stage_num_blocks=(2, 2, 18, 2))

    @staticmethod
    def _create_swin_b(
        input_shape: Tuple[int, ...]
    ) -> SwinTransformer:
        """
        Create SWIN-B transformer.

        Returns
        -------
        ResNet
            SWIN-B transformer model.
        """
        return SwinTransformer(input_shape=input_shape,
                               proj_dim=128,
                               stage_num_blocks=(2, 2, 18, 2),
                               num_heads=(4, 8, 16, 32))

    @staticmethod
    def _create_swin_l(
        input_shape: Tuple[int, ...]
    ) -> SwinTransformer:
        """
        Create SWIN-L transformer.

        Returns
        -------
        ResNet
            SWIN-L transformer model.
        """
        return SwinTransformer(input_shape=input_shape,
                               proj_dim=192,
                               stage_num_blocks=(2, 2, 18, 2),
                               num_heads=(6, 12, 24, 48))

    def __init__(self):
        self.__constructors: Dict[str, Callable[..., SwinTransformer]] = {
            'SWIN_T': self._create_swin_t,
            'SWIN_S': self._create_swin_s,
            'SWIN_B': self._create_swin_b,
            'SWIN_L': self._create_swin_l
        }

    def create_swin(
        self,
        model_name: str,
        input_shape: Tuple[int, ...]
    ) -> SwinTransformer:
        """
        Create SWIN transformer model.

        Parameters
        ----------
        model_name : str
            Name of SWIN.
            It should be one of ["SWIN_T", "SWIN_S", "SWIN_B", "SWIN_L"].
        input_shape : Tuple[int, ...]
            Input data shape.
            It should be [H, W, C] or [H, W, D, C].

        Raises
        ------
        ValueError
            Creation of specified model name is not supported.
        """
        if model_name not in self.__constructors:
            raise ValueError(f'Creation of {model_name=} is not supported.')
        return self.__constructors[model_name](input_shape=input_shape)

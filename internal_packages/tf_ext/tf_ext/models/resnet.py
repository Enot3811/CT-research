"""Module with ResNet implementation and factory for its easy creation."""

from functools import partial
from typing import List, Optional, Dict, Any, Callable

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from ..layers import BottleneckResidualBlock, ResidualBlock


class ResNet(Model):
    """
    ResNet model from the paper https://arxiv.org/pdf/1512.03385.pdf.
    """

    def __init__(
        self,
        stage_num_blocks: List[int],
        dimensionality: int,
        initial_channels: int = 64,
        channels_upsampling: int = 2,
        bottlenecks_coef: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize ResNet with specified parameters.

        Parameters
        ----------
        stage_num_blocks : List[int]
            Numbers of ResNet blocks on each stage.
            Number of stages is defined by list length.
        dimensionality : int
            Dimensionality of the input data.
            Only 2 & 3 are supported.
        initial_channels : int, optional
            Number of channels after initial convolution
            as well as the base for channel upsampling.
        channels_upsampling : int, optional
            Coefficient of channel upsampling after each stage.
            Formula: ``channels_count = base * upsmpl ** stage_i``.
        bottlenecks_coef : Optional[float], optional
            Reducing coefficient for bottleneck realization.
            If it is specified, ResNet will use `BottleneckResidualBlock`
            otherwise `ResidualBlock`. Bottleneck channels count will be
            calculated as coefficient multiplied on layer channels.

        Raises
        ------
        ValueError
            Unsupported dimensionality was passed.
        """
        super().__init__(**kwargs)

        self._config = {
            'dimensionality': dimensionality,
            'stage_num_blocks': stage_num_blocks,
            'initial_channels': initial_channels,
            'channels_upsampling': channels_upsampling,
            'bottlenecks_coef': bottlenecks_coef,
            **kwargs
        }
        self.dimensionality = dimensionality
        self._init_encoder(self._config)

    def _init_encoder(self, model_config: Dict[str, Any]):
        """
        Initialize encoder part of ResNet.

        Parameters
        ----------
        model_config : Dict[str, Any]
            Configs for encoder.
        """
        if self.dimensionality == 2:
            conv_class = layers.Conv2D
        elif self.dimensionality == 3:
            conv_class = layers.Conv3D
        else:
            raise ValueError(f'Unsupported {self.dimensionality=}.')

        # Initial stage creation: simple conv with stride
        # Replace original max-pool from paper with first residual block stride
        initial_conv = conv_class(
            filters=model_config['initial_channels'],
            kernel_size=7,
            strides=(2,) * self.dimensionality,
            padding='same')
        initial_bn = layers.BatchNormalization()
        initial_act = layers.ReLU()
        self.init_stage = Sequential([
            initial_conv,
            initial_bn,
            initial_act
        ], name='stage_1')

        # Residual stages creation
        self.stages = []
        in_channels = model_config['initial_channels']
        for i in range(len(model_config['stage_num_blocks'])):
            # Define channels of the stage
            # Each next stage has new channels count defined by formula
            out_channels = (model_config['channels_upsampling'] ** i *
                            model_config['initial_channels'])

            # In case when resnet uses new residual layers - bottle-neck layer
            # It's necessary to define bottle-neck channels count
            bottleneck_channels: Optional[int]
            if model_config['bottlenecks_coef'] is not None:
                bottleneck_channels = int(
                    out_channels * model_config['bottlenecks_coef']
                )
                residual_layer = partial(
                    BottleneckResidualBlock,
                    bottleneck_channels=bottleneck_channels
                )
            else:
                residual_layer = partial(ResidualBlock)

            # Define blocks of the stage
            # First block - residual(from=in_ch, to=out_ch)
            # All next blocks - residual(from=out_ch, to=out_ch)
            # Next stage first block must start from in_ch = out_ch
            num_residual_blocks = model_config['stage_num_blocks'][i]
            blocks = []
            for j in range(num_residual_blocks):
                # Spatial change only at first layer follow to the arch
                stride = 2 if j == 0 else 1
                blocks.append(residual_layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dimensionality=self.dimensionality,
                    stride=stride))
                in_channels = out_channels
            # +2 because iteration from 0 and first is alredy defined
            self.stages.append(Sequential(blocks, name=f'stage_{i + 2}'))

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        """
        Call ResNet on batch of data.

        Parameters
        ----------
        inputs : tf.Tensor
            Data tensor.
            It's assumed that it's shape is ``[B, H, W, C]`` (for 2D)
            or ``[B, H, W, D, C]`` (for 3D).
        training : tf.Tensor
            Training control flag. Not used.
        mask : tf.Tensor
            Mask for data. Not used.

        Returns
        -------
        tf.Tensor
            Output tensor: ``[B, NEW_H, NEW_W, features]`` for 2D. Same on 3D.
        """
        x = self.init_stage(inputs)
        for stage in self.stages:
            x = stage(x)
        return x

    def get_config(self):
        """Get config from which model can be created."""
        return self._config

    @classmethod
    def from_config(cls, config, custom_objects=None) -> 'ResNet':
        """Create model instance from its config."""
        return cls(**config)


class ResNetFactory:
    """
    Factory of ResNet models.
    """

    @staticmethod
    def _create_resnet18(dimensionality: int) -> ResNet:
        """
        Create ResNet18.

        Returns
        -------
        ResNet
            ResNet18 model.
        """
        return ResNet(stage_num_blocks=[2, 2, 2, 2],
                      dimensionality=dimensionality,
                      name='resnet18')

    @staticmethod
    def _create_resnet34(dimensionality: int) -> ResNet:
        """
        Create ResNet34.

        Returns
        -------
        ResNet
            ResNet34 model.
        """
        return ResNet(stage_num_blocks=[3, 4, 6, 3],
                      dimensionality=dimensionality,
                      name='resnet34')

    @staticmethod
    def _create_resnet50(dimensionality: int) -> ResNet:
        """
        Create ResNet50.

        Returns
        -------
        ResNet
            ResNet50 model.
        """
        return ResNet(
            stage_num_blocks=[3, 4, 6, 3],
            dimensionality=dimensionality,
            channels_upsampling=2,
            bottlenecks_coef=0.25,
            initial_channels=256,
            name='resnet50'
        )

    def __init__(self):
        self.__constructors: Dict[str, Callable[..., ResNet]] = {
            'ResNet18': self._create_resnet18,
            'ResNet34': self._create_resnet34,
            'ResNet50': self._create_resnet50,
        }

    def create_resnet(
        self,
        model_name: str,
        dimensionality: int
    ) -> ResNet:
        """
        Create ResNet model.

        Parameters
        ----------
        model_name : str
            Name of ResNet.
            It should be one of ["ResNet18", "ResNet34", "ResNet50"].
        dimensionality: int
            Dimensionality of the input data: ``2`` or ``3``.

        Raises
        ------
        ValueError:
            Unsupported model name.
        """
        if not(model_name in self.__constructors.keys()):
            raise ValueError(f'Creation of {model_name=} is not supported.')
        
        return self.__constructors[model_name](dimensionality=dimensionality)

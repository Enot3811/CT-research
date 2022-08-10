"""Tensorflow custom layers."""


from .mlp import MLP
from .transform_layer import TransformLayer
from .patch_cutter import PatchCutter
from .bottleneck_residual_block import BottleneckResidualBlock
from .residual_block import ResidualBlock

CUSTOM_OBJECTS = {
    'MLP': MLP,
    'TransformLayer': TransformLayer,
    'PatchCutter': PatchCutter,
    'BottleneckResidualBlock': BottleneckResidualBlock,
    'ResidualBlock': ResidualBlock
}

__all__ = list(CUSTOM_OBJECTS.keys())

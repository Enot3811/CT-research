"""Tensorflow custom models."""

from .resnet import ResNet, ResNetFactory  # noqa

CUSTOM_OBJECTS = {
    'ResNet': ResNet
}

__all__ = list(CUSTOM_OBJECTS.keys())

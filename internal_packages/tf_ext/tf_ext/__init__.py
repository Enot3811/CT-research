"""
Package with tensorflow extensions: models, losses, e.t.c.

All main project's functionality that work in TF will be implemented here.
"""


from .tfh5_dataset import TFH5Dataset, TFH5Generator, TFH5SplitManager  # noqa
from .h5_serialization import KerasH5Manager  # noqa

from .layers import CUSTOM_OBJECTS as CUSTOM_LAYERS
from .models import CUSTOM_OBJECTS as CUSTOM_MODELS

CUSTOM_OBJECTS = {**CUSTOM_LAYERS, **CUSTOM_MODELS}

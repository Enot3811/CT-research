"""
Sub-package that contains sampler-tools from different CT-data from h5-ds.

Each sampler generates task-specific data from specified h5-dataset.

Each sampler is correspond to some specific CV task: classification,
segmentation, detection, e.t.c. Each sampler works with h5-ds only, it's
expected limitation. Different samplers may use the same `Sample` inside to
sample data from. Each sampler yields array-data, no custom structures is
expected because no ML-framework can work with it correctly. Each sampler is
the last bridge between abstract view of some complex data and "clean" arrays.

Each sampler is already implements necessary for tensorflow methods.
"""

from .spine_cls_2d import SpineCLSSampler2D  # noqa
from .spine_segm_2d import SpineSGMSampler2D  # noqa
from .spine_segm_3d import SpineSGMSampler3D  # noqa
from .knee_sampler import KneeSampler  # noqa

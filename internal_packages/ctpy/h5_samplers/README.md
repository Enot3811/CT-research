## Description

Sub-package that contains sampler-tools from different CT-data from h5-ds.

Each sampler generates task-specific data from specified h5-dataset.

Each sampler is correspond to some specific CV task: classification,
segmentation, detection, e.t.c. Each sampler works with h5-ds only, it's
expected limitation. Different samplers may use the same `Sample` inside to
sample data from. Each sampler yields array-data, no custom structures is
expected because no ML-framework can work with it correctly. Each sampler is
the last bridge between abstract view of some complex data and "clean" arrays.

Each sampler is already implements necessary for tensorflow methods.

## Contents

### [`knee_sampler.py`](knee_sampler.py)

2D Classification sampler for knee X-ray dataset.

It can be used as a generator that yields 2D image of X-ray and its label.

### [`sampler.py`](sampler.py)

Base sampler implementation to generalize some common logic.

### [`spine_cls_2d.py`](spine_cls_2d.py)

Sampler that creates generator of 2D spine classification data.

2D spine classification data = CT-slice & its label (has spine or not).

### [`spine_segm_2d.py`](spine_segm_2d.py)

Sampler that creates generator of 2D spine segmentation data.

2D spine segmentation data = CT-slice & its binary mask (where 1 is a spine).

### [`spine_segm_3d.py`](spine_segm_3d.py)

3D segmentation sampler for CT Spine dataset.

This sampler may be used for binary segmentation task:
mark vertebra pixels on CT volume.

It can be used as a generator that yields 3D tile from CT and its 3D mask.

### [`types.py`](types.py)

Types definition during sub-package.
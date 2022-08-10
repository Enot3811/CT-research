## Description

Sub-package for CT-scans data reading.

## Contents

### [`datasets`](datasets)

Sub-package for wrapping DS-s with CT.
Each DS contains at least ``CTSample`` generator creator and other possible.
Package also contains factory that create necessary ds-class from passed root.

### [`h5_samplers`](h5_samplers)

Sub-package that contains sampler-tools from different CT-data from h5-ds.
Each sampler generates task-specific data from specified h5-dataset.
Each sampler is correspond to some specific CV task: classification,
segmentation, detection, e.t.c. Each sampler works with h5-ds only, it's
expected limitation. Different samplers may use the same `Sample` inside to
sample data from. Each sampler yields array-data, no custom structures is
expected because no ML-framework can work with it correctly. Each sampler is
the last bridge between abstract view of some complex data and "clean" arrays.
Each sampler is already implements necessary for tensorflow methods.

### [`samples`](samples)

Sub-package with `Sample`-instances of different CT views.
Each CT view may contain only CT or with some metadata, or some annotation.

### [`serializers`](serializers)

Sub-package with serialize tools for different `Sample`-classes.
Each serialize-tool knows how to read & dump sample from different FS-views.
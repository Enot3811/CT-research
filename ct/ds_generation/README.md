## Description

Scripts & modules that performs CT-datasets generation.

CT-dataset is a h5-file with all necessary data for specific task.
This dataset should contain at least CT-data & resolution information.

Task-specific datasets may contain other info: labels, masks, e.t.c.

## Contents

### [`gen_h5.py`](gen_h5.py)

Generate h5-datasets from original CT-dataset which is supported.

Created h5 datasets will contain `CTSample` objects.

Script takes ds-root as input path, creates CTDataset object and generates
h5-dataset for each: train, test & val.

### [`gen_h5_uniform.py`](gen_h5_uniform.py)

Script that converts CT dataset to the only one fixed voxel size.

### [`gen_splits.py`](gen_splits.py)

Script for creating splits for CT datasets.

Script gets all sample names and split them according to specified fractions.
Default dataset splits are ``train.txt``, ``val.txt`` and ``test.txt``.

If there are some new samples that were not written in existing splits,
they are distributed according to specified fractions.

### [`inspect_ds.py`](inspect_ds.py)

Debug script that create ``DSGenerator`` objects for specified dataset roots
and iterate over them visualizing read ``CTSample`` objects.
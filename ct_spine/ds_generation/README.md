## Description

DOC: TODO

## Contents

### [`tests`](tests)

Functional tests for dataset generation.

### [`gen_h5.py`](gen_h5.py)

Generate h5-dataset from CTSpine_1k folder with `SpineSample` objects.

Script takes ds-root as input path, creates CTDataset object and generates
h5-dataset for each: train, test & val.

### [`gen_h5_uniform.py`](gen_h5_uniform.py)

Script that converts spine CT dataset to the only one fixed voxel size.

### [`inspect_h5.py`](inspect_h5.py)

Inspect passed HDF5 dataset with `SpineSample` data.
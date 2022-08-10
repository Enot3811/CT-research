## Description

H5 generation scripts for kneeKL dataset.

## Contents

### [`gen_h5.py`](gen_h5.py)

Generate h5-datasets from KneeKL X-ray dataset.

Script takes ds-root as input path, creates X-ray Dataset object and generates
h5-dataset for each: train, test & val.

### [`inspect_h5.py`](inspect_h5.py)

Inspect passed HDF5 dataset with `KneeSample` data.
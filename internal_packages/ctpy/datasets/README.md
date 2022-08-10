## Description

Sub-package for wrapping DS-s with CT.

Each DS contains at least ``CTSample`` generator creator and other possible.

Package also contains factory that create necessary ds-class from passed root.

## Contents

### [`ctspine1k.py`](ctspine1k.py)

Module with CTSpine_1k dataset wrapper implementation.

This view allows to get necessary samples from DS: `CTSample` or `SpineSample`.

### [`dataset.py`](dataset.py)

Base class for each DS to generate data from it.

### [`ds_factory.py`](ds_factory.py)

Factory class for creating DSGenerators with different types.

### [`knee_kl.py`](knee_kl.py)

Module with KneeKL X-ray dataset wrapper implementation.

This view allows to get necessary samples from DS: `KneeSample`, names, e.t.c

### [`mosmed.py`](mosmed.py)

Module with CTGenerator for Mosmed dataset.

`MosmedDS` returns sample names.
Its CTSample-generator extracts dicom files from mosmed tars to temp folder
and then reads them.

### [`msd.py`](msd.py)

Module with CTGenerator for all datasets included to MSD.

MSD returns path relative to root directory.
Its CTSample generator reads NIfTI.gz files from any split of MSD.
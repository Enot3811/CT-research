## Description

Sub-package with `Sample`-instances of different CT views.

Each CT view may contain only CT or with some metadata, or some annotation.

## Contents

### [`ctsample.py`](ctsample.py)

CT data storing structure.

This structure has only CT data and resolution data,
no annotation or other additional information.

### [`knee_sample.py`](knee_sample.py)

Knee X-ray data storing structure.

This structure has only X-ray data with corresponding label,
no annotation or other additional information.

### [`spine_sample.py`](spine_sample.py)

Spine sample data storing structure.

This structure has only CT data, resolution data and semantic mask,
no annotation or other additional information.
## Description

Sub-package with serialize tools for different `Sample`-classes.

Each serialize-tool knows how to read & dump sample from different FS-views.

## Contents

### [`ctsample_serialize.py`](ctsample_serialize.py)

Class-wrapper that dispatches all serialization functionality for `CTSample`.

### [`knee_sample_serialize.py`](knee_sample_serialize.py)

Class-wrapper that dispatches all serialization functionality for `KneeSample`.

### [`spine_sample_serialize.py`](spine_sample_serialize.py)

Class-wrapper dispatching all serialization functionality for `SpineSample`.
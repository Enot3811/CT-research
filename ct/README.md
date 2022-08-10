## Description

DOC: TODO

## Contents

### [`ds_generation`](ds_generation)

Scripts & modules that performs CT-datasets generation.
CT-dataset is a h5-file with all necessary data for specific task.
This dataset should contain at least CT-data & resolution information.
Task-specific datasets may contain other info: labels, masks, e.t.c.

### [data_sampling](data_sampling)

Tools for CT data sampling: generators, test script for searching optimal
pipeline.
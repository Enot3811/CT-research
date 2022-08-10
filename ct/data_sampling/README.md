## Description

DOC: TODO

## Contents

### [`configs`](configs)

Configs for debug scripts that require them.

### [`ct_sampler_aug.py`](ct_sampler_aug.py)

CT sampler with augmentation implementation.

Uses simulation of the real augmentations to waste the time. Just for tests.

### [`test_aug.py`](test_aug.py)

Script for testing TF augmentations best place.

This script checks for each test configuration:
1) augmentations in generator performance
2) augmentations in map performance

### [`test_io.py`](test_io.py)

Script for testing TF I/O default tools.

It allows to vary
1) Amount of datasets to load from
1) Each dataset split
2) Size of loading batch

### [`test_tb.py`](test_tb.py)

Script for profiling tf.data.Dataset for CT with TF tools.

## Loading pipeline tests

1. Generate synth ds (or several ds-s) using [script](generate_synth_h5.py).
2. Take config templates at [configs](configs).
3. Replace paths to datasets using fast-replace of "DS" key.
4. Run [test_io.py](test_io.py) with corresponding configurations.
5. Run [test_aug.py](test_aug.py) with corresponding configuration.

# Content

Small description of each file or sub-folder in this folder.

## [configs](configs)

Sub-folder with test's configs with template names instead of ds-s paths.

## [debug_2d_load.py](debug_2d_load.py)

Script for debugging 2D CT slices load. It creates DS with split on CPU processes
and iterates along it till the end.

## [generate_synth_h5.py](generate_synth_h5.py)

Script for random h5 file with CT data generation. Allows varying of count & shape.

## [test_aug.py](test_aug.py)

Script runs specified reading configuration through profile benchmark in two modes:
  1. augs in generator
  2. augs in .map + @tf_function

This test allows to define what place is the best for augmentation.

Reading configuration is set via `json` config file. Its structure should be similar to:

```json
{
  "tests": [
    {
      "datasets": [
        {
          "h5_path": "PATH",
          "split_number": 1,
          "generator_name": "SOME_GEN_NAME"
        }
      ]
    }
  ],
  "generators": {
    "SOME_GEN_NAME": {
      "class": "VISIBLE_GENERATOR_CLASS_NAME",
      "crop_shape": [
        96,
        96,
        96
      ],
      "buffer_size": 1
    }
  }
}
```

Profiling benchmark is:
  1. Read Batch
  2. Simulate operations with fixed runtime for each

## [test_io.py](test_io.py)

Script runs specified reading configuration. Profile & configuration are similar to
`test_aug.py`.

## [test_tb.py](test_tb.py)

Script allows to run simple model training process using config from default templates.
The main purpose of this script - run training and profile it to analyze
data loading approach.
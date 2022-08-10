## Description

Configs for debug scripts that require them.

## Contents

### [`test_aug.json`](test_aug.json)

This config is used for `test_aug.py` script. It's assumed that each test will
be run using augmentations inside generator and using map function.

It has placeholder line: "DS" which should be replaced with the path to read dataset.

### [`test_io_1.json`](test_io_1.json)

This config is used for `test_io.py` script. It has placeholder line: "DS" which
should be replaced with the path to read dataset.

It's assumed that each test will be run directly as it written.

### [`test_io_2.json`](test_io_2.json)

This config is used for `test_io.py` script. It has placeholder line: "DS" which
should be replaced with the path to read dataset.

It's assumed that each test will be run directly as it written.

### [`test_tb.json`](test_tb.json)

This config is used for `test_tb.py` script. It has placeholder line: "DS" which
should be replaced with the path to read dataset.

It's assumed that script will be run using this config, run simple model training
and log profiling information in tensorboard. Later it's possible to inspect
these logs for analysing computation time.
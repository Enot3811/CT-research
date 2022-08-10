## Description

Train managers for typical models.

Each manager knows how to work with its specific models-set.
Each manager can be created as for many different models, as for just one.

## Contents

### [`classifier_manager.py`](classifier_manager.py)

Train manager for classification model.

### [`mae_manager.py`](mae_manager.py)

Train manager for MAE model.

### [`manager.py`](manager.py)

Base train-process manager.

Manager controls train process of keras-model & implements some default logic:
    - structured storing of the weights, metadata & metrics
    - handle NaN-s
    - handle plateau-case
    e.t.c.

### [`resnet_manager.py`](resnet_manager.py)

Train manager for ResNet model.
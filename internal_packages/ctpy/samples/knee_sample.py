"""
Knee X-ray data storing structure.

This structure has only X-ray data with corresponding label,
no annotation or other additional information.
"""

import numpy as np


class KneeSample:
    """
    Structure for storing one X-ray knee sample.
    """

    xray: np.ndarray
    """
    Knee X-ray pixel data.
    """

    name: str
    """
    Name of knee sample.
    """

    kl_label: int
    """
    Label of knee sample.
    """

    @property
    def shape(self):
        """Return shape of the knee X-ray sample."""
        return self.xray.shape

    def __init__(self,
                 xray: np.ndarray,
                 label: int,
                 name: str):
        self.xray = xray
        self.kl_label = label
        self.name = name

from __future__ import annotations

import numpy as np
from skcomponents.optics import LinearPolarizer


class MeadowlarkOWLPolarizer(LinearPolarizer):

    def __init__(self, orientation):
        wavelength = np.arange(400, 1650.0, 50.0)
        contrast = np.array(
            [
                2.000e05,
                6.000e05,
                2.000e06,
                3.000e06,
                3.000e06,
                2.000e06,
                1.500e06,
                2.000e05,
                1.000e04,
                3.000e03,
                1.200e03,
                1.100e03,
                1.100e03,
                1.100e03,
                1.185e03,
                1.270e03,
                1.355e03,
                1.440e03,
                1.525e03,
                1.610e03,
                1.695e03,
                1.780e03,
                1.865e03,
                1.950e03,
                2.000e03,
            ]
        )
        transmission = np.array(
            [
                0.45,
                0.58,
                0.68,
                0.85,
                0.69,
                0.7,
                0.74,
                0.75,
                0.75,
                0.78,
                0.8,
                0.81,
                0.82,
                0.83,
                0.84,
                0.83,
                0.8,
                0.83,
                0.84,
                0.83,
                0.8,
                0.8,
                0.81,
                0.82,
                0.83,
            ]
        )

        super().__init__(
            orientation=orientation,
            wavelength_nm=wavelength,
            contrast_ratio=contrast,
            transmission=transmission,
        )

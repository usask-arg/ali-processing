from __future__ import annotations

import numpy as np
from skcomponents.optics import Filter, LiquidCrystalRotator


class ArcOptixLCR(LiquidCrystalRotator):

    def __init__(
        self,
        twist_angle: float = 0.0,
        phase: float = 180.0,
        thickness: float = 1,
        reference_wavelength: float = 850.0,
    ):

        t = [
            0.82,
            0.85,
            0.87,
            0.885,
            0.9,
            0.905,
            0.91,
            0.905,
            0.9,
            0.885,
            0.87,
            0.845,
            0.82,
            0.8,
            0.78,
            0.765,
            0.75,
            0.725,
            0.7,
            0.66,
            0.62,
            0.62,
            0.62,
            0.585,
            0.55,
        ]
        w = np.arange(400, 1650, 50)

        self._filter = Filter(w, t)
        super().__init__(
            twist_angle, phase, reference_wavelength=reference_wavelength * thickness
        )

    def matrix(self, wavelength) -> np.array:

        return super().matrix(wavelength) @ self._filter.matrix(wavelength)

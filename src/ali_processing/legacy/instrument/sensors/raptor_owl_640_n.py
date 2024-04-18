from __future__ import annotations

import numpy as np
from skcomponents.optics import Filter

from ali_processing.legacy.instrument.sensors.cardinal_640 import Cardinal640


class RaptorOWL640N(Cardinal640):

    def __init__(self):
        """

        InGaAs SWIR camera with good sensitivity between 400 and 1600 nm.
        https://www.raptorphotonics.com/products/owl-640-ii/

        Notes
        -----
        .. list-table::
           :widths: 100 100
           :header-rows: 0

           * - ADC bits
             - 14
           * - Rows
             - 512
           * - Columns
             - 640
           * - Readout noise
             - 180
           * - Well depth
             - 650,000
           * - Dark current @273K
             - 2000 e/s
        """
        super().__init__()
        self._adc_bits = 14
        epa = 6.241509074e18
        self.dark_current_temps = np.array([273.15 + 15.0, 273.15 - 15.0])
        self.dark_current_values = np.array([28000 / epa, 2000 / epa])
        self.num_rows = 512
        self.num_columns = 640
        self.processors = self._create_post_processors()

    def quantum_efficiency(self) -> Filter:

        wavel = np.array(
            [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]
        )
        transmission = np.array(
            [0.4, 0.4, 0.42, 0.45, 0.55, 0.75, 0.85, 0.9, 0.92, 0.92, 0.9, 0.87, 0.8]
        )
        return Filter(wavelength_nm=wavel, transmission=transmission)

    @property
    def gain(self):
        if self._gain == 0:
            return "low"
        elif self._gain == 1:
            return "high"
        else:
            raise ValueError("Unrecognized gain mode")

    @gain.setter
    def gain(self, value):
        if type(value) is str:
            if value.lower() == "low":
                self._gain = 0
            elif value.lower() == "high":
                self._gain = 1
            else:
                raise ValueError("Unrecognized gain mode")
        else:
            self._gain = int(value)

        if self._gain == 0:
            self._ccd_readout_noise = 180
            self._max_well_depth = 650_000
        elif self._gain == 1:
            self._ccd_readout_noise = 18
            self._max_well_depth = 10_000

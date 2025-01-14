from __future__ import annotations

import numpy as np
from ali_processing.legacy.instrument.sensors.base_sensor import Sensor
from skcomponents.optics import Filter


class Cardinal640(Sensor):

    def __init__(self):
        super().__init__(
            exposure_time=1,
            ccd_temperature=275,
            ccd_readout_noise=350,
            max_well_depth=1_000_000,
            adc_bits=13,
        )
        self._gain = 0
        self.gain = 0
        self.dark_current_temps = np.array([322.0, 298.0, 276.0])
        self.dark_current_values = np.array([10e-15, 1e-15, 0.1e-15]) * 4
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
        elif self._gain == 2:
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
            self._max_well_depth = 600_000
        elif self._gain == 1:
            self._ccd_readout_noise = 35
            self._max_well_depth = 12_000

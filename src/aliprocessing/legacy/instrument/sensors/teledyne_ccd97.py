from __future__ import annotations

import numpy as np
from ali_processing.legacy.instrument.sensors.base_sensor import Sensor
from skcomponents.optics import Filter


class TeledyneCCD97(Sensor):

    def __init__(self):
        super().__init__(
            exposure_time=1,
            ccd_temperature=273,
            ccd_readout_noise=2.2,
            max_well_depth=130_000,
            adc_bits=14,
        )

        epa = 6.241509074e18
        self.dark_current_temps = np.array([298.0, 273.0, 253.0])
        self.dark_current_values = np.array([400 / epa, 33 / epa, 2 / epa])
        self.num_rows = 512
        self.num_columns = 640

    def quantum_efficiency(self) -> Filter:

        wavel = np.array([350.0, 450.0, 550.0, 650.0, 750.0, 850.0, 950.0, 1050.0])
        transmission = np.array([0.2, 0.7, 0.95, 0.9, 0.8, 0.5, 0.2, 0.03])
        return Filter(wavelength_nm=wavel, transmission=transmission)

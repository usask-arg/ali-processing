from __future__ import annotations

import numpy as np
from ali_processing.legacy.instrument.sensors.cardinal_640 import Cardinal640


class Cardinal1280(Cardinal640):

    def __init__(self):
        super().__init__()
        self.dark_current_temps = np.array([322.0, 298.0, 276.0])
        self.dark_current_values = np.array([10e-15, 1e-15, 0.1e-15])
        self.num_rows = 1024
        self.num_columns = 1280

    @property
    def gain(self):
        if self._gain == 0:
            return "low"
        elif self._gain == 1:
            return "medium"
        elif self._gain == 2:
            return "high"
        else:
            raise ValueError("Unrecognized gain mode")

    @gain.setter
    def gain(self, value):
        if type(value) is str:
            if value.lower() == "low":
                self._gain = 0
            elif value.lower() == "medium":
                self._gain = 1
            elif value.lower() == "high":
                self._gain = 2
            else:
                raise ValueError("Unrecognized gain mode")
        else:
            self._gain = int(value)

        if self._gain == 0:
            self._ccd_readout_noise = 350
            self._max_well_depth = 1_000_000
        elif self._gain == 1:
            self._ccd_readout_noise = 170
            self._max_well_depth = 500_000
        elif self._gain == 2:
            self._ccd_readout_noise = 35
            self._max_well_depth = 10_000

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from skcomponents.optics import AOTF, Filter


class Brimrose20mmAOTF(AOTF):

    def __init__(self):
        self._wavelength_nm = np.array(
            [
                400,
                450,
                500,
                550,
                600,
                650,
                700,
                750,
                800,
                850,
                900,
                950,
                1000,
                1050,
                1100,
                1150,
                1200,
                1250,
                1300,
                1350,
                1400,
                1450,
                1500,
                1550,
                1600,
            ]
        )
        self._transmission = np.ones_like(self._wavelength_nm) * 0.95
        self._diffraction_efficiency = np.array(
            [
                0.01,
                0.025,
                0.05,
                0.1,
                0.55,
                0.6,
                0.55,
                0.45,
                0.15,
                0.15,
                0.5,
                0.55,
                0.57,
                0.59,
                0.6,
                0.59,
                0.57,
                0.55,
                0.53,
                0.45,
                0.3,
                0.25,
                0.23,
                0.2,
                0.2,
            ]
        )

        self.t_filter = Filter(self._wavelength_nm, self._transmission)
        self.de_filter = Filter(self._wavelength_nm, self._diffraction_efficiency)
        super().__init__()

    def diffraction_efficiency(self, wavelength):
        return self.de_filter.transmission(wavelength)

    def transmission(self, wavelength):
        return self.t_filter.transmission(wavelength)

    def matrix(self, wavelength):
        return (
            self.transmission(wavelength)
            * self.diffraction_efficiency(wavelength)
            * super().matrix(wavelength)
        )


class BrimroseAOTFSingleChannel(AOTF):
    """
    Single transducer 10mm AOTF. Measurements of diffraction efficiency provided by Daniel Letros.

    Parameters
    ----------
    central_wavelength: float
        Central wavelength of the AOTF. This can be used to shift the central wavelength of the AOTF
        for testing of other systems.
    """

    def __init__(self, central_wavelength: float | None = None):
        self._wavelength_nm = np.array(
            [
                604.134,
                610.689,
                617.315,
                623.834,
                630.704,
                638.095,
                645.425,
                653.14,
                660.726,
                669.111,
                677.337,
                685.974,
                694.862,
                704.063,
                713.827,
                723.755,
                733.933,
                744.383,
                749.882,
                755.448,
                760.972,
                766.716,
                772.454,
                778.401,
                784.511,
                790.565,
                796.783,
                803.187,
                809.791,
                816.38,
                823.15,
                830.135,
                837.096,
                844.435,
                851.761,
                859.312,
                866.918,
                874.918,
                882.879,
                891.028,
                899.307,
                907.874,
                916.405,
                925.507,
                934.483,
                943.777,
                953.373,
                962.73,
                971.497,
                982.403,
                993.024,
                1004.049,
                1015.014,
                1026.371,
                1036.349,
                1048.49,
                1060.558,
                1073.229,
                1085.815,
                1097.21,
                1110.794,
                1124.954,
                1139.006,
            ]
        )
        if central_wavelength:
            self._wavelength_nm = (self._wavelength_nm - 850) * (
                central_wavelength / 850
            ) + central_wavelength
        self._diffraction_efficiency = np.array(
            [
                0.09744021,
                0.09668691,
                0.1040007,
                0.10888101,
                0.11901388,
                0.11739969,
                0.10871549,
                0.10900268,
                0.13063443,
                0.16185204,
                0.21979453,
                0.3084155,
                0.4256343,
                0.56445037,
                0.68453913,
                0.76835277,
                0.80200238,
                0.80920881,
                0.81198999,
                0.81382407,
                0.80710356,
                0.79618183,
                0.77957955,
                0.75021718,
                0.70769231,
                0.65539567,
                0.60638723,
                0.56201219,
                0.53086957,
                0.49934291,
                0.48135259,
                0.47540372,
                0.47659491,
                0.48845925,
                0.50517979,
                0.51745201,
                0.53213403,
                0.53948511,
                0.56141414,
                0.58038416,
                0.5837588,
                0.58305206,
                0.57454584,
                0.5599991,
                0.54474211,
                0.52075203,
                0.50040079,
                0.4856917,
                0.53113712,
                0.52898961,
                0.5159129,
                0.49545287,
                0.46390468,
                0.42465971,
                0.36891265,
                0.3227579,
                0.26577954,
                0.20847833,
                0.15788345,
                0.11808555,
                0.09157119,
                0.07217465,
                0.05689757,
            ]
        )
        self._transmission = np.ones_like(self._wavelength_nm) * 0.95

        self.t_filter = Filter(self._wavelength_nm, self._transmission)
        self.de_filter = Filter(self._wavelength_nm, self._diffraction_efficiency)
        super().__init__()

    def diffraction_efficiency(self, wavelength):
        return self.de_filter.transmission(wavelength)

    def transmission(self, wavelength):
        return self.t_filter.transmission(wavelength)

    def matrix(self, wavelength):
        return (
            self.transmission(wavelength)
            * self.diffraction_efficiency(wavelength)
            * super().matrix(wavelength)
        )


class ER2AOTF(Brimrose20mmAOTF):

    def __init__(self):

        super().__init__()

        filename = Path(__file__).resolve().parent / "data" / "ER2_aotf_calibration.nc"

        data = xr.open_dataset(filename)
        self._diffraction_efficiency = data.de_values.to_numpy() / 100
        self._wavelength_nm = data.de_wavelengths.to_numpy()
        self._transmission = np.ones_like(self._wavelength_nm)

        self.t_filter = Filter(self._wavelength_nm, self._transmission)
        self.de_filter = Filter(self._wavelength_nm, self._diffraction_efficiency)


if __name__ == "__main__":

    aotf = BrimroseAOTFSingleChannel()

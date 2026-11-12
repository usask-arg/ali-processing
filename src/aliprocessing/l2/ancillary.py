from __future__ import annotations

import numpy as np
import sasktran2 as sk
import xarray as xr
from sasktran2.atmosphere import Atmosphere
from sasktran2.constituent.base import Constituent
from skretrieval.retrieval.ancillary import GenericAncillary

from aliprocessing.instrument import band_average

SOLAR_SPECTRUM = "solar/solar_irradiance_hsrs_2022_11_30_extended.nc"


class ChannelSolarIrradiance(Constituent):
    """Solar irradiance band-averaged through each channel's spectral response.

    L1B radiance is the photon-weighted band average of the spectral radiance
    through each channel's SRF. Where the atmosphere varies slowly across a
    passband, that equals the radiance for unit irradiance at the channel centre
    times the solar irradiance averaged the same way, which is what this gives at
    each channel's centre wavelength. The model wavelengths must be the channel
    centres, as they are with a delta-function line shape.

    Parameters
    ----------
    cal_db
        Calibration database, for the spectral throughput and centre wavelengths.
    solar_spectrum
        High-resolution solar irradiance with "wavelength" (nm) and "irradiance"
        (W m-2 nm-1). Defaults to the TSIS-1 HSRS spectrum used by sasktran2.
    tolerance_nm
        Largest difference between a model wavelength and a channel centre.
    """

    def __init__(
        self,
        cal_db: xr.Dataset,
        solar_spectrum: xr.Dataset | None = None,
        tolerance_nm: float = 0.01,
    ) -> None:
        needed = ("center_wavelength", "spectral_throughput", "throughput_wavelength")
        missing = [name for name in needed if name not in cal_db.variables]
        if missing:
            msg = (
                f"the calibration database has no {', '.join(missing)}; L2 needs a "
                "calibration database that follows the cal_db schema"
            )
            raise ValueError(msg)
        if solar_spectrum is None:
            path = sk.database.StandardDatabase().path(SOLAR_SPECTRUM)
            with xr.open_dataset(path) as ds:
                solar_spectrum = ds.load()
        self._center_wavelength = cal_db["center_wavelength"].to_numpy()
        self._irradiance = band_average(
            cal_db,
            solar_spectrum["wavelength"].to_numpy(),
            solar_spectrum["irradiance"].to_numpy(),
        )
        self._tolerance_nm = tolerance_nm

    @property
    def irradiance(self) -> np.ndarray:
        """Band-averaged solar irradiance of each channel, W m-2 nm-1."""
        return self._irradiance

    def add_to_atmosphere(self, atmo: Atmosphere):
        wavelengths = np.asarray(atmo.wavelengths_nm, dtype=float)
        distance = np.abs(wavelengths[:, np.newaxis] - self._center_wavelength)
        channel = np.argmin(distance, axis=1)
        unmatched = distance[np.arange(len(wavelengths)), channel] > self._tolerance_nm
        if np.any(unmatched):
            msg = (
                f"model wavelengths {wavelengths[unmatched]} are not channel centres "
                f"{self._center_wavelength}"
            )
            raise ValueError(msg)
        atmo.storage.solar_irradiance[:] = self._irradiance[channel]

    def register_derivative(self, atmo, name):
        pass


class Ancillary(GenericAncillary):
    """Pressure and temperature, plus the solar irradiance for the cal_db channels."""

    def __init__(self, *args, cal_db: xr.Dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self._solar_irradiance = ChannelSolarIrradiance(cal_db)

    def add_to_atmosphere(self, atmo):
        super().add_to_atmosphere(atmo)

        atmo["solar_irradiance"] = self._solar_irradiance

"""Solar irradiance for the ALI channels, shared by L2 and the simulator.

L1B radiance is the photon-weighted band average of the spectral radiance
through each channel's spectral response function (SRF). Where the atmosphere
varies slowly across a passband, that equals the radiance at the channel centre
for unit irradiance times the solar irradiance averaged the same way.
``ChannelSolarIrradiance`` gives that irradiance at each channel centre, so a
sasktran2 calculation at the channel centres with it as the solar constituent
models L1B radiance. The L2 retrieval uses it, and a simulator can use it for a
front-end radiance at the channel centres that is consistent with L2:

    atmosphere = sk.Atmosphere(
        geometry, config, wavelengths_nm=cal_db["center_wavelength"].to_numpy()
    )
    atmosphere["solar_irradiance"] = ChannelSolarIrradiance(cal_db)
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from sasktran2.atmosphere import Atmosphere
from sasktran2.constituent.base import Constituent
from sasktran2.solar import SolarModel

from aliprocessing.instrument import band_average, channel_at_wavelength

# Box-average width and grid step for the solar spectrum. Box averages, unlike
# samples, keep the Fraunhofer lines' contribution at any step; 0.1 nm agrees
# with integrating the native TSIS-1 HSRS spectrum to about 1e-5.
SOLAR_STEP_NM = 0.1


class ChannelSolarIrradiance(Constituent):
    """Solar irradiance band-averaged through each channel's SRF, at its centre.

    Parameters
    ----------
    cal_db
        Calibration database, for the spectral throughput and centre wavelengths.
    solar_model
        Solar spectrum to average, in ``"average"`` mode. Defaults to
        sasktran2's default spectrum (TSIS-1 HSRS) averaged over
        ``SOLAR_STEP_NM``.
    tolerance_nm
        Largest difference between a model wavelength and a channel centre.
    """

    def __init__(
        self,
        cal_db: xr.Dataset,
        solar_model: SolarModel | None = None,
        tolerance_nm: float = 0.01,
    ) -> None:
        needed = ("center_wavelength", "spectral_throughput", "throughput_wavelength")
        missing = [name for name in needed if name not in cal_db.variables]
        if missing:
            msg = (
                f"the calibration database has no {', '.join(missing)}; it needs to "
                "follow the cal_db schema"
            )
            raise ValueError(msg)
        if solar_model is None:
            solar_model = SolarModel(mode="average", resolution=SOLAR_STEP_NM)

        throughput_wavelength = cal_db["throughput_wavelength"].to_numpy()
        low, high = throughput_wavelength[0], throughput_wavelength[-1]
        num = int(np.ceil((high - low) / SOLAR_STEP_NM)) + 1
        wavelength = np.linspace(low, high, num)

        self._cal_db = cal_db[["center_wavelength"]]
        self._irradiance = band_average(
            cal_db, wavelength, solar_model.irradiance(wavelength)
        )
        self._tolerance_nm = tolerance_nm

    @property
    def irradiance(self) -> np.ndarray:
        """Band-averaged solar irradiance of each channel, W m-2 nm-1."""
        return self._irradiance

    def add_to_atmosphere(self, atmo: Atmosphere):
        channel = channel_at_wavelength(
            self._cal_db, atmo.wavelengths_nm, self._tolerance_nm
        )
        atmo.storage.solar_irradiance[:] = self._irradiance[channel]

    def register_derivative(self, atmo, name):
        pass

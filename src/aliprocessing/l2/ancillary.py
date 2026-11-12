from __future__ import annotations

import xarray as xr
from skretrieval.retrieval.ancillary import GenericAncillary

from aliprocessing.solar import ChannelSolarIrradiance


class Ancillary(GenericAncillary):
    """Pressure and temperature, plus the solar irradiance for the cal_db channels."""

    def __init__(self, *args, cal_db: xr.Dataset, **kwargs):
        super().__init__(*args, **kwargs)
        self._solar_irradiance = ChannelSolarIrradiance(cal_db)

    def add_to_atmosphere(self, atmo):
        super().add_to_atmosphere(atmo)

        atmo["solar_irradiance"] = self._solar_irradiance

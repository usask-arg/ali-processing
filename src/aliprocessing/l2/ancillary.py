from __future__ import annotations

import sasktran2 as sk
from skretrieval.retrieval.ancillary import GenericAncillary


class Ancillary(GenericAncillary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_to_atmosphere(self, atmo):
        super().add_to_atmosphere(atmo)

        atmo["solar_irradiance"] = sk.constituent.SolarIrradiance(
            mode="average", resolution=1
        )

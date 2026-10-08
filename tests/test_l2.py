from __future__ import annotations

import numpy as np
import sasktran2 as sk

from aliprocessing.l2.ancillary import Ancillary
from aliprocessing.l2.optical import aerosol_median_radius_db


def test_ancillary_adds_rayleigh_and_solar_irradiance():
    altitudes = np.arange(0.0, 65001.0, 1000.0)
    geometry = sk.Geometry1D(
        0.6,
        0.0,
        6371000.0,
        altitudes,
        sk.InterpolationMethod.LinearInterpolation,
        sk.GeometryType.Spherical,
    )
    atmo = sk.Atmosphere(geometry, sk.Config(), wavelengths_nm=np.array([745.0]))
    anc = Ancillary(
        altitudes,
        101325.0 * np.exp(-altitudes / 7000.0),
        np.full(len(altitudes), 250.0),
    )

    anc.add_to_atmosphere(atmo)

    assert isinstance(atmo["rayleigh"], sk.constituent.Rayleigh)
    assert isinstance(atmo["solar_irradiance"], sk.constituent.SolarIrradiance)


def test_aerosol_median_radius_db_clamps_single_scatter_albedo():
    db = aerosol_median_radius_db()

    ssa = db._database["xs_scattering"] / db._database["xs_total"]

    assert float(ssa.max()) < 1.0
    assert float(ssa.min()) > 0.0
    np.testing.assert_array_equal(
        db._database["wavelength_nm"],
        [470, 525, 745, 1020, 1230, 1450, 1500, 1550, 1600, 1650],
    )
    np.testing.assert_array_equal(
        db._database["median_radius"], np.arange(10, 600, 10.0)
    )

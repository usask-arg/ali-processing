"""Plumbing test of the full chain: FER -> instrument -> L1A -> L1B -> L2.

A sasktran2 front-end radiance (FER) for a known atmosphere drives the simple
instrument forward model; the L1A goes through L1A->L1B and the L2 retrieval,
which should give back the atmosphere. The FER uses the same physics as the L2
forward model and is monochromatic at each channel's centre, so this checks the
data flow, units and conventions between the stages, not realism. Realistic
end-to-end simulations belong in the simulator package.
"""

from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr
from conftest import SMALL_INSTRUMENT_KWARGS, SPACECRAFT

from aliprocessing.cal_db import SimpleInstrument, simulated_cal_db
from aliprocessing.forward_model import (
    exposure_for_fill,
    observation_sequence,
    simulate_l1a,
)
from aliprocessing.geometry import (
    InstrumentFrame,
    boresight_elevation_for_tangent_altitude,
    sun_direction,
    tangent_geometry,
)
from aliprocessing.l1b.data import L1bImage
from aliprocessing.l2.optical import aerosol_median_radius_db
from aliprocessing.processing.l1a_to_l1b import process_l1a_to_l1b
from aliprocessing.processing.l1b_to_l2 import process_l1b_to_l2_image

START = "2026-06-21T16:00"
ATMOSPHERE_ALTITUDES = np.arange(0.0, 65001.0, 1000.0)
FER_TANGENT_ALTITUDES = np.arange(-3e3, 56001.0, 500.0)

# Truth: a Gaussian aerosol layer, unlike the shape of L2's prior, with L2's
# prior median radius
TRUE_EXTINCTION = 1e-8 + 2e-6 * np.exp(
    -0.5 * ((ATMOSPHERE_ALTITUDES - 20e3) / 4e3) ** 2
)
TRUE_MEDIAN_RADIUS = 80.0

# Finer than the shared small instrument: L2 models radiance at points, so the
# L1B vertical smoothing (the resampling kernel) has to be small for the
# retrieval to return the truth
CHAIN_INSTRUMENT = SimpleInstrument(
    **{
        **SMALL_INSTRUMENT_KWARGS,
        "rows_per_subimage": 160,
        "gap_rows": 8,
        "altitude_grid": (0.0, 50e3, 500.0),
    }
)


def _front_end_radiance(cal_db, solar_zenith_angle, relative_azimuth):
    """Band-centre Stokes radiance against tangent altitude, as a scene."""
    config = sk.Config()
    config.num_stokes = 3
    config.stokes_basis = sk.StokesBasis.Observer
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 8
    config.init_successive_orders_with_discrete_ordinates = True
    config.num_successive_orders_iterations = 1
    config.num_sza = 1

    cos_sza = np.cos(np.deg2rad(solar_zenith_angle))
    geometry = sk.Geometry1D(
        cos_sza=cos_sza,
        solar_azimuth=0.0,
        earth_radius_m=6372000.0,
        altitude_grid_m=ATMOSPHERE_ALTITUDES,
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )
    viewing = sk.ViewingGeometry()
    for altitude in FER_TANGENT_ALTITUDES:
        viewing.add_ray(
            sk.TangentAltitudeSolar(
                altitude,
                np.deg2rad(relative_azimuth),
                SPACECRAFT["spacecraft_altitude"],
                cos_sza,
            )
        )

    atmosphere = sk.Atmosphere(
        geometry, config, wavelengths_nm=cal_db["center_wavelength"].to_numpy()
    )
    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)
    atmosphere["rayleigh"] = sk.constituent.Rayleigh()
    atmosphere["o3"] = sk.climatology.mipas.constituent("O3", sk.optical.O3DBM())
    atmosphere["aerosol"] = sk.constituent.ExtinctionScatterer(
        aerosol_median_radius_db(),
        ATMOSPHERE_ALTITUDES,
        TRUE_EXTINCTION,
        745.0,
        "extend",
        median_radius=np.full(len(ATMOSPHERE_ALTITUDES), TRUE_MEDIAN_RADIUS),
    )
    atmosphere["surface"] = sk.constituent.LambertianSurface(0.3)
    atmosphere["solar_irradiance"] = sk.constituent.SolarIrradiance(
        mode="average", resolution=1
    )
    radiance = (
        sk.Engine(config, geometry, viewing)
        .calculate_radiance(atmosphere)["radiance"]
        .transpose("wavelength", "los", "stokes")
        .to_numpy()
    )

    dims = ["spectral_channel", "altitude"]
    scene = xr.Dataset(
        {name: (dims, radiance[..., i]) for i, name in enumerate("IQU")},
        coords={"altitude": FER_TANGENT_ALTITUDES},
    )
    pressure_temperature = xr.Dataset(
        {
            "pressure": ("altitude", atmosphere.pressure_pa),
            "temperature": ("altitude", atmosphere.temperature_k),
        },
        coords={"altitude": ATMOSPHERE_ALTITUDES},
    )
    return scene, pressure_temperature


@pytest.mark.slow
def test_chain_returns_the_atmosphere():
    cal_db = simulated_cal_db(CHAIN_INSTRUMENT)
    elevation = boresight_elevation_for_tangent_altitude(
        SPACECRAFT["spacecraft_latitude"],
        SPACECRAFT["spacecraft_longitude"],
        SPACECRAFT["spacecraft_altitude"],
        SPACECRAFT["boresight_azimuth"],
        CHAIN_INSTRUMENT.boresight_tangent_altitude,
    )

    # FER at the boresight geometry; the across-track variation is ignored
    frame = InstrumentFrame.from_attitude(
        SPACECRAFT["spacecraft_latitude"],
        SPACECRAFT["spacecraft_longitude"],
        SPACECRAFT["spacecraft_altitude"],
        SPACECRAFT["boresight_azimuth"],
        elevation,
        0.0,
    )
    boresight = tangent_geometry(
        frame.observer, frame.boresight[np.newaxis], sun_direction(np.datetime64(START))
    )
    scene, pressure_temperature = _front_end_radiance(
        cal_db,
        float(boresight.solar_zenith_angle[0]),
        float(boresight.relative_solar_azimuth_angle[0]),
    )

    frames = observation_sequence(
        cal_db,
        start=START,
        num_sets=1,
        exposure_time=exposure_for_fill(cal_db, scene),
        boresight_elevation=elevation,
        **SPACECRAFT,
    )
    l1b = process_l1a_to_l1b(simulate_l1a(cal_db, scene, frames, noise=False), cal_db)

    # The bin next to the boresight
    image = L1bImage.from_l1b(l1b, 0, 1, low_alt=8e3, high_alt=46e3)
    state = process_l1b_to_l2_image(
        image, pressure_temperature, None, minimizer_kwargs={"verbose": 0}
    )

    altitude_dim = state["stratospheric_aerosol_extinction_per_m"].dims[0]
    altitudes = {altitude_dim: [15e3, 20e3, 25e3, 30e3]}
    extinction = state["stratospheric_aerosol_extinction_per_m"].interp(altitudes)
    error = state["stratospheric_aerosol_extinction_per_m_1sigma_error"].interp(
        altitudes
    )
    truth = np.interp(altitudes[altitude_dim], ATMOSPHERE_ALTITUDES, TRUE_EXTINCTION)

    np.testing.assert_allclose(extinction, truth, rtol=0.1)
    assert np.all(np.abs(extinction - truth) < 3 * error)
    median_radius = float(
        state["stratospheric_aerosol_median_radius"].interp({altitude_dim: 20e3})
    )
    assert median_radius == pytest.approx(TRUE_MEDIAN_RADIUS, rel=0.2)

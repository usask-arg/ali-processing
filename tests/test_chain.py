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
    scene_from_radiance,
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
from aliprocessing.solar import ChannelSolarIrradiance

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


def _band_averaged_solar_irradiance(cal_db):
    """The TSIS-1 HSRS solar irradiance averaged through each channel's SRF.

    Written out here from the native spectrum rather than calling the library,
    as an independent check of ChannelSolarIrradiance.
    """
    path = sk.database.StandardDatabase().path(
        "solar/solar_irradiance_hsrs_2022_11_30_extended.nc"
    )
    with xr.open_dataset(path) as solar:
        wavelength = solar["wavelength"].to_numpy()
        irradiance = solar["irradiance"].to_numpy()
    throughput_wavelength = cal_db["throughput_wavelength"].to_numpy()
    band = (wavelength >= throughput_wavelength[0]) & (
        wavelength <= throughput_wavelength[-1]
    )
    wavelength, irradiance = wavelength[band], irradiance[band]

    averages = []
    for throughput in cal_db["spectral_throughput"].to_numpy():
        photon_weight = (
            np.interp(wavelength, throughput_wavelength, throughput) * wavelength
        )
        averages.append(
            np.trapezoid(photon_weight * irradiance, wavelength)
            / np.trapezoid(photon_weight, wavelength)
        )
    return np.array(averages)


def _water_cloud(cal_db, extinction_per_m):
    """A liquid water cloud, 5 um droplets, between 9 and 12 km."""
    droplets = sk.database.MieDatabase(
        sk.mie.distribution.LogNormalDistribution().freeze(mode_width=1.5),
        sk.mie.refractive.Water(),
        cal_db["center_wavelength"].to_numpy(),
        median_radius=np.array([5000.0]),
    )
    in_layer = (ATMOSPHERE_ALTITUDES >= 9e3) & (ATMOSPHERE_ALTITUDES <= 12e3)
    return sk.constituent.ExtinctionScatterer(
        droplets,
        ATMOSPHERE_ALTITUDES,
        np.where(in_layer, extinction_per_m, 0.0),
        745.0,
        "zero",
        median_radius=np.full(len(ATMOSPHERE_ALTITUDES), 5000.0),
    )


def _boresight_sun(boresight):
    """One geometry for every line of sight: the boresight's tangent point."""
    zenith = float(boresight.solar_zenith_angle[0])
    return {
        "solar_zenith_angle": zenith,
        "relative_solar_azimuth_angle": float(
            boresight.relative_solar_azimuth_angle[0]
        ),
        "reference_cos_sza": np.cos(np.deg2rad(zenith)),
        "earth_radius_m": 6372000.0,
    }


def _l2_sun(image):
    """The geometry L2 uses for an image, on the FER's tangent altitudes.

    Each line of sight has its own solar angles, and the multiple-scattering
    source and Earth radius come from the image's reference point, as in
    skretrieval's model geometry. An across-track bin off the boresight sees
    a slightly different sun from the boresight; below 10 km, where the limb is
    bright and the noise small, a FER at the boresight's sun misfits by many
    standard deviations.
    """
    ds = image.spectra["I"].ds
    altitude = ds["tangent_altitude"].to_numpy()
    reference = sk.WGS84()
    reference.from_lat_lon_alt(
        float(image.reference_latitude()["I"]),
        float(image.reference_longitude()["I"]),
        0.0,
    )
    return {
        "solar_zenith_angle": np.interp(
            FER_TANGENT_ALTITUDES, altitude, ds["solar_zenith_angle"].to_numpy()
        ),
        "relative_solar_azimuth_angle": np.interp(
            FER_TANGENT_ALTITUDES,
            altitude,
            ds["relative_solar_azimuth_angle"].to_numpy(),
        ),
        "reference_cos_sza": float(image.reference_cos_sza()["I"]),
        "earth_radius_m": float(np.linalg.norm(reference.location)),
    }


def _front_end_radiance(cal_db, sun, cloud_extinction=0.0):
    """Band-centre Stokes radiance against tangent altitude, as a scene.

    ``sun`` is the viewing and solar geometry, from ``_boresight_sun`` or
    ``_l2_sun``.
    """
    config = sk.Config()
    config.num_stokes = 3
    config.stokes_basis = sk.StokesBasis.Observer
    config.multiple_scatter_source = sk.MultipleScatterSource.DiscreteOrdinates
    config.num_streams = 8
    config.init_successive_orders_with_discrete_ordinates = True
    config.num_successive_orders_iterations = 1
    config.num_sza = 1

    geometry = sk.Geometry1D(
        cos_sza=sun["reference_cos_sza"],
        solar_azimuth=0.0,
        earth_radius_m=sun["earth_radius_m"],
        altitude_grid_m=ATMOSPHERE_ALTITUDES,
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )
    viewing = sk.ViewingGeometry()
    solar_zenith = np.broadcast_to(
        sun["solar_zenith_angle"], FER_TANGENT_ALTITUDES.shape
    )
    relative_azimuth = np.broadcast_to(
        sun["relative_solar_azimuth_angle"], FER_TANGENT_ALTITUDES.shape
    )
    for altitude, zenith, azimuth in zip(
        FER_TANGENT_ALTITUDES, solar_zenith, relative_azimuth, strict=True
    ):
        viewing.add_ray(
            sk.TangentAltitudeSolar(
                altitude,
                np.deg2rad(azimuth),
                SPACECRAFT["spacecraft_altitude"],
                np.cos(np.deg2rad(zenith)),
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
    if cloud_extinction > 0:
        atmosphere["cloud"] = _water_cloud(cal_db, cloud_extinction)
    atmosphere["surface"] = sk.constituent.LambertianSurface(0.3)
    # The solar constituent L2 uses, as a simulator would at the channel centres
    atmosphere["solar_irradiance"] = ChannelSolarIrradiance(cal_db)
    scene = scene_from_radiance(
        cal_db,
        sk.Engine(config, geometry, viewing).calculate_radiance(atmosphere),
        FER_TANGENT_ALTITUDES,
    )
    pressure_temperature = xr.Dataset(
        {
            "pressure": ("altitude", atmosphere.pressure_pa),
            "temperature": ("altitude", atmosphere.temperature_k),
        },
        coords={"altitude": ATMOSPHERE_ALTITUDES},
    )
    return scene, pressure_temperature


def _run_chain(instrument, cloud_extinction=0.0, tropopause=None, low_alt=4e3):
    """FER -> L1A -> L1B -> L2 for the across-track bin next to the boresight."""
    cal_db = simulated_cal_db(instrument)
    elevation = boresight_elevation_for_tangent_altitude(
        SPACECRAFT["spacecraft_latitude"],
        SPACECRAFT["spacecraft_longitude"],
        SPACECRAFT["spacecraft_altitude"],
        SPACECRAFT["boresight_azimuth"],
        instrument.boresight_tangent_altitude,
    )

    # The FER is one profile against tangent altitude, so it can match the
    # geometry of only one across-track bin. A first FER at the boresight's sun
    # gives an L1B carrying that bin's geometry; the FER used is then computed
    # with exactly the geometry L2 uses for it.
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

    def l1b_image(sun):
        scene, pressure_temperature = _front_end_radiance(cal_db, sun, cloud_extinction)
        frames = observation_sequence(
            cal_db,
            start=START,
            num_sets=1,
            exposure_time=exposure_for_fill(cal_db, scene),
            boresight_elevation=elevation,
            **SPACECRAFT,
        )
        l1b = process_l1a_to_l1b(
            simulate_l1a(cal_db, scene, frames, noise=False), cal_db
        )
        image = L1bImage.from_l1b(l1b, 0, 1, low_alt=low_alt, high_alt=46e3)
        return image, pressure_temperature

    image, _ = l1b_image(_boresight_sun(boresight))
    image, pressure_temperature = l1b_image(_l2_sun(image))
    if tropopause is not None:
        pressure_temperature["tropopause_altitude"] = ((), tropopause)
    return process_l1b_to_l2_image(
        image, pressure_temperature, cal_db, config={"minimizer": {"verbose": 0}}
    )


def _extinction_against_truth(state, altitudes):
    altitude_dim = state["stratospheric_aerosol_extinction_per_m"].dims[0]
    at = {altitude_dim: altitudes}
    extinction = state["stratospheric_aerosol_extinction_per_m"].interp(at)
    error = state["stratospheric_aerosol_extinction_per_m_1sigma_error"].interp(at)
    truth = np.interp(altitudes, ATMOSPHERE_ALTITUDES, TRUE_EXTINCTION)
    return extinction, error, truth


@pytest.mark.slow
def test_chain_returns_the_atmosphere():
    state = _run_chain(CHAIN_INSTRUMENT)

    # Clear sky: no cloud, so stage 2 starts at the bottom of stage 1's window
    assert np.isnan(float(state["cloud_top_altitude"]))
    assert float(state["retrieval_lower_altitude"]) == 5000.0

    extinction, error, truth = _extinction_against_truth(
        state, [15e3, 20e3, 25e3, 30e3]
    )
    np.testing.assert_allclose(extinction, truth, rtol=0.1)
    assert np.all(np.abs(extinction - truth) < 3 * error)
    altitude_dim = state["stratospheric_aerosol_extinction_per_m"].dims[0]
    median_radius = float(
        state["stratospheric_aerosol_median_radius"].interp({altitude_dim: 20e3})
    )
    assert median_radius == pytest.approx(TRUE_MEDIAN_RADIUS, rel=0.2)


# The cloud signatures are strongest between the shortest and longest channels
CLOUD_INSTRUMENT = SimpleInstrument(
    **{
        **SMALL_INSTRUMENT_KWARGS,
        "rows_per_subimage": 160,
        "gap_rows": 8,
        "altitude_grid": (0.0, 50e3, 500.0),
        "center_wavelengths_nm": (525.0, 1560.0),
    }
)


@pytest.mark.slow
def test_chain_retrieves_above_a_cloud():
    # An optically thick water cloud with its top at 12 km, just above an 11 km
    # tropopause
    state = _run_chain(CLOUD_INSTRUMENT, cloud_extinction=5e-5, tropopause=11e3)

    cloud_top = float(state["cloud_top_altitude"])
    assert 12e3 <= cloud_top <= 14e3
    assert state["cloud_colour_mask"].any()
    assert state["cloud_dolp_mask"].any()
    assert float(state["retrieval_lower_altitude"]) == cloud_top + 1000.0

    # Well above the cloud the aerosol is recovered
    extinction, _, truth = _extinction_against_truth(state, [25e3])
    np.testing.assert_allclose(extinction, truth, rtol=0.05)

    # Nearer the cloud, light it reflects upward brightens the limb, which L2's
    # clear-sky model does not include: stage 2 comes out low by a bias that
    # shrinks with height (about -15% at 15 km, -6% at 20 km, 0 at 25 km, many
    # standard deviations, varying across platforms). Stage 2 still improves
    # on stage 1, whose extinction the cloud contaminates throughout.
    altitudes = [15e3, 20e3, 25e3]
    extinction, _, truth = _extinction_against_truth(state, altitudes)
    altitude_dim = state["stratospheric_aerosol_extinction_per_m"].dims[0]
    first_pass = state["stratospheric_aerosol_extinction_per_m_first_pass"].interp(
        {altitude_dim: altitudes}
    )
    assert np.all(np.abs(extinction - truth) < np.abs(first_pass - truth))


def test_channel_solar_irradiance_matches_the_native_spectrum():
    cal_db = simulated_cal_db(CHAIN_INSTRUMENT)
    np.testing.assert_allclose(
        ChannelSolarIrradiance(cal_db).irradiance,
        _band_averaged_solar_irradiance(cal_db),
        rtol=1e-4,
    )

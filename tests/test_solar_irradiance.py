"""Band averaging through the cal_db SRFs, and the solar irradiance L2 uses."""

from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr
from sasktran2.solar import SolarModel

from aliprocessing.instrument import band_average, channel_at_wavelength
from aliprocessing.solar import SOLAR_STEP_NM, ChannelSolarIrradiance

WAVELENGTH = np.arange(600.0, 1200.0, 0.01)


def test_band_average_of_a_constant(small_cal_db):
    np.testing.assert_allclose(
        band_average(small_cal_db, WAVELENGTH, np.full(WAVELENGTH.size, 3.0)), 3.0
    )


def test_band_average_is_photon_weighted(small_cal_db, small_instrument):
    # For a Gaussian passband, the photon (tau * lambda) weighting puts the mean
    # of a linear spectrum at c + sigma^2 / c rather than at the centre c
    sigma = small_instrument.bandpass_fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    centers = np.array(small_instrument.center_wavelengths_nm)
    np.testing.assert_allclose(
        band_average(small_cal_db, WAVELENGTH, WAVELENGTH),
        centers + sigma**2 / centers,
        rtol=1e-6,
    )


def test_band_average_broadcasts_over_leading_dimensions(small_cal_db):
    spectra = np.stack([np.ones(WAVELENGTH.size), 2 * np.ones(WAVELENGTH.size)])
    np.testing.assert_allclose(
        band_average(small_cal_db, WAVELENGTH, spectra), [[1.0, 2.0], [1.0, 2.0]]
    )


def test_band_average_needs_the_whole_passband(small_cal_db):
    short = WAVELENGTH[WAVELENGTH < 1000]
    with pytest.raises(ValueError, match="passbands span"):
        band_average(small_cal_db, short, short)


def test_band_average_ignores_negligible_tails(small_cal_db):
    # A grid on the throughput range whose last point falls a hair short through
    # floating-point steps, as np.arange can give
    throughput_wavelength = small_cal_db["throughput_wavelength"].to_numpy()
    grid = np.arange(throughput_wavelength[0], throughput_wavelength[-1], 0.02)
    assert grid[-1] < throughput_wavelength[-1]
    np.testing.assert_allclose(
        band_average(small_cal_db, grid, np.ones(grid.size)), 1.0
    )


def test_channel_at_wavelength(small_cal_db):
    np.testing.assert_array_equal(
        channel_at_wavelength(small_cal_db, [1020.0, 745.0, 745.005]), [1, 0, 0]
    )
    with pytest.raises(ValueError, match="not channel centres"):
        channel_at_wavelength(small_cal_db, [800.0])


def _atmosphere(wavelengths):
    config = sk.Config()
    geometry = sk.Geometry1D(
        cos_sza=0.5,
        solar_azimuth=0.0,
        earth_radius_m=6372000.0,
        altitude_grid_m=np.arange(0.0, 10001.0, 1000.0),
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )
    return sk.Atmosphere(geometry, config, wavelengths_nm=np.asarray(wavelengths))


def test_solar_irradiance_is_band_averaged_at_each_channel(small_cal_db):
    # A spectrum with a deep line just off one channel's centre: a sample at
    # the centre misses it, the SRF average does not
    irradiance = np.where(np.abs(WAVELENGTH - 748.0) < 0.5, 0.0, 1.0)
    solar = SolarModel(
        ds=xr.Dataset(
            {"irradiance": ("wavelength", irradiance)},
            coords={"wavelength": WAVELENGTH},
        ),
        mode="average",
        resolution=SOLAR_STEP_NM,
    )
    constituent = ChannelSolarIrradiance(small_cal_db, solar_model=solar)
    expected = band_average(small_cal_db, WAVELENGTH, irradiance)
    assert expected[0] < 0.99
    # The line's hard edges are resolved to about the 0.1 nm box width
    np.testing.assert_allclose(constituent.irradiance, expected, rtol=1e-3)

    centers = small_cal_db["center_wavelength"].to_numpy()
    atmosphere = _atmosphere(centers[::-1])
    constituent.add_to_atmosphere(atmosphere)
    np.testing.assert_allclose(
        atmosphere.storage.solar_irradiance, constituent.irradiance[::-1]
    )


def test_default_matches_the_native_solar_spectrum(small_cal_db):
    # Box averages on the SOLAR_STEP_NM grid agree with integrating the native
    # HSRS spectrum, read directly from sasktran2's database
    path = sk.database.StandardDatabase().path(
        "solar/solar_irradiance_hsrs_2022_11_30_extended.nc"
    )
    with xr.open_dataset(path) as solar:
        wavelength = solar["wavelength"].to_numpy()
        irradiance = solar["irradiance"].to_numpy()
    native = band_average(small_cal_db, wavelength, irradiance)
    np.testing.assert_allclose(
        ChannelSolarIrradiance(small_cal_db).irradiance, native, rtol=1e-4
    )


def test_default_differs_from_a_1_nm_average(small_cal_db):
    constituent = ChannelSolarIrradiance(small_cal_db)
    one_nm = SolarModel(mode="average", resolution=1).irradiance(
        small_cal_db["center_wavelength"].to_numpy()
    )
    # Fraunhofer structure within the 10 nm passbands moves the average by
    # tenths of a percent at 745 and 1020 nm
    relative = constituent.irradiance / one_nm - 1
    assert np.all(np.abs(relative) < 0.02)
    assert np.all(np.abs(relative) > 1e-3)


def test_model_wavelengths_must_be_channel_centres(small_cal_db):
    constituent = ChannelSolarIrradiance(small_cal_db)
    with pytest.raises(ValueError, match="not channel centres"):
        constituent.add_to_atmosphere(_atmosphere([745.0, 800.0]))


def test_calibration_database_without_srf_is_rejected():
    # An older calibration database with only sample wavelengths
    old = xr.Dataset(coords={"sample_wavelengths": np.arange(400.0, 800.0)})
    with pytest.raises(ValueError, match="follow the cal_db schema"):
        ChannelSolarIrradiance(old)

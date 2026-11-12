"""Band averaging through the cal_db SRFs, and the solar irradiance L2 uses."""

from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk
import xarray as xr

from aliprocessing.instrument import band_average
from aliprocessing.l2.ancillary import ChannelSolarIrradiance

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
    with pytest.raises(ValueError, match="passbands span"):
        band_average(
            small_cal_db, WAVELENGTH[WAVELENGTH < 1000], WAVELENGTH[WAVELENGTH < 1000]
        )


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
    solar = xr.Dataset(
        {"irradiance": ("wavelength", irradiance)}, coords={"wavelength": WAVELENGTH}
    )
    constituent = ChannelSolarIrradiance(small_cal_db, solar_spectrum=solar)
    expected = band_average(small_cal_db, WAVELENGTH, irradiance)
    assert expected[0] < 0.99

    centers = small_cal_db["center_wavelength"].to_numpy()
    atmosphere = _atmosphere(centers[::-1])
    constituent.add_to_atmosphere(atmosphere)
    np.testing.assert_allclose(atmosphere.storage.solar_irradiance, expected[::-1])


def test_default_spectrum_differs_from_a_1_nm_average(small_cal_db):
    constituent = ChannelSolarIrradiance(small_cal_db)
    one_nm = sk.solar.SolarModel(mode="average", resolution=1).irradiance(
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
    with pytest.raises(ValueError, match="follows the cal_db schema"):
        ChannelSolarIrradiance(old)

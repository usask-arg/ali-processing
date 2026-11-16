"""Mapping the retrieval state onto L2 profiles and granules, without retrieving."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aliprocessing.l2.config import load_retrieval_config
from aliprocessing.processing.l1b_to_l2 import l2_granule, l2_profile
from aliprocessing.schemas import validate

ALTITUDE = np.array([5e3, 10e3, 15e3])
ALBEDO_WAVELENGTHS = np.array([745.0, 1020.0])
EXTINCTION = "stratospheric_aerosol_extinction_per_m"
MEDIAN_RADIUS = "stratospheric_aerosol_median_radius"
START = np.datetime64("2026-11-16T12:00:00", "ns")


def _retrieved(name, values, kernel_scale):
    """A described quantity; the solution error is 3 times the noise error."""
    profile, kernel = ["altitude"], ["altitude", "altitude_2"]
    return {
        name: (profile, values),
        f"{name}_prior": (profile, np.full(3, values[0])),
        f"{name}_1sigma_error": (profile, 0.1 * values),
        f"{name}_1sigma_solution_error": (profile, 0.3 * values),
        f"{name}_averaging_kernel": (kernel, kernel_scale * np.eye(3)),
    }


def _state(
    *,
    observation_set=0,
    status=2,
    two_stage=True,
    median_radius=True,
    albedo_errors=True,
) -> xr.Dataset:
    """A state as process_l1b_to_l2_image returns it."""
    extinction = np.array([1e-6, 2e-6, 3e-6])
    data_vars = {
        **_retrieved(EXTINCTION, extinction, 0.5),
        **_retrieved("o3_vmr", np.full(3, 5e-6), 0.01),
        "lambertian_albedo": ("wavelengths_nm", [0.28, 0.32]),
        "time": ((), START + observation_set * np.timedelta64(3, "s")),
        "latitude": ((), 52.0),
        "longitude": ((), -106.0),
        "solar_zenith_angle": ((), 60.0),
        "solar_azimuth_angle": ((), 30.0),
        "solar_scattering_angle": ((), 64.3),
        "num_iterations": ((), np.int64(9)),
        "cost": ((), 1.2),
        "optimizer_status": ((), np.int64(status)),
        "tropopause_altitude": ((), 11e3),
    }
    if median_radius:
        data_vars |= _retrieved(MEDIAN_RADIUS, np.array([70.0, 80.0, 90.0]), 0.2)
    if albedo_errors:
        data_vars["lambertian_albedo_1sigma_error"] = ("wavelengths_nm", [0.01, 0.03])
        data_vars["lambertian_albedo_1sigma_solution_error"] = (
            "wavelengths_nm",
            [0.02, 0.06],
        )
    if two_stage:
        data_vars |= {
            f"{EXTINCTION}_first_pass": ("altitude", 2 * extinction),
            "cloud_top_altitude": ((), 9e3),
            "retrieval_lower_altitude": ((), 10e3),
            "cloud_mask": ("cloud_altitude", np.array([0, 1, 0], np.int8)),
        }
    return xr.Dataset(
        data_vars,
        coords={
            "altitude": ALTITUDE,
            "altitude_2": ALTITUDE,
            "wavelengths_nm": ALBEDO_WAVELENGTHS,
            "cloud_altitude": [8e3, 9e3, 10e3],
        },
    )


def _granule(config, num_sets=2, num_bins=2, **state_kwargs):
    profiles = [
        [
            l2_profile(_state(observation_set=i, **state_kwargs), config)
            for _ in range(num_bins)
        ]
        for i in range(num_sets)
    ]
    return l2_granule(profiles, np.linspace(-1.0, 1.0, num_bins))


@pytest.fixture
def config():
    return load_retrieval_config()


def test_granule_is_valid(config):
    l2 = _granule(config)
    assert validate(l2, "l2") == []
    assert dict(l2.sizes) == {
        "observation_set": 2,
        "across_track": 2,
        "altitude": 3,
        "altitude2": 3,
        "albedo_wavelength": 2,
    }
    assert l2["aerosol_extinction"].dims == ("observation_set", "across_track", "altitude")
    np.testing.assert_array_equal(
        l2["time"], [START, START + np.timedelta64(3, "s")]
    )
    np.testing.assert_array_equal(l2["across_track_angle"], [-1.0, 1.0])
    assert l2["aerosol_extinction"].attrs["units"] == "m-1"


def test_granule_survives_a_netcdf_round_trip(config, tmp_path):
    path = tmp_path / "l2.nc"
    _granule(config).to_netcdf(path)
    with xr.open_dataset(path) as l2:
        assert validate(l2, "l2") == []
        assert l2["converged"].dtype == bool


def test_noise_and_solution_errors(config):
    l2 = _granule(config).isel(observation_set=0, across_track=0)
    extinction = l2["aerosol_extinction"].to_numpy()
    np.testing.assert_allclose(l2["aerosol_extinction_noise_error"], 0.1 * extinction)
    np.testing.assert_allclose(
        l2["aerosol_extinction_solution_error"], 0.3 * extinction
    )
    np.testing.assert_allclose(l2["surface_albedo_noise_error"], [0.01, 0.03])
    np.testing.assert_allclose(l2["surface_albedo_solution_error"], [0.02, 0.06])
    np.testing.assert_allclose(l2["aerosol_median_radius"], [70.0, 80.0, 90.0])
    np.testing.assert_allclose(l2["aerosol_extinction_averaging_kernel"], 0.5 * np.eye(3))
    np.testing.assert_array_equal(l2["albedo_wavelength"], ALBEDO_WAVELENGTHS)


def test_two_stage_fields(config):
    l2 = _granule(config).isel(observation_set=0, across_track=0)
    assert l2.attrs["retrieval_stages"] == 2
    assert float(l2["cloud_top_altitude"]) == 9e3
    assert float(l2["lower_altitude_limit"]) == 10e3
    assert float(l2["upper_altitude_limit"]) == 40e3
    np.testing.assert_allclose(
        l2["aerosol_extinction_stage1"], 2 * l2["aerosol_extinction"]
    )
    assert float(l2["tropopause_altitude"]) == 11e3


def test_single_stage_uses_the_configured_window():
    config = load_retrieval_config({"two_stage": {"enabled": False}})
    l2 = _granule(config, two_stage=False)
    assert validate(l2, "l2") == []
    assert l2.attrs["retrieval_stages"] == 1
    assert "aerosol_extinction_stage1" not in l2
    assert "cloud_top_altitude" not in l2
    np.testing.assert_array_equal(l2["lower_altitude_limit"], 5000.0)


@pytest.mark.parametrize(("status", "converged"), [(2, True), (0, False), (-1, False)])
def test_converged_follows_the_optimizer_status(config, status, converged):
    l2 = _granule(config, status=status)
    assert np.all(l2["optimizer_status"] == status)
    assert np.all(l2["converged"] == converged)


def test_diagnostics_and_placeholder_flags(config):
    l2 = _granule(config)
    np.testing.assert_array_equal(l2["num_function_evaluations"], 9)
    np.testing.assert_allclose(l2["cost"], 1.2)
    np.testing.assert_allclose(l2["scattering_angle"], 64.3)
    assert np.all(l2["profile_quality_flag"] == 0)
    assert np.all(l2["aerosol_extinction_quality_flag"] == 0)


def test_median_radius_left_out_when_not_retrieved(config):
    del config["state"]["aerosols"]["stratospheric_aerosol"]["retrieved_quantities"][
        "median_radius"
    ]
    l2 = _granule(config, median_radius=False)
    assert validate(l2, "l2") == []
    assert not any(name.startswith("aerosol_median_radius") for name in l2.variables)
    assert "aerosol_median_radius_retrieval_space" not in l2.attrs


def test_disabled_albedo_is_left_out(config):
    l2 = _granule(config, albedo_errors=False)
    assert validate(l2, "l2") == []
    assert "surface_albedo" not in l2
    assert "albedo_wavelength" not in l2.dims


def test_o3_and_cloud_profiles_are_not_stored(config):
    l2 = _granule(config)
    assert not any("o3" in name or "cloud_mask" in name for name in l2.variables)
    assert "cloud_altitude" not in l2.dims


def test_retrieval_spaces(config):
    aerosol = config["state"]["aerosols"]["stratospheric_aerosol"]
    assert l2_profile(_state(), config).attrs["aerosol_extinction_retrieval_space"] == (
        "linear"
    )

    # A top-level log_space applies to quantities without their own, here the
    # median radius
    aerosol["log_space"] = True
    attrs = l2_profile(_state(), config).attrs
    assert attrs["aerosol_extinction_retrieval_space"] == "linear"
    assert attrs["aerosol_median_radius_retrieval_space"] == "log"


def test_profiles_with_different_settings_are_rejected(config):
    single = load_retrieval_config({"two_stage": {"enabled": False}})
    profiles = [[l2_profile(_state(), config), l2_profile(_state(), single)]]
    with pytest.raises(ValueError, match="retrieval settings"):
        l2_granule(profiles, [-1.0, 1.0])


def test_profiles_on_different_altitude_grids_are_rejected(config):
    other = _state().assign_coords(altitude=ALTITUDE + 1.0, altitude_2=ALTITUDE + 1.0)
    profiles = [[l2_profile(_state(), config), l2_profile(other, config)]]
    with pytest.raises(ValueError, match="altitude"):
        l2_granule(profiles, [-1.0, 1.0])


def test_bins_of_one_observation_set_must_share_its_time(config):
    profiles = [
        [
            l2_profile(_state(observation_set=0), config),
            l2_profile(_state(observation_set=1), config),
        ]
    ]
    with pytest.raises(ValueError, match="time"):
        l2_granule(profiles, [-1.0, 1.0])


def test_empty_granule_is_rejected():
    with pytest.raises(ValueError, match="at least one profile"):
        l2_granule([], [])

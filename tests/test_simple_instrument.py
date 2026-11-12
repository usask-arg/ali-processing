"""The placeholder calibration database and the L1A forward model."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from conftest import SMALL_INSTRUMENT_KWARGS

from aliprocessing.cal_db import SimpleInstrument, simulated_cal_db
from aliprocessing.forward_model import (
    exposure_for_fill,
    scene_from_radiance,
    simulate_l1a,
)
from aliprocessing.instrument import (
    PLANCK,
    SPEED_OF_LIGHT,
    detector_variance,
    photon_weighted_throughput,
    pixel_array,
    saturation_counts,
    stokes_sensitivities,
)
from aliprocessing.schemas import validate


# ----------------------------------------------------------------- cal_db
def test_full_size_placeholder_layout():
    instrument = SimpleInstrument()
    cal_db = simulated_cal_db(instrument)
    assert cal_db.sizes["detector_row"] == 1024
    assert cal_db.sizes["detector_column"] == 1280
    assert cal_db.sizes["spectral_channel"] == 7
    analyzer = cal_db["pixel_analyzer"].to_numpy()
    assert [np.count_nonzero(analyzer == k) for k in range(3)] == [320 * 1280] * 3
    # The pixel rows are finer than the 250 m grid, so the kernel is the grid step
    assert instrument.pixel_altitude_spacing < 250
    assert float(cal_db["altitude_kernel_half_width"]) == 250


def test_subimages_share_field_angles(small_cal_db, small_instrument):
    elevation = pixel_array(small_cal_db, "pixel_elevation_angle")
    step = small_instrument.rows_per_subimage + small_instrument.gap_rows
    rows = slice(0, small_instrument.rows_per_subimage)
    np.testing.assert_array_equal(elevation[rows], elevation[step : step + 40])
    # Row 0 of each sub-image looks highest
    assert elevation[0, 0] > elevation[39, 0]


def test_cal_db_is_seeded(small_instrument):
    first = simulated_cal_db(small_instrument, seed=4)
    assert first.identical(simulated_cal_db(small_instrument, seed=4))
    assert not first.identical(simulated_cal_db(small_instrument, seed=5))


# ---------------------------------------------------------- forward model
def test_simulated_l1a_is_valid(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(num_sets=2), seed=1)
    assert validate(l1a, "l1a", cal_db=small_cal_db) == []


def test_blank_frame_is_dark_only(small_cal_db, limb_scene, make_frames):
    frames = make_frames()
    l1a = simulate_l1a(small_cal_db, limb_scene, frames, noise=False)
    blank = l1a["simulation"]["image_noiseless"].isel(time=-1).to_numpy()
    expected = pixel_array(small_cal_db, "dark_offset") + float(
        frames["exposure_time"][-1]
    ) * pixel_array(small_cal_db, "dark_current")
    assert int(frames["spectral_channel"][-1]) == -1
    np.testing.assert_allclose(blank, expected)


def test_brighter_at_lower_tangent_altitude(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(), noise=False)
    image = l1a["image"].isel(time=0).to_numpy()
    # Row 0 of the first sub-image looks highest, row 39 lowest
    assert np.mean(image[39]) > np.mean(image[0])


def test_exposure_fills_to_the_requested_fraction(
    small_cal_db, limb_scene, make_frames
):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(), noise=False)
    gain = float(small_cal_db["detector_gain"])
    signal = l1a["image"].isel(time=slice(0, 2)) - pixel_array(
        small_cal_db, "dark_offset"
    )
    peak_fraction = (
        float(signal.max()) * gain / float(small_cal_db["detector_full_well"])
    )
    # 0.6 of saturation for the brightest scene radiance; the pixels see a bit less
    assert 0.4 < peak_fraction <= 0.6
    assert not l1a["saturated"].any()


def test_overexposure_saturates(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(exposure_time=5.0), seed=1)
    assert l1a["saturated"].isel(time=0).any()
    assert float(l1a["image"].max()) <= saturation_counts(small_cal_db)


def test_noise_matches_the_detector_model(small_cal_db, limb_scene, make_frames):
    # One whole observation set; frame 0 is checked
    frames = make_frames(blank_between_sets=False)
    realizations = np.stack(
        [
            simulate_l1a(small_cal_db, limb_scene, frames, seed=seed)["image"]
            .isel(time=0)
            .to_numpy()
            for seed in range(200)
        ]
    )
    noiseless = (
        simulate_l1a(small_cal_db, limb_scene, frames, noise=False)["image"]
        .isel(time=0)
        .to_numpy()
    )
    predicted = detector_variance(noiseless, small_cal_db, num_averaged_exposures=1)

    np.testing.assert_allclose(
        realizations.mean(axis=0).mean(), noiseless.mean(), rtol=1e-3
    )
    ratio = realizations.var(axis=0, ddof=1) / predicted
    # 200 realizations: the variance of each pixel is known to about 10%
    assert np.median(ratio) == pytest.approx(1.0, abs=0.03)


def test_averaging_exposures_reduces_noise(small_cal_db, limb_scene, make_frames):
    def pixel_scatter(num_exposures):
        frames = make_frames(
            num_averaged_exposures=num_exposures, blank_between_sets=False
        )
        images = np.stack(
            [
                simulate_l1a(small_cal_db, limb_scene, frames, seed=s)["image"]
                .isel(time=0)
                .to_numpy()
                for s in range(60)
            ]
        )
        return np.median(images.std(axis=0, ddof=1))

    assert pixel_scatter(4) / pixel_scatter(1) == pytest.approx(0.5, abs=0.07)


# ------------------------------------------------- shared model pieces
@pytest.mark.parametrize("roll", [0.0, 30.0, -45.0])
def test_stokes_sensitivities_rotate_with_roll(small_cal_db, small_instrument, roll):
    # An analyzer at theta from the instrument up axis is at theta + roll from
    # the vertical; checked directly because the forward model and L1A->L1B
    # share this function, so a sign error would cancel in a round trip
    a, b = stokes_sensitivities(small_cal_db, 0, np.deg2rad(roll))
    analyzer = pixel_array(small_cal_db, "pixel_analyzer")
    efficiency = small_instrument.analyzer_efficiency
    for k, theta in enumerate(small_instrument.analyzer_angles_deg):
        pixels = analyzer == k
        angle = np.deg2rad(2 * (theta + roll))
        np.testing.assert_allclose(a[pixels], efficiency * np.cos(angle), atol=1e-12)
        np.testing.assert_allclose(b[pixels], efficiency * np.sin(angle), atol=1e-12)


def test_photon_weighted_throughput_of_a_gaussian(small_cal_db, small_instrument):
    sigma = small_instrument.bandpass_fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
    centers = np.array(small_instrument.center_wavelengths_nm)
    expected = (
        small_instrument.peak_throughput
        * sigma
        * np.sqrt(2 * np.pi)
        * centers
        * 1e-9
        / (PLANCK * SPEED_OF_LIGHT)
    )
    np.testing.assert_allclose(
        photon_weighted_throughput(small_cal_db), expected, rtol=1e-6
    )


def test_vignetting_reaches_its_stated_value_at_the_corners():
    instrument = SimpleInstrument(
        **{**SMALL_INSTRUMENT_KWARGS, "pixel_response_nonuniformity": 0.0}
    )
    etendue = pixel_array(simulated_cal_db(instrument), "pixel_etendue")
    unvignetted = (
        np.pi
        * (instrument.aperture_diameter / 2) ** 2
        * np.deg2rad(instrument.vertical_ifov_deg)
        * np.deg2rad(instrument.horizontal_ifov_deg)
    )
    # Row 0 and the last column of the first sub-image is a corner
    corner = etendue[0, -1] / unvignetted
    assert corner == pytest.approx(1 - instrument.vignetting, rel=1e-9)


def test_scene_altitude_order_does_not_matter(small_cal_db, limb_scene, make_frames):
    frames = make_frames()
    ascending = simulate_l1a(small_cal_db, limb_scene, frames, noise=False)
    descending = simulate_l1a(
        small_cal_db,
        limb_scene.isel(altitude=slice(None, None, -1)),
        frames,
        noise=False,
    )
    np.testing.assert_array_equal(ascending["image"], descending["image"])


def test_scene_must_cover_the_altitudes_l1b_uses(small_cal_db, limb_scene, make_frames):
    short = limb_scene.sel(altitude=slice(5e3, None))
    with pytest.raises(ValueError, match="the scene covers"):
        simulate_l1a(small_cal_db, short, make_frames(), noise=False)
    with pytest.raises(ValueError, match="the scene covers"):
        exposure_for_fill(small_cal_db, short)


def test_scene_must_have_one_entry_per_channel(small_cal_db, limb_scene, make_frames):
    with pytest.raises(ValueError, match="spectral channels"):
        simulate_l1a(
            small_cal_db,
            limb_scene.isel(spectral_channel=[0]),
            make_frames(),
            noise=False,
        )


def test_integer_positions_give_valid_l1a(small_cal_db, limb_scene, make_frames):
    # Integer inputs are stored as floats, and the output is checked on creation
    frames = make_frames(spacecraft_latitude=40, spacecraft_longitude=-100)
    assert frames["spacecraft_latitude"].dtype.kind == "f"
    simulate_l1a(small_cal_db, limb_scene, frames, noise=False)


# ------------------------------------------------- scenes from sasktran2
def _sasktran_like_radiance(wavelengths, tangent_altitudes, stokes=("I", "Q", "U")):
    """Radiance shaped like sasktran2's output, with values that encode where they
    came from: channel centre + tangent altitude in km + Stokes index / 10."""
    values = (
        np.asarray(wavelengths)[:, np.newaxis, np.newaxis]
        + np.asarray(tangent_altitudes)[np.newaxis, :, np.newaxis] / 1e3
        + np.arange(len(stokes))[np.newaxis, np.newaxis, :] / 10
    )
    return xr.Dataset(
        {"radiance": (["wavelength", "los", "stokes"], values)},
        coords={"wavelength": wavelengths, "stokes": list(stokes)},
    )


def test_scene_from_radiance_orders_channels_and_altitudes(small_cal_db):
    tangent_altitudes = np.array([20e3, 0.0, 10e3])
    radiance = _sasktran_like_radiance([1020.0, 745.0], tangent_altitudes)
    scene = scene_from_radiance(small_cal_db, radiance, tangent_altitudes)

    np.testing.assert_array_equal(scene["altitude"], [0.0, 10e3, 20e3])
    np.testing.assert_allclose(scene["I"].sel(altitude=10e3), [755.0, 1030.0])
    np.testing.assert_allclose(scene["U"].sel(altitude=0.0), [745.2, 1020.2])
    assert scene["I"].attrs["units"] == "W m-2 sr-1 nm-1"


def test_scene_from_unpolarized_radiance(small_cal_db):
    radiance = _sasktran_like_radiance([745.0, 1020.0], [0.0, 10e3], stokes=("I",))
    scene = scene_from_radiance(small_cal_db, radiance["radiance"], [0.0, 10e3])
    assert np.all(scene["Q"] == 0)
    assert np.all(scene["U"] == 0)


def test_scene_needs_every_channel_centre(small_cal_db):
    with pytest.raises(ValueError, match="exactly one at each"):
        scene_from_radiance(
            small_cal_db, _sasktran_like_radiance([745.0, 745.0], [0.0]), [0.0]
        )
    with pytest.raises(ValueError, match="not channel centres"):
        scene_from_radiance(
            small_cal_db, _sasktran_like_radiance([745.0, 800.0], [0.0]), [0.0]
        )


def test_scene_needs_one_altitude_per_line_of_sight(small_cal_db):
    with pytest.raises(ValueError, match="lines of sight"):
        scene_from_radiance(
            small_cal_db, _sasktran_like_radiance([745.0, 1020.0], [0.0, 1e3]), [0.0]
        )

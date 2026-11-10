"""The placeholder calibration database and the L1A forward model."""

from __future__ import annotations

import numpy as np
import pytest

from aliprocessing.cal_db import SimpleInstrument, simulated_cal_db
from aliprocessing.forward_model import simulate_l1a
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
    frames = make_frames().isel(time=[0])
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
        frames = make_frames(num_averaged_exposures=num_exposures).isel(time=[0])
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

from __future__ import annotations

import numpy as np
import pytest
from conftest import SMALL_INSTRUMENT_KWARGS, SPACECRAFT

from aliprocessing.cal_db import SimpleInstrument, simulated_cal_db
from aliprocessing.forward_model import (
    exponential_scene,
    exposure_for_fill,
    simulate_l1a,
)
from aliprocessing.geometry import boresight_elevation_for_tangent_altitude
from aliprocessing.instrument import static_dark
from aliprocessing.l1b.data import IGNORED_NOISE, L1bImage
from aliprocessing.processing.l1a_to_l1b import process_l1a_to_l1b
from aliprocessing.schemas import SchemaValidationError, validate

Q, U = 0.1, 0.03


def _truth(l1b, scene):
    """The scene's band-averaged radiance on the L1B grid."""
    return (
        scene["I"]
        .interp(altitude=l1b["altitude"])
        .rename(spectral_channel="wavelength")
        .assign_coords(wavelength=l1b["wavelength"])
    )


@pytest.fixture(scope="module")
def noiseless_l1b(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(num_sets=2), noise=False)
    return process_l1a_to_l1b(l1a, small_cal_db)


def test_output_is_valid_l1b(noiseless_l1b, small_cal_db):
    assert validate(noiseless_l1b, "l1b", cal_db=small_cal_db) == []
    assert dict(noiseless_l1b.sizes) == {
        "observation_set": 2,
        "across_track": 4,
        "wavelength": 2,
        "altitude": 51,
        "reference_altitude": 3,
    }


def test_noiseless_round_trip(noiseless_l1b, limb_scene):
    assert np.isfinite(noiseless_l1b["radiance"]).all()
    # Local-linear resampling leaves only the curvature bias of the kernel,
    # about h² / (12 H²) = 0.3% for h = 1.3 km and a 7 km scale height
    relative = noiseless_l1b["radiance"] / _truth(noiseless_l1b, limb_scene) - 1
    assert float(np.abs(relative).max()) < 0.005
    # q and u are constant in the scene, so the ratios are exact
    np.testing.assert_allclose(noiseless_l1b["q"], Q, rtol=1e-9)
    np.testing.assert_allclose(noiseless_l1b["u"], U, rtol=1e-9)
    np.testing.assert_allclose(noiseless_l1b["dolp"], np.hypot(Q, U), rtol=1e-9)


def test_q_and_u_do_not_depend_on_roll(small_cal_db, limb_scene, make_frames):
    # q and u are in the local-vertical frame, so rolling the instrument must not
    # change them; this checks the forward and inverse rotations agree
    l1a = simulate_l1a(
        small_cal_db, limb_scene, make_frames(boresight_roll=30.0), noise=False
    )
    l1b = process_l1a_to_l1b(l1a, small_cal_db)
    finite = np.isfinite(l1b["q"])
    np.testing.assert_allclose(l1b["q"].to_numpy()[finite], Q, rtol=1e-9)
    np.testing.assert_allclose(l1b["u"].to_numpy()[finite], U, rtol=1e-9)


def test_dark_estimate_is_pluggable(small_cal_db, limb_scene, make_frames):
    # The true dark has drifted 40 counts above the calibration database
    drifted = small_cal_db.copy()
    drifted["dark_offset"] = small_cal_db["dark_offset"] + 40.0
    l1a = simulate_l1a(drifted, limb_scene, make_frames(), noise=False)

    stale = process_l1a_to_l1b(l1a, small_cal_db)
    corrected = process_l1a_to_l1b(
        l1a, small_cal_db, dark=lambda frame, cal_db: static_dark(frame, cal_db) + 40.0
    )
    truth = _truth(corrected, limb_scene)
    assert float(np.abs(corrected["radiance"] / truth - 1).max()) < 0.005
    # The stale dark leaves extra signal, worst where the scene is faint
    assert float((stale["radiance"] / truth - 1).sel(altitude=50e3).min()) > 0.05


def test_noise_estimate_matches_scatter(small_cal_db, limb_scene, make_frames):
    frames = make_frames()
    runs = [
        process_l1a_to_l1b(
            simulate_l1a(small_cal_db, limb_scene, frames, seed=seed), small_cal_db
        ).isel(observation_set=0)
        for seed in range(40)
    ]
    stacked = {
        name: np.stack(
            [run[name].sel(altitude=slice(5e3, 35e3)).to_numpy() for run in runs]
        )
        for name in (
            "radiance",
            "radiance_noise",
            "q",
            "q_noise",
            "u",
            "u_noise",
            "q_u_correlation",
        )
    }
    for name in ("radiance", "q", "u"):
        ratio = stacked[name].std(axis=0, ddof=1) / stacked[f"{name}_noise"].mean(
            axis=0
        )
        # 40 realizations: each cell's scatter is known to about 11%
        assert np.median(ratio) == pytest.approx(1.0, abs=0.06), name

    # Ideal analyzers at -60, 0 and 60 degrees leave q and u nearly uncorrelated,
    # and the reported correlation matches the realized one
    q_error = stacked["q"] - stacked["q"].mean(axis=0)
    u_error = stacked["u"] - stacked["u"].mean(axis=0)
    realized = (q_error * u_error).mean(axis=0) / (
        q_error.std(axis=0) * u_error.std(axis=0)
    )
    reported = stacked["q_u_correlation"].mean(axis=0)
    assert np.abs(np.median(reported)) < 0.1
    assert np.median(realized - reported) == pytest.approx(0.0, abs=0.06)


def test_cells_outside_the_field_of_view_are_nan(limb_scene, make_frames):
    # Coverage of 5 to 45 km leaves the ends of the 0-50 km grid unobserved
    narrow = SimpleInstrument(
        **{**SMALL_INSTRUMENT_KWARGS, "tangent_altitude_coverage": (5e3, 45e3)}
    )
    cal_db = simulated_cal_db(narrow)
    frames = make_frames()
    l1b = process_l1a_to_l1b(
        simulate_l1a(cal_db, limb_scene, frames, noise=False), cal_db
    )

    radiance = l1b["radiance"].isel(observation_set=0, wavelength=0)
    assert np.isnan(radiance.sel(altitude=[0.0, 50e3])).all()
    assert np.isfinite(radiance.sel(altitude=25e3)).all()
    # The arc lifts the top of the outer bins just above 45 km, so 45 km is
    # bracketed there; the centre bins top out just below it, and a cell with
    # pixels on one side only would be an extrapolation
    np.testing.assert_array_equal(
        np.isfinite(radiance.sel(altitude=45e3)), [True, False, False, True]
    )
    assert validate(l1b, "l1b", cal_db=cal_db) == []


def test_saturation_is_flagged(small_cal_db, limb_scene, make_frames):
    # Ten times the exposure that fills the brightest pixel to 60%: rows below
    # about 15 km saturate, the faint top of the limb does not
    exposure = 10 * exposure_for_fill(small_cal_db, limb_scene)
    l1a = simulate_l1a(
        small_cal_db, limb_scene, make_frames(exposure_time=exposure), seed=1
    )
    fraction = process_l1a_to_l1b(l1a, small_cal_db)["saturated_fraction"].isel(
        observation_set=0, wavelength=0
    )
    assert float(fraction.sel(altitude=5e3).min()) == 1.0
    assert float(fraction.sel(altitude=45e3).max()) == 0.0


def test_times_and_spacecraft_follow_the_frames(noiseless_l1b, make_frames):
    frames = make_frames(num_sets=2)
    in_set = frames["observation_set"] >= 0
    np.testing.assert_array_equal(
        noiseless_l1b["time"].to_numpy().ravel(), frames["time"][in_set].to_numpy()
    )
    np.testing.assert_allclose(noiseless_l1b["spacecraft_altitude"], 500e3)


def test_geometry_at_reference_altitudes(noiseless_l1b):
    latitude = noiseless_l1b["tangent_latitude"].isel(observation_set=0, wavelength=0)
    los_azimuth = noiseless_l1b["los_azimuth_angle"]
    # Looking east from 40N, the tangent points are within ~25 degrees of the spacecraft
    assert float(np.abs(latitude - 40.0).max()) < 5.0
    longitude = noiseless_l1b["tangent_longitude"]
    assert float(longitude.min()) > -100.0
    assert float(np.abs(los_azimuth - 90.0).max()) < 25.0
    # Across-track bins sit on either side of the boresight
    north_to_south = latitude.sel(reference_altitude=25e3).to_numpy()
    assert np.all(np.diff(north_to_south) < 0)


def test_blank_frames_are_optional(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(
        small_cal_db, limb_scene, make_frames(blank_between_sets=False), noise=False
    )
    assert process_l1a_to_l1b(l1a, small_cal_db).sizes["observation_set"] == 1


def test_invalid_l1a_is_rejected(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(), noise=False)
    incomplete = l1a.to_dataset(inherit=False).isel(time=[1, 2])
    with pytest.raises(SchemaValidationError, match="observation-sets-complete"):
        process_l1a_to_l1b(incomplete, small_cal_db)


def test_l1b_image_for_l2(noiseless_l1b):
    image = L1bImage.from_l1b(noiseless_l1b, 0, 2, max_dolp_noise=np.inf)
    assert set(image.spectra) == {"I", "dolp"}
    for spectra in image.spectra.values():
        ds = spectra.ds
        assert dict(ds["radiance"].sizes) == {"wavelength": 2, "los": 51}
        np.testing.assert_allclose(ds["tangent_altitude"], noiseless_l1b["altitude"])
    cell = noiseless_l1b.isel(observation_set=0, across_track=2)
    np.testing.assert_allclose(image.spectra["I"].ds["radiance"], cell["radiance"])
    np.testing.assert_allclose(image.spectra["dolp"].ds["radiance"], cell["dolp"])
    # Geometry is interpolated from the reference altitudes onto each altitude
    np.testing.assert_allclose(
        image.spectra["I"].ds["solar_zenith_angle"].isel(los=25),
        cell["solar_zenith_angle"].sel(reference_altitude=25e3).mean("wavelength"),
    )


def test_l1b_image_drops_altitudes_outside_the_field_of_view(noiseless_l1b):
    l1b = noiseless_l1b.copy(deep=True)
    l1b["radiance"][0, 2, 1, :3] = np.nan
    image = L1bImage.from_l1b(l1b, 0, 2, max_dolp_noise=np.inf)
    tangent_altitude = image.spectra["I"].ds["tangent_altitude"]
    assert float(tangent_altitude.min()) == 3e3
    assert tangent_altitude.size == 48


def test_l1b_image_handles_wrapping_angles(noiseless_l1b):
    # Viewing away from the sun near the dateline, looking north: every angle
    # sits next to its wrap point, and differs between channels and altitudes
    l1b = noiseless_l1b.copy(deep=True)
    shape = l1b["relative_solar_azimuth_angle"].shape
    alternating = np.where(np.indices(shape).sum(axis=0) % 2 == 0, 1.0, -1.0)
    l1b["relative_solar_azimuth_angle"][:] = 179.5 * alternating
    l1b["tangent_longitude"][:] = 179.8 * alternating
    l1b["los_azimuth_angle"][:] = np.where(alternating > 0, 359.6, 0.4)

    ds = L1bImage.from_l1b(l1b, 0, 2, max_dolp_noise=np.inf).spectra["I"].ds
    assert np.all(np.abs(ds["relative_solar_azimuth_angle"]) > 179)
    assert np.all(np.abs(ds["tangent_longitude"]) > 179)
    los_azimuth = ds["los_azimuth_angle"].to_numpy()
    assert np.all((los_azimuth < 1) | (los_azimuth > 359))
    assert np.all((los_azimuth >= 0) & (los_azimuth < 360))


def test_frame_pointed_above_the_grid_gives_nan(small_cal_db, limb_scene, make_frames):
    # Boresight at 200 km: no pixel is near the 0-50 km grid
    elevation = boresight_elevation_for_tangent_altitude(
        SPACECRAFT["spacecraft_latitude"],
        SPACECRAFT["spacecraft_longitude"],
        SPACECRAFT["spacecraft_altitude"],
        SPACECRAFT["boresight_azimuth"],
        200e3,
    )
    l1a = simulate_l1a(
        small_cal_db,
        limb_scene,
        make_frames(boresight_elevation=elevation),
        noise=False,
    )
    l1b = process_l1a_to_l1b(l1a, small_cal_db)
    assert np.isnan(l1b["radiance"]).all()
    assert np.isfinite(l1b["tangent_latitude"]).all()


def test_granule_of_blank_frames_is_rejected(small_cal_db, limb_scene, make_frames):
    frames = make_frames()
    blank = frames.isel(time=np.flatnonzero(frames["spectral_channel"].to_numpy() < 0))
    l1a = simulate_l1a(small_cal_db, limb_scene, blank, noise=False)
    with pytest.raises(ValueError, match="no observation sets"):
        process_l1a_to_l1b(l1a, small_cal_db)


def test_l1b_image_ignores_noisy_dolp_but_keeps_radiance(noiseless_l1b):
    threshold = 0.01
    cell = noiseless_l1b.isel(observation_set=0, across_track=2)
    image = L1bImage.from_l1b(noiseless_l1b, 0, 2, max_dolp_noise=threshold)

    # Every altitude stays, for both I and dolp
    for spectra in image.spectra.values():
        np.testing.assert_array_equal(spectra.ds["tangent_altitude"], cell["altitude"])
    np.testing.assert_allclose(image.spectra["I"].ds["radiance"], cell["radiance"])

    # dolp is ignored from the first altitude where its noise is too large
    noise = image.spectra["dolp"].ds["radiance_noise"].to_numpy()
    noisy = (cell["dolp_noise"] > threshold).any("wavelength").to_numpy()
    first_noisy = int(np.argmax(noisy))
    assert 20 < first_noisy < 50
    np.testing.assert_allclose(
        noise[:, :first_noisy],
        cell["dolp_noise"].transpose("wavelength", "altitude")[:, :first_noisy],
    )
    assert np.all(noise[:, first_noisy:] == IGNORED_NOISE)


def test_ignored_dolp_is_contiguous(noiseless_l1b):
    # One noisy cell low down: dolp is ignored from there upwards, with no holes
    l1b = noiseless_l1b.copy(deep=True)
    l1b["dolp_noise"][0, 2, 1, 10] = 1.0
    noise = (
        L1bImage.from_l1b(l1b, 0, 2, max_dolp_noise=0.01)
        .spectra["dolp"]
        .ds["radiance_noise"]
        .to_numpy()
    )
    assert np.all(noise[:, :10] < IGNORED_NOISE)
    assert np.all(noise[:, 10:] == IGNORED_NOISE)


def test_low_dolp_keeps_all_radiance(small_cal_db, make_frames):
    # Nearly unpolarized light: dolp is noise-dominated through most of the
    # stratosphere, but the radiance must still reach L2 at every altitude
    scene = exponential_scene(
        small_cal_db, radiance_at_zero=[0.05, 0.03], q=0.003, u=0.0
    )
    frames = make_frames(exposure_time=exposure_for_fill(small_cal_db, scene))
    l1b = process_l1a_to_l1b(
        simulate_l1a(small_cal_db, scene, frames, seed=2), small_cal_db
    )
    image = L1bImage.from_l1b(l1b, 0, 2, max_dolp_noise=0.01)

    assert image.spectra["I"].ds["tangent_altitude"].size == l1b.sizes["altitude"]
    ignored = (image.spectra["dolp"].ds["radiance_noise"] == IGNORED_NOISE).all(
        "wavelength"
    )
    first = int(np.argmax(ignored.to_numpy()))
    assert ignored.any()
    assert ignored.to_numpy()[first:].all()


def test_dolp_noise_threshold_is_required(noiseless_l1b):
    with pytest.raises(TypeError, match="max_dolp_noise"):
        L1bImage.from_l1b(noiseless_l1b, 0, 2)


def test_l1b_image_drops_saturated_altitudes(small_cal_db, limb_scene, make_frames):
    exposure = 10 * exposure_for_fill(small_cal_db, limb_scene)
    l1a = simulate_l1a(
        small_cal_db, limb_scene, make_frames(exposure_time=exposure), noise=False
    )
    l1b = process_l1a_to_l1b(l1a, small_cal_db)
    saturated = (
        l1b["saturated_fraction"].isel(observation_set=0, across_track=2) > 0
    ).any("wavelength")
    assert saturated.any()

    tangent_altitude = (
        L1bImage.from_l1b(l1b, 0, 2, max_dolp_noise=np.inf)
        .spectra["I"]
        .ds["tangent_altitude"]
    )
    np.testing.assert_array_equal(tangent_altitude, l1b["altitude"][~saturated])


def test_no_gaps_inside_the_field_of_view(limb_scene, make_frames):
    # One column per across-track bin, so the arc cannot fill in between rows:
    # the kernel alone must span the widest row spacing, at the bottom of the
    # coverage
    sparse = SimpleInstrument(
        **{**SMALL_INSTRUMENT_KWARGS, "num_columns": 4, "horizontal_ifov_deg": 0.8}
    )
    cal_db = simulated_cal_db(sparse)
    frames = make_frames()
    l1b = process_l1a_to_l1b(
        simulate_l1a(cal_db, limb_scene, frames, noise=False), cal_db
    )
    assert np.isfinite(l1b["radiance"]).all()


def test_invalid_cal_db_is_rejected(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(), noise=False)
    broken = small_cal_db.copy(deep=True)
    broken["pixel_etendue"][0, 0] = np.nan
    with pytest.raises(SchemaValidationError, match="pixel-calibration-finite"):
        process_l1a_to_l1b(l1a, broken)


def test_non_finite_dark_pixels_are_left_out(small_cal_db, limb_scene, make_frames):
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(), noise=False)
    reference = process_l1a_to_l1b(l1a, small_cal_db)

    def dark_with_a_bad_pixel(frame, cal_db):
        dark = static_dark(frame, cal_db)
        dark[20, 10] = np.nan
        return dark

    l1b = process_l1a_to_l1b(l1a, small_cal_db, dark=dark_with_a_bad_pixel)
    # One pixel out of each cell's many: still finite, and barely changed
    assert np.isfinite(l1b["radiance"]).all()
    np.testing.assert_allclose(l1b["radiance"], reference["radiance"], rtol=0.02)


def test_four_analyzers(limb_scene, make_frames):
    four = SimpleInstrument(
        **{**SMALL_INSTRUMENT_KWARGS, "analyzer_angles_deg": (0.0, 45.0, 90.0, 135.0)}
    )
    cal_db = simulated_cal_db(four)
    l1b = process_l1a_to_l1b(
        simulate_l1a(cal_db, limb_scene, make_frames(), noise=False), cal_db
    )
    finite = np.isfinite(l1b["q"])
    assert finite.any()
    np.testing.assert_allclose(l1b["q"].to_numpy()[finite], Q, rtol=1e-9)
    np.testing.assert_allclose(l1b["u"].to_numpy()[finite], U, rtol=1e-9)

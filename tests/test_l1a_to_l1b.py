from __future__ import annotations

import numpy as np
import pytest
from conftest import SMALL_INSTRUMENT_KWARGS

from aliprocessing.cal_db import SimpleInstrument, simulated_cal_db
from aliprocessing.forward_model import exposure_for_fill, simulate_l1a
from aliprocessing.instrument import static_dark
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
        name: np.stack([run[name].sel(altitude=slice(5e3, 35e3)).to_numpy() for run in runs])
        for name in ("radiance", "radiance_noise", "q", "q_noise", "u", "u_noise", "q_u_correlation")
    }
    for name in ("radiance", "q", "u"):
        ratio = stacked[name].std(axis=0, ddof=1) / stacked[f"{name}_noise"].mean(axis=0)
        # 40 realizations: each cell's scatter is known to about 11%
        assert np.median(ratio) == pytest.approx(1.0, abs=0.06), name

    # Ideal analyzers at -60, 0 and 60 degrees leave q and u nearly uncorrelated,
    # and the reported correlation matches the realized one
    q_error = stacked["q"] - stacked["q"].mean(axis=0)
    u_error = stacked["u"] - stacked["u"].mean(axis=0)
    realized = (q_error * u_error).mean(axis=0) / (q_error.std(axis=0) * u_error.std(axis=0))
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
    l1b = process_l1a_to_l1b(simulate_l1a(cal_db, limb_scene, frames, noise=False), cal_db)

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
    l1a = simulate_l1a(small_cal_db, limb_scene, make_frames(exposure_time=exposure), seed=1)
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

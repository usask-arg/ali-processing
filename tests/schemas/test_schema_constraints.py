from __future__ import annotations

import numpy as np
import pytest

from aliprocessing.instrument import across_track_bin
from aliprocessing.schemas import (
    CONSTRAINT_CHECKS,
    DEFERRED_CONSTRAINTS,
    PRODUCTS,
    load_schema,
    validate,
)
from aliprocessing.schemas._constraints import MAX_DEMODULATION_CONDITION


def _constraint_failures(issues):
    return {
        issue.location.removeprefix("constraints.")
        for issue in issues
        if issue.severity == "error" and issue.location.startswith("constraints.")
    }


@pytest.mark.parametrize("product", PRODUCTS)
def test_every_constraint_is_checked_or_explicitly_deferred(product):
    in_schema = {constraint["id"] for constraint in load_schema(product)["constraints"]}
    checked = set(CONSTRAINT_CHECKS.get(product, {}))
    deferred = set(DEFERRED_CONSTRAINTS.get(product, {}))
    assert checked.isdisjoint(deferred)
    assert checked | deferred == in_schema


@pytest.mark.parametrize(
    ("product", "fields"),
    [
        ("l1a", ["image", "time", "spectral_channel"]),
        ("l1b", ["time", "radiance", "dolp", "altitude"]),
        ("cal_db", ["pixel_analyzer", "spectral_throughput", "altitude"]),
    ],
)
def test_constraints_skip_when_their_fields_are_missing(product, fields, request):
    ds = request.getfixturevalue(product).drop_vars(fields)
    assert _constraint_failures(validate(ds, product)) == set()


def test_constraints_against_calibration_pass(l1a, l1b, cal_db):
    assert validate(l1a, "l1a", cal_db=cal_db) == []
    assert validate(l1b, "l1b", cal_db=cal_db) == []


# ------------------------------------------------------------------------ L1A
def test_l1a_time_not_increasing(l1a):
    l1a = l1a.isel(time=[1, 0, 2, 3, 4])
    assert "time-increasing" in _constraint_failures(validate(l1a, "l1a"))


def test_l1a_image_not_finite(l1a):
    l1a["image"][0, 0, 0] = np.inf
    assert _constraint_failures(validate(l1a, "l1a")) == {"image-finite"}


def test_l1a_detector_does_not_match_calibration(l1a, cal_db):
    l1a = l1a.isel(detector_column=slice(1, None))
    issues = validate(l1a, "l1a", cal_db=cal_db)
    assert _constraint_failures(issues) == {"detector-matches-calibration"}


def test_l1a_spectral_channel_unknown(l1a, cal_db):
    l1a["spectral_channel"][0] = 2
    issues = validate(l1a, "l1a", cal_db=cal_db)
    assert "spectral-channel-known" in _constraint_failures(issues)


def test_l1a_observation_set_missing_a_channel(l1a, cal_db):
    l1a = l1a.isel(time=[0, 1, 2, 3])
    issues = validate(l1a, "l1a", cal_db=cal_db)
    assert _constraint_failures(issues) == {"observation-sets-complete"}


def test_l1a_observation_set_with_repeated_channel(l1a):
    l1a["spectral_channel"][4] = 0
    assert _constraint_failures(validate(l1a, "l1a")) == {"observation-sets-complete"}


def test_l1a_blank_frame_in_an_observation_set(l1a):
    l1a["observation_set"][2] = 0
    assert _constraint_failures(validate(l1a, "l1a")) == {"observation-sets-complete"}


def test_l1a_frame_without_observation_set(l1a):
    l1a["observation_set"][4] = -1
    assert "observation-sets-complete" in _constraint_failures(validate(l1a, "l1a"))


def test_l1a_observation_set_decreasing(l1a):
    l1a["observation_set"][:] = [1, 1, -1, 0, 0]
    assert _constraint_failures(validate(l1a, "l1a")) == {"observation-sets-complete"}


def test_l1a_wheel_order_does_not_matter(l1a, cal_db):
    l1a["spectral_channel"][:] = [1, 0, -1, 1, 0]
    assert validate(l1a, "l1a", cal_db=cal_db) == []


# ------------------------------------------------------------------------ L1B
def test_l1b_times_need_not_increase_along_wavelength(l1b):
    l1b["time"] = l1b["time"].isel(wavelength=[1, 0])
    assert validate(l1b, "l1b") == []


def test_l1b_repeated_time_in_a_set(l1b):
    l1b["time"][0, 1] = l1b["time"][0, 0]
    assert _constraint_failures(validate(l1b, "l1b")) == {"time-ordered"}


def test_l1b_sets_overlap_in_time(l1b):
    l1b["time"] = l1b["time"].isel(observation_set=[1, 0])
    assert _constraint_failures(validate(l1b, "l1b")) == {"time-ordered"}


def test_l1b_altitude_not_increasing(l1b):
    l1b = l1b.isel(altitude=[1, 0, 2, 3, 4])
    assert _constraint_failures(validate(l1b, "l1b")) == {"grid-increasing"}


def test_l1b_grid_does_not_match_calibration(l1b, cal_db):
    l1b = l1b.assign_coords(altitude=l1b["altitude"] + 1.0)
    issues = validate(l1b, "l1b", cal_db=cal_db)
    assert _constraint_failures(issues) == {"grid-matches-calibration"}


def test_l1b_nan_outside_field_of_view_is_allowed(l1b):
    for name in ["radiance", "radiance_noise", "q", "q_noise", "u", "u_noise",
                 "dolp", "dolp_noise", "q_u_correlation"]:  # fmt: skip
        l1b[name][:, :, :, -1] = np.nan
    assert validate(l1b, "l1b") == []


def test_l1b_partly_finite_cell(l1b):
    l1b["u_noise"][0, 0, 0, 0] = np.nan
    assert _constraint_failures(validate(l1b, "l1b")) == {"finite-together"}


def test_l1b_geometry_not_finite(l1b):
    l1b["tangent_latitude"][0, 0, 0, 0] = np.nan
    assert _constraint_failures(validate(l1b, "l1b")) == {"finite-together"}


def test_l1b_dolp_inconsistent(l1b):
    l1b["dolp"][0, 0, 0, 0] += 0.01
    assert _constraint_failures(validate(l1b, "l1b")) == {"dolp-consistent"}


# --------------------------------------------------------------------- CAL_DB
def test_cal_db_pixel_analyzer_out_of_range(cal_db):
    cal_db["pixel_analyzer"][0, 0] = 3
    assert "analyzer-pixels" in _constraint_failures(validate(cal_db, "cal_db"))


def test_cal_db_analyzer_without_pixels(cal_db):
    cal_db["pixel_analyzer"] = cal_db["pixel_analyzer"].where(
        cal_db["pixel_analyzer"] != 2, -1
    )
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert "analyzer-pixels" in failures


def test_cal_db_used_pixel_without_field_angle(cal_db):
    cal_db["pixel_azimuth_angle"][0, 0] = np.nan
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert "pixel-calibration-finite" in failures


def test_cal_db_used_pixel_without_etendue(cal_db):
    cal_db["pixel_etendue"][1, 3] = 0.0
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"pixel-calibration-finite"}


def test_cal_db_unphysical_mueller_elements(cal_db):
    cal_db["q_sensitivity"][0, 0, 0] = 0.9
    cal_db["u_sensitivity"][0, 0, 0] = 0.9
    assert "mueller-physical" in _constraint_failures(validate(cal_db, "cal_db"))


def test_cal_db_analyzers_at_the_same_angle(cal_db):
    # Analyzer 2 copies analyzer 0, leaving U unmeasured
    for name in ["q_sensitivity", "u_sensitivity"]:
        cal_db[name][..., 6:] = cal_db[name][..., 0:2].to_numpy()
    assert _constraint_failures(validate(cal_db, "cal_db")) == {"demodulable"}


def test_cal_db_analyzers_nearly_at_the_same_angle(cal_db):
    # Analyzer 2 at -59.8 degrees, 0.2 degrees from analyzer 0: full rank, but
    # U is barely measured
    two_theta = np.deg2rad(2 * -59.8)
    cal_db["q_sensitivity"][..., 6:] = np.cos(two_theta)
    cal_db["u_sensitivity"][..., 6:] = np.sin(two_theta)
    matrix = np.array(
        [[1, np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in (-120, 0, -119.6)]
    )
    assert np.linalg.matrix_rank(matrix) == 3
    assert np.linalg.cond(matrix) > MAX_DEMODULATION_CONDITION
    assert _constraint_failures(validate(cal_db, "cal_db")) == {"demodulable"}


def test_cal_db_ideal_analyzers_are_well_conditioned(cal_db):
    # Ideal analyzers at -60, 0 and 60 degrees give condition number sqrt(2)
    theta = np.deg2rad([-60.0, 0.0, 60.0])
    matrix = np.stack([np.ones(3), np.cos(2 * theta), np.sin(2 * theta)], axis=1)
    np.testing.assert_allclose(np.linalg.cond(matrix), np.sqrt(2))
    assert validate(cal_db, "cal_db") == []


def test_cal_db_throughput_zero_for_a_channel(cal_db):
    cal_db["spectral_throughput"][1] = 0.0
    assert _constraint_failures(validate(cal_db, "cal_db")) == {"throughput-valid"}


def test_cal_db_center_outside_passband(cal_db):
    cal_db["center_wavelength"][0] = 800.0
    assert _constraint_failures(validate(cal_db, "cal_db")) == {"throughput-valid"}


def test_cal_db_reference_outside_altitude_grid(cal_db):
    cal_db = cal_db.assign_coords(reference_altitude=[15e3, 35e3])
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert failures == {"reference-within-grid"}


def test_cal_db_across_track_bin_missing_an_analyzer(cal_db):
    cal_db["pixel_azimuth_angle"][:, 7] = -0.5
    failures = _constraint_failures(validate(cal_db, "cal_db"))
    assert "across-track-coverage" in failures


def test_across_track_bins_are_half_open():
    edges = np.array([-1.0, 0.0, 1.0])
    azimuth = np.array([-1.0, -0.1, 0.0, 0.99, 1.0, -1.5, np.nan])
    np.testing.assert_array_equal(
        across_track_bin(azimuth, edges), [0, 0, 1, 1, -1, -1, -1]
    )

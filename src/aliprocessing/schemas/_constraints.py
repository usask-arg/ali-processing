"""Checks for the `constraints` section of each schema, keyed by constraint id.

Each check takes the dataset and an optional calibration database and returns a
list of problems. A check returns nothing when a field it needs is missing; the
per-field checks already report missing fields.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

# Largest condition number of the analyzer matrix [1, a, b] accepted by the
# demodulable constraint. Ideal analyzers at -60, 0 and 60 degrees give sqrt(2).
MAX_DEMODULATION_CONDITION = 100.0

# Tolerances for comparing derived or copied values
_MUELLER_ATOL = 1e-9
_DOLP_RTOL = 1e-6
_DOLP_ATOL = 1e-9

Check = Callable[[xr.Dataset, xr.Dataset | None], list[str]]

L1B_MEASUREMENTS = (
    "radiance",
    "radiance_noise",
    "q",
    "q_noise",
    "u",
    "u_noise",
    "dolp",
    "dolp_noise",
    "q_u_correlation",
)
L1B_GEOMETRY = (
    "tangent_latitude",
    "tangent_longitude",
    "spacecraft_latitude",
    "spacecraft_longitude",
    "spacecraft_altitude",
    "solar_zenith_angle",
    "relative_solar_azimuth_angle",
    "los_azimuth_angle",
)
_PIXEL_CALIBRATION = (
    "pixel_elevation_angle",
    "pixel_azimuth_angle",
    "pixel_etendue",
    "q_sensitivity",
    "u_sensitivity",
)


def _has(ds: xr.Dataset, *names: str) -> bool:
    return all(name in ds.variables for name in names)


def _ids(mask: np.ndarray, ids: np.ndarray | None = None) -> str:
    ids = np.flatnonzero(mask) if ids is None else ids[mask]
    return ", ".join(str(i) for i in ids)


def across_track_bin(azimuth: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Across-track bin of each pixel azimuth, or -1 outside every bin.

    Bins are half-open, [edges[j], edges[j + 1]). NaN azimuths are outside.
    """
    index = np.digitize(azimuth, edges) - 1
    return np.where((index >= 0) & (index < len(edges) - 1), index, -1)


def _strictly_increasing(*names: str) -> Check:
    def check(ds, _cal_db):
        return [
            f"{name} is not strictly increasing"
            for name in names
            if _has(ds, name) and not np.all(np.diff(ds[name].to_numpy()) > 0)
        ]

    return check


def _all_finite(*names: str) -> Check:
    def check(ds, _cal_db):
        return [
            f"{name} has {np.count_nonzero(~np.isfinite(ds[name]))} non-finite values"
            for name in names
            if _has(ds, name) and not np.all(np.isfinite(ds[name]))
        ]

    return check


# ------------------------------------------------------------------------ L1A
def _time_increasing(ds, _cal_db):
    if not _has(ds, "time") or ds["time"].ndim != 1:
        return []
    if np.all(np.diff(ds["time"].to_numpy()) > np.timedelta64(0, "ns")):
        return []
    return ["time is not strictly increasing"]


def _detector_matches_calibration(ds, cal_db):
    if cal_db is None:
        return []
    return [
        f"{dim} has length {ds.sizes[dim]}, but the calibration database has "
        f"{cal_db.sizes[dim]}"
        for dim in ("detector_row", "detector_column")
        if dim in ds.sizes
        and dim in cal_db.sizes
        and ds.sizes[dim] != cal_db.sizes[dim]
    ]


def _spectral_channel_known(ds, cal_db):
    if cal_db is None or "spectral_channel" not in cal_db.sizes:
        return []
    if not _has(ds, "spectral_channel"):
        return []
    channel = ds["spectral_channel"].to_numpy()
    unknown = channel >= cal_db.sizes["spectral_channel"]
    if not np.any(unknown):
        return []
    message = (
        f"spectral_channel at frames {_ids(unknown)} is not in the calibration "
        f"database, which has {cal_db.sizes['spectral_channel']} channels"
    )
    return [message]


def _observation_sets_complete(ds, cal_db):
    if not _has(ds, "spectral_channel", "observation_set"):
        return []
    channel = ds["spectral_channel"].to_numpy()
    obs_set = ds["observation_set"].to_numpy()
    blank = channel == -1

    problems = []
    if np.any(obs_set[blank] != -1):
        problems.append(
            f"blank frames {_ids(blank & (obs_set != -1))} have an observation_set"
        )
    unassigned = ~blank & (obs_set == -1)
    if np.any(unassigned):
        problems.append(f"frames {_ids(unassigned)} have no observation_set")

    in_set = ~blank & (obs_set != -1)
    if np.any(np.diff(obs_set[in_set]) < 0):
        problems.append("observation_set decreases in time")

    if cal_db is not None and "spectral_channel" in cal_db.sizes:
        expected = list(range(cal_db.sizes["spectral_channel"]))
    else:
        first = obs_set[in_set][:1]
        expected = sorted(channel[in_set & np.isin(obs_set, first)].tolist())
    incomplete = [
        s
        for s in np.unique(obs_set[in_set])
        if sorted(channel[in_set & (obs_set == s)].tolist()) != expected
    ]
    if incomplete:
        problems.append(
            f"observation sets {', '.join(map(str, incomplete))} do not hold exactly "
            f"one frame at each spectral channel {expected}"
        )
    return problems


# ------------------------------------------------------------------------ L1B
def _time_ordered(ds, _cal_db):
    if not _has(ds, "time"):
        return []
    time = ds["time"].transpose("observation_set", "wavelength").to_numpy()
    problems = []
    repeated = np.array([len(np.unique(row)) < len(row) for row in time], dtype=bool)
    if np.any(repeated):
        problems.append(f"observation sets {_ids(repeated)} repeat a time")
    overlaps = time.min(axis=1)[1:] <= time.max(axis=1)[:-1]
    if np.any(overlaps):
        problems.append(
            f"observation sets {_ids(overlaps, np.arange(1, len(time)))} "
            "start before the previous set ends"
        )
    return problems


_L1B_GRID_FROM_CAL_DB = {
    "wavelength": "center_wavelength",
    "bandpass_fwhm": "bandpass_fwhm",
    "altitude": "altitude",
    "reference_altitude": "reference_altitude",
}


def _grid_matches_calibration(ds, cal_db):
    if cal_db is None:
        return []
    problems = []
    for name, cal_name in _L1B_GRID_FROM_CAL_DB.items():
        if not (_has(ds, name) and _has(cal_db, cal_name)):
            continue
        values, expected = ds[name].to_numpy(), cal_db[cal_name].to_numpy()
        if values.shape != expected.shape or not np.allclose(values, expected):
            problems.append(
                f"{name} differs from {cal_name} in the calibration database"
            )
    return problems


def _finite_together(ds, _cal_db):
    present = [name for name in L1B_MEASUREMENTS if _has(ds, name)]
    problems = []
    if present:
        dims = ds[present[0]].dims
        finite = np.stack(
            [np.isfinite(ds[name].transpose(*dims).to_numpy()) for name in present]
        )
        mixed = np.any(finite, axis=0) & ~np.all(finite, axis=0)
        if np.any(mixed):
            problems.append(
                f"{np.count_nonzero(mixed)} cells are finite in some of "
                f"{', '.join(present)} but not all"
            )
    problems += _all_finite(*L1B_GEOMETRY)(ds, None)
    if _has(ds, "time") and np.any(np.isnat(ds["time"].to_numpy())):
        problems.append("time has missing values")
    return problems


def _dolp_consistent(ds, _cal_db):
    if not _has(ds, "dolp", "q", "u"):
        return []
    dims = ds["dolp"].dims
    dolp = ds["dolp"].to_numpy()
    expected = np.hypot(
        ds["q"].transpose(*dims).to_numpy(), ds["u"].transpose(*dims).to_numpy()
    )
    both = np.isfinite(dolp) & np.isfinite(expected)
    if np.allclose(dolp[both], expected[both], rtol=_DOLP_RTOL, atol=_DOLP_ATOL):
        return []
    return ["dolp differs from sqrt(q² + u²)"]


# --------------------------------------------------------------------- CAL_DB
def _pixel_array(ds: xr.Dataset, name: str) -> np.ndarray:
    """``name`` with the detector dimensions last."""
    var = ds[name]
    leading = [
        dim for dim in var.dims if dim not in ("detector_row", "detector_column")
    ]
    return var.transpose(*leading, "detector_row", "detector_column").to_numpy()


def _analyzer_pixels(ds, _cal_db):
    if not _has(ds, "pixel_analyzer") or "analyzer" not in ds.sizes:
        return []
    pixel_analyzer = ds["pixel_analyzer"].to_numpy()
    num_analyzers = ds.sizes["analyzer"]
    problems = []
    if np.any(pixel_analyzer >= num_analyzers):
        problems.append(
            f"pixel_analyzer has values above the last analyzer, {num_analyzers - 1}"
        )
    empty = [k for k in range(num_analyzers) if not np.any(pixel_analyzer == k)]
    if empty:
        problems.append(f"analyzers {', '.join(map(str, empty))} have no pixels")
    return problems


def _pixel_calibration_finite(ds, _cal_db):
    if not _has(ds, "pixel_analyzer"):
        return []
    used = (
        ds["pixel_analyzer"].transpose("detector_row", "detector_column").to_numpy()
        >= 0
    )
    problems = []
    for name in _PIXEL_CALIBRATION:
        if not _has(ds, name):
            continue
        values = _pixel_array(ds, name)
        bad = np.any(~np.isfinite(values), axis=tuple(range(values.ndim - 2))) & used
        if np.any(bad):
            problems.append(
                f"{name} is not finite at {np.count_nonzero(bad)} used pixels"
            )
    if _has(ds, "pixel_etendue"):
        etendue = _pixel_array(ds, "pixel_etendue")
        if np.any(etendue[used] <= 0):
            problems.append("pixel_etendue is not positive at every used pixel")
    return problems


def _mueller_physical(ds, _cal_db):
    if not _has(ds, "q_sensitivity", "u_sensitivity"):
        return []
    norm = np.hypot(
        _pixel_array(ds, "q_sensitivity"), _pixel_array(ds, "u_sensitivity")
    )
    excess = norm[np.isfinite(norm)] > 1 + _MUELLER_ATOL
    if not np.any(excess):
        return []
    return [
        f"q_sensitivity² + u_sensitivity² exceeds 1 at {np.count_nonzero(excess)} pixels"
    ]


def _bin_layout(ds: xr.Dataset):
    """Analyzer and across-track bin of every pixel, or None if fields are missing."""
    if not _has(ds, "pixel_analyzer", "pixel_azimuth_angle", "across_track_edge"):
        return None
    analyzer = (
        ds["pixel_analyzer"].transpose("detector_row", "detector_column").to_numpy()
    )
    bins = across_track_bin(
        _pixel_array(ds, "pixel_azimuth_angle"), ds["across_track_edge"].to_numpy()
    )
    return analyzer, bins, ds.sizes["across_track_edge"] - 1


def _demodulable(ds, _cal_db):
    layout = _bin_layout(ds)
    if layout is None or not _has(ds, "q_sensitivity", "u_sensitivity"):
        return []
    analyzer, bins, num_bins = layout
    a_all = _pixel_array(ds, "q_sensitivity")
    b_all = _pixel_array(ds, "u_sensitivity")

    bad = []
    for channel in range(a_all.shape[0]):
        for j in range(num_bins):
            rows = []
            for k in np.unique(analyzer[analyzer >= 0]):
                pixels = (analyzer == k) & (bins == j)
                if np.any(pixels):
                    a = np.nanmean(a_all[channel][pixels])
                    b = np.nanmean(b_all[channel][pixels])
                    rows.append([1.0, a, b])
            matrix = np.array(rows).reshape(-1, 3)
            if (
                len(matrix) < 3
                or not np.all(np.isfinite(matrix))
                or np.linalg.matrix_rank(matrix) < 3
                or np.linalg.cond(matrix) > MAX_DEMODULATION_CONDITION
            ):
                bad.append(f"(channel {channel}, bin {j})")
    if not bad:
        return []
    return ["the analyzer matrix is singular or ill-conditioned at " + ", ".join(bad)]


def _throughput_valid(ds, _cal_db):
    if not _has(ds, "throughput_wavelength", "spectral_throughput"):
        return []
    wavelength = ds["throughput_wavelength"].to_numpy()
    if not np.all(np.diff(wavelength) > 0):
        return ["throughput_wavelength is not strictly increasing"]
    throughput = (
        ds["spectral_throughput"]
        .transpose("spectral_channel", "throughput_wavelength")
        .to_numpy()
    )
    centers = (
        ds["center_wavelength"].to_numpy() if _has(ds, "center_wavelength") else None
    )

    problems = []
    for channel, row in enumerate(throughput):
        passband = wavelength[row > 0]
        if passband.size == 0:
            problems.append(
                f"spectral_throughput of channel {channel} is zero everywhere"
            )
        elif centers is not None and not (
            passband.min() <= centers[channel] <= passband.max()
        ):
            problems.append(
                f"center_wavelength of channel {channel}, {centers[channel]}, is outside "
                f"its passband [{passband.min()}, {passband.max()}]"
            )
    return problems


def _reference_within_grid(ds, _cal_db):
    if not _has(ds, "altitude", "reference_altitude"):
        return []
    altitude = ds["altitude"].to_numpy()
    reference = ds["reference_altitude"].to_numpy()
    outside = (reference < altitude.min()) | (reference > altitude.max())
    if not np.any(outside):
        return []
    return [f"reference_altitude {reference[outside].tolist()} is outside altitude"]


def _across_track_coverage(ds, _cal_db):
    layout = _bin_layout(ds)
    if layout is None or "analyzer" not in ds.sizes:
        return []
    analyzer, bins, num_bins = layout
    missing = [
        f"(bin {j}, analyzer {k})"
        for j in range(num_bins)
        for k in range(ds.sizes["analyzer"])
        if not np.any((analyzer == k) & (bins == j))
    ]
    if not missing:
        return []
    return ["no pixels at " + ", ".join(missing)]


CONSTRAINT_CHECKS: dict[str, dict[str, Check]] = {
    "l1a": {
        "time-increasing": _time_increasing,
        "image-finite": _all_finite("image"),
        "detector-matches-calibration": _detector_matches_calibration,
        "spectral-channel-known": _spectral_channel_known,
        "observation-sets-complete": _observation_sets_complete,
    },
    "l1b": {
        "time-ordered": _time_ordered,
        "grid-increasing": _strictly_increasing(
            "altitude", "reference_altitude", "across_track_angle"
        ),
        "grid-matches-calibration": _grid_matches_calibration,
        "finite-together": _finite_together,
        "dolp-consistent": _dolp_consistent,
    },
    "cal_db": {
        "analyzer-pixels": _analyzer_pixels,
        "pixel-calibration-finite": _pixel_calibration_finite,
        "mueller-physical": _mueller_physical,
        "demodulable": _demodulable,
        "throughput-valid": _throughput_valid,
        "grid-increasing": _strictly_increasing(
            "altitude", "reference_altitude", "across_track_edge"
        ),
        "reference-within-grid": _reference_within_grid,
        "across-track-coverage": _across_track_coverage,
    },
}

# Constraints in the schemas that are not checked yet, with the reason
DEFERRED_CONSTRAINTS: dict[str, dict[str, str]] = {}

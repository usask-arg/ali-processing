"""L1A→L1B processing for the simple instrument model.

Implements the ``model.inverse`` section of the calibration database schema:
calibrate each pixel, resample each analyzer onto the altitude grid using every
pixel's own tangent altitude, then demodulate each cell into I, Q and U.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aliprocessing.geometry import (
    InstrumentFrame,
    LocalSphere,
    look_at_tangent_altitude,
    sun_direction,
    tangent_geometry,
)
from aliprocessing.instrument import (
    DarkModel,
    detector_variance,
    electron_rate_per_radiance,
    pixel_array,
    static_dark,
    stokes_sensitivities,
)
from aliprocessing.schemas import apply_metadata, check
from aliprocessing.schemas._constraints import across_track_bin

_CELL_VARIABLES = (
    "radiance",
    "radiance_noise",
    "q",
    "q_noise",
    "u",
    "u_noise",
    "dolp",
    "dolp_noise",
    "q_u_correlation",
    "saturated_fraction",
)
_GEOMETRY_VARIABLES = (
    "tangent_latitude",
    "tangent_longitude",
    "solar_zenith_angle",
    "relative_solar_azimuth_angle",
    "los_azimuth_angle",
)
_SPACECRAFT_VARIABLES = ("spacecraft_latitude", "spacecraft_longitude", "spacecraft_altitude")


def process_l1a_to_l1b(
    l1a: xr.Dataset | xr.DataTree,
    cal_db: xr.Dataset,
    *,
    dark: DarkModel = static_dark,
) -> xr.Dataset:
    """Process an L1A granule to L1B.

    Parameters
    ----------
    l1a
        L1A granule. Only the root group is read; the ``simulation`` group of a
        simulated granule is ignored.
    cal_db
        Calibration database.
    dark
        Dark estimate for each frame, ``dark(frame, cal_db)`` in counts per pixel.
        Defaults to the calibration database's static dark.

    Returns
    -------
    xr.Dataset
        The L1B granule, checked against the L1B schema.
    """
    if isinstance(l1a, xr.DataTree):
        l1a = l1a.to_dataset(inherit=False)
    check(l1a, "l1a", cal_db=cal_db)
    layout = _PixelLayout.from_cal_db(cal_db)

    channel = l1a["spectral_channel"].to_numpy()
    obs_set = l1a["observation_set"].to_numpy()
    set_ids = np.unique(obs_set[channel >= 0])
    num_channels = cal_db.sizes["spectral_channel"]

    cells = {name: [] for name in _CELL_VARIABLES}
    geometry = {name: [] for name in _GEOMETRY_VARIABLES}
    per_frame = {name: [] for name in ("time", *_SPACECRAFT_VARIABLES)}
    for set_id in set_ids:
        set_cells = {name: [] for name in _CELL_VARIABLES}
        set_geometry = {name: [] for name in _GEOMETRY_VARIABLES}
        set_frame = {name: [] for name in per_frame}
        for w in range(num_channels):
            (index,) = np.flatnonzero((obs_set == set_id) & (channel == w))
            frame = l1a.isel(time=index)
            frame_cells, frame_geometry = _process_frame(frame, w, cal_db, layout, dark)
            for name in _CELL_VARIABLES:
                set_cells[name].append(frame_cells[name])
            for name in _GEOMETRY_VARIABLES:
                set_geometry[name].append(frame_geometry[name])
            for name in per_frame:
                set_frame[name].append(frame[name].to_numpy())
        # Stack channels on axis 1: (across_track, wavelength, ...)
        for name in _CELL_VARIABLES:
            cells[name].append(np.stack(set_cells[name], axis=1))
        for name in _GEOMETRY_VARIABLES:
            geometry[name].append(np.stack(set_geometry[name], axis=1))
        for name in per_frame:
            per_frame[name].append(np.stack(set_frame[name]))

    edges = cal_db["across_track_edge"].to_numpy()
    cell_dims = ["observation_set", "across_track", "wavelength", "altitude"]
    geometry_dims = ["observation_set", "across_track", "wavelength", "reference_altitude"]
    frame_dims = ["observation_set", "wavelength"]
    l1b = xr.Dataset(
        {
            "bandpass_fwhm": ("wavelength", cal_db["bandpass_fwhm"].to_numpy()),
            "across_track_angle": ("across_track", 0.5 * (edges[:-1] + edges[1:])),
            **{name: (cell_dims, np.stack(values)) for name, values in cells.items()},
            **{name: (geometry_dims, np.stack(values)) for name, values in geometry.items()},
            **{name: (frame_dims, np.stack(values)) for name, values in per_frame.items()},
        },
        coords={
            "wavelength": cal_db["center_wavelength"].to_numpy(),
            "altitude": cal_db["altitude"].to_numpy(),
            "reference_altitude": cal_db["reference_altitude"].to_numpy(),
        },
        attrs={"radiance_noise_model": "detector"},
    )
    l1b = apply_metadata(l1b, "l1b")
    check(l1b, "l1b", cal_db=cal_db)
    return l1b


@dataclass(frozen=True)
class _PixelLayout:
    """The used pixels and their fixed properties, flattened."""

    used: np.ndarray
    analyzer: np.ndarray
    across_track_bin: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    num_analyzers: int
    num_bins: int
    bin_centers: np.ndarray

    @classmethod
    def from_cal_db(cls, cal_db: xr.Dataset) -> _PixelLayout:
        analyzer = pixel_array(cal_db, "pixel_analyzer")
        azimuth = pixel_array(cal_db, "pixel_azimuth_angle")
        edges = cal_db["across_track_edge"].to_numpy()
        bins = across_track_bin(azimuth, edges)
        used = (analyzer >= 0) & (bins >= 0)
        return cls(
            used=used,
            analyzer=analyzer[used],
            across_track_bin=bins[used],
            azimuth=azimuth[used],
            elevation=pixel_array(cal_db, "pixel_elevation_angle")[used],
            num_analyzers=cal_db.sizes["analyzer"],
            num_bins=len(edges) - 1,
            bin_centers=0.5 * (edges[:-1] + edges[1:]),
        )


def _process_frame(frame, channel, cal_db, layout, dark):
    used = layout.used
    exposure_time = float(frame["exposure_time"])
    num_exposures = int(frame.get("num_averaged_exposures", 1))
    gain = float(cal_db["detector_gain"])
    image = frame["image"].transpose("detector_row", "detector_column").to_numpy()

    # 1. Calibrate each pixel to m = I + a' Q + b' U, in radiance units
    scale = gain / (exposure_time * electron_rate_per_radiance(cal_db, channel))
    signal = ((image - dark(frame, cal_db)) * scale)[used]
    variance = (detector_variance(image, cal_db, num_exposures) * scale**2)[used]

    instrument = InstrumentFrame.from_attitude(
        *(
            float(frame[name])
            for name in (
                "spacecraft_latitude",
                "spacecraft_longitude",
                "spacecraft_altitude",
                "boresight_azimuth_angle",
                "boresight_elevation_angle",
                "boresight_roll_angle",
            )
        )
    )
    a, b = (s[used] for s in stokes_sensitivities(cal_db, channel, instrument.roll_rad))
    if "saturated" in frame:
        saturated = frame["saturated"].transpose("detector_row", "detector_column").to_numpy()[used]
    else:
        saturated = np.zeros(signal.shape, dtype=bool)

    # 2. Tangent altitude of every pixel
    sphere = LocalSphere.at_tangent_point(instrument.observer, instrument.boresight)
    looks = instrument.look_vectors(layout.azimuth, layout.elevation)
    tangent_altitude = sphere.tangent_altitude(instrument.observer, looks)

    # 3. Resample each (across-track bin, analyzer) onto the altitude grid
    resampled = _resample(
        tangent_altitude,
        cal_db["altitude"].to_numpy(),
        float(cal_db["altitude_kernel_half_width"]),
        layout.across_track_bin * layout.num_analyzers + layout.analyzer,
        layout.num_bins * layout.num_analyzers,
        {"m": signal, "a": a, "b": b},
        variance,
        saturated,
    )
    shape = (layout.num_bins, layout.num_analyzers, -1)
    estimate = {name: resampled[name].reshape(shape) for name in ("m", "a", "b", "variance")}
    with np.errstate(invalid="ignore", divide="ignore"):
        saturated_fraction = (
            resampled["count_saturated"].reshape(shape).sum(axis=1)
            / resampled["count"].reshape(shape).sum(axis=1)
        )

    # 4. Demodulate each cell
    cells = _demodulate(estimate["m"], estimate["a"], estimate["b"], estimate["variance"])
    cells["saturated_fraction"] = np.where(
        np.isfinite(cells["radiance"]), saturated_fraction, np.nan
    )

    # Geolocation and solar geometry at the reference altitudes of each bin
    sun = sun_direction(frame["time"].to_numpy())
    reference_altitude = cal_db["reference_altitude"].to_numpy()
    looks = np.concatenate(
        [
            look_at_tangent_altitude(instrument, sphere, center, reference_altitude)
            for center in layout.bin_centers
        ]
    )
    points = tangent_geometry(instrument.observer, looks, sun)
    geometry_shape = (layout.num_bins, len(reference_altitude))
    geometry = {
        "tangent_latitude": points.latitude.reshape(geometry_shape),
        "tangent_longitude": points.longitude.reshape(geometry_shape),
        "solar_zenith_angle": points.solar_zenith_angle.reshape(geometry_shape),
        "relative_solar_azimuth_angle": points.relative_solar_azimuth_angle.reshape(
            geometry_shape
        ),
        "los_azimuth_angle": points.los_azimuth_angle.reshape(geometry_shape),
    }
    return cells, geometry


def _resample(altitude, grid, half_width, group, num_groups, values, variance, saturated):
    """Local-linear kernel estimates on ``grid`` for each group of pixels.

    For each grid altitude z_i, fits a straight line in tangent altitude to the
    group's pixels within ``half_width``, weighted by the triangular kernel
    w = 1 − |z_p − z_i| / h, and evaluates it at z_i. Unlike a weighted mean,
    this has no first-order bias when the pixel altitudes are unevenly spread
    around z_i. The estimate is linear in the pixels, Σ ℓ_p x_p with Σ ℓ_p = 1,
    so the same ℓ_p resample every value and give the variance Σ ℓ_p² var_p.

    Cells without pixels on both sides of z_i would be extrapolated and are NaN.
    Returns flat arrays of shape (num_groups * len(grid),): each value's
    estimate, ``variance``, and the number of contributing pixels (``count``) and
    of saturated ones (``count_saturated``).
    """
    num_grid = len(grid)
    size = num_groups * num_grid
    low = np.searchsorted(grid, altitude - half_width, side="right")
    high = np.searchsorted(grid, altitude + half_width, side="left")

    pixels, cells = [], []
    for offset in range(int(np.max(high - low, initial=0))):
        index = low + offset
        (inside,) = np.nonzero(index < high)
        pixels.append(inside)
        cells.append(group[inside] * num_grid + index[inside])
    pixel = np.concatenate(pixels)
    cell = np.concatenate(cells)
    offset = altitude[pixel] - grid[cell % num_grid]
    weight = 1.0 - np.abs(offset) / half_width

    def total(contribution):
        return np.bincount(cell, weights=contribution, minlength=size)

    s0, s1, s2 = total(weight), total(weight * offset), total(weight * offset**2)
    determinant = s0 * s2 - s1**2
    bracketed = (total(offset <= 0) > 0) & (total(offset >= 0) > 0)
    # All pixels at one altitude: the line's slope is undefined, use the mean
    sloped = determinant > 1e-12 * s0**2 * half_width**2
    with np.errstate(invalid="ignore", divide="ignore"):
        local_linear = weight * (s2[cell] - offset * s1[cell]) / determinant[cell]
        coefficient = np.where(sloped[cell], local_linear, weight / s0[cell])

    result = {name: total(coefficient * value[pixel]) for name, value in values.items()}
    result["variance"] = total(coefficient**2 * variance[pixel])
    for name in (*values, "variance"):
        result[name] = np.where(bracketed, result[name], np.nan)
    result["count"] = total(np.ones_like(weight))
    result["count_saturated"] = total(saturated[pixel].astype(float))
    return result


def _demodulate(m, a, b, variance):
    """Weighted least-squares I, Q, U per cell, and the L1B quantities.

    Inputs have shape (bins, analyzers, altitudes). Cells where any analyzer
    has no pixels are NaN.
    """
    # Move analyzers last: (bins, altitudes, analyzers)
    m, a, b, variance = (np.moveaxis(x, 1, -1) for x in (m, a, b, variance))
    design = np.stack([np.ones_like(a), a, b], axis=-1)
    valid = np.all(np.isfinite(design), axis=(-1, -2)) & np.all(
        np.isfinite(m) & (variance > 0), axis=-1
    )

    # Invalid cells get a well-posed placeholder system and are set to NaN at the end
    weight = np.where(valid[..., np.newaxis], 1.0 / np.where(variance > 0, variance, 1.0), 1.0)
    design = np.where(valid[..., np.newaxis, np.newaxis], design, np.eye(3)[: design.shape[-2]])
    m = np.where(valid[..., np.newaxis], m, 0.0)

    normal = np.einsum("...ki,...k,...kj->...ij", design, weight, design)
    covariance = np.linalg.inv(normal)
    stokes = np.einsum("...ij,...kj,...k,...k->...i", covariance, design, weight, m)

    # Placeholder cells have I = 0; they are set to NaN below
    I = np.where(valid, stokes[..., 0], 1.0)  # noqa: N806
    Q, U = stokes[..., 1], stokes[..., 2]  # noqa: N806
    q, u = Q / I, U / I
    # Jacobian of (q, u) with respect to (I, Q, U)
    jacobian = np.zeros((*I.shape, 2, 3))
    jacobian[..., 0, 0] = -Q / I**2
    jacobian[..., 0, 1] = 1 / I
    jacobian[..., 1, 0] = -U / I**2
    jacobian[..., 1, 2] = 1 / I
    qu_covariance = jacobian @ covariance @ np.swapaxes(jacobian, -1, -2)
    var_q, var_u, cov_qu = qu_covariance[..., 0, 0], qu_covariance[..., 1, 1], qu_covariance[..., 0, 1]

    dolp = np.hypot(q, u)
    # Linearized at dolp > 0; at dolp = 0 the derivative is undefined, so use the
    # mean of the q and u variances
    with np.errstate(invalid="ignore", divide="ignore"):
        var_dolp = np.where(
            dolp > 0,
            (q**2 * var_q + u**2 * var_u + 2 * q * u * cov_qu) / dolp**2,
            0.5 * (var_q + var_u),
        )

    results = {
        "radiance": I,
        "radiance_noise": np.sqrt(covariance[..., 0, 0]),
        "q": q,
        "q_noise": np.sqrt(var_q),
        "u": u,
        "u_noise": np.sqrt(var_u),
        "dolp": dolp,
        "dolp_noise": np.sqrt(var_dolp),
    }
    with np.errstate(invalid="ignore", divide="ignore"):
        results["q_u_correlation"] = cov_qu / np.sqrt(var_q * var_u)
    return {name: np.where(valid, value, np.nan) for name, value in results.items()}

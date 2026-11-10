"""Forward model: band-averaged polarized radiance to L1A detector frames.

Implements the ``model.forward`` section of the calibration database schema.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from aliprocessing.geometry import InstrumentFrame, LocalSphere
from aliprocessing.instrument import (
    electron_rate_per_radiance,
    instrument_sensitivities,
    pixel_array,
    rotate_sensitivities,
    saturation_counts,
)
from aliprocessing.schemas import apply_metadata, check

RADIANCE_UNITS = "W m-2 sr-1 nm-1"


def simulate_l1a(
    cal_db: xr.Dataset,
    scene: xr.Dataset,
    frames: xr.Dataset,
    *,
    noise: bool = True,
    seed: int | None = None,
) -> xr.DataTree:
    """Simulate L1A frames.

    Parameters
    ----------
    cal_db
        Calibration database.
    scene
        Band-averaged Stokes radiance ``I``, ``Q`` and ``U`` in the Stokes frame,
        with dims (spectral_channel, altitude), one entry per calibration database
        channel in its order, and an ``altitude`` coordinate in metres. Each
        pixel sees the scene at its tangent altitude, interpolated linearly. The
        scene must cover the altitudes L1A→L1B uses, the L1B altitude grid
        widened by the kernel half-width; beyond its ends, which only pixels
        that L1B ignores see, the end values are held.
    frames
        Everything L1A records about each frame except ``image`` and
        ``saturated``, for example from ``observation_sequence``.
    noise
        Apply shot noise, read noise and quantization. Without noise the image
        is the expected signal, still limited by saturation.
    seed
        Seed for the noise realization.

    Returns
    -------
    xr.DataTree
        The L1A granule, with the noise-free image in the ``simulation`` group.
    """
    rng = np.random.default_rng(seed)
    analyzer = pixel_array(cal_db, "pixel_analyzer")
    used = analyzer >= 0
    azimuth = pixel_array(cal_db, "pixel_azimuth_angle")[used]
    elevation = pixel_array(cal_db, "pixel_elevation_angle")[used]

    gain = float(cal_db["detector_gain"])
    read_noise = float(cal_db["detector_read_noise"])
    full_well = float(cal_db["detector_full_well"])
    adc_max = saturation_counts(cal_db)
    offset = pixel_array(cal_db, "dark_offset")
    dark_rate = (
        pixel_array(cal_db, "dark_current")
        if "dark_current" in cal_db
        else np.zeros_like(offset)
    )
    scene = _checked_scene(scene, cal_db)
    scene_altitude = scene["altitude"].to_numpy()
    stokes = {
        name: scene[name].transpose("spectral_channel", "altitude").to_numpy()
        for name in ("I", "Q", "U")
    }

    images, noiseless, saturated = [], [], []
    calibration = {}
    for i in range(frames.sizes["time"]):
        frame = frames.isel(time=i)
        exposure_time = float(frame["exposure_time"])
        num_exposures = int(frame.get("num_averaged_exposures", 1))
        channel = int(frame["spectral_channel"])

        expected = exposure_time * dark_rate * gain
        if channel >= 0:
            instrument = InstrumentFrame.from_l1a_frame(frame)
            looks = instrument.look_vectors(azimuth, elevation)
            sphere = LocalSphere.at_tangent_point(
                instrument.observer, instrument.boresight
            )
            tangent_altitude = sphere.tangent_altitude(instrument.observer, looks)
            I, Q, U = (  # noqa: E741
                np.interp(tangent_altitude, scene_altitude, stokes[name][channel])
                for name in ("I", "Q", "U")
            )
            if channel not in calibration:
                a, b = instrument_sensitivities(cal_db, channel)
                rate = electron_rate_per_radiance(cal_db, channel)
                calibration[channel] = (a[used], b[used], rate[used])
            a, b, rate = calibration[channel]
            a, b = rotate_sensitivities(a, b, instrument.roll_rad)
            signal = I + a * Q + b * U
            expected = expected.copy()
            expected[used] += exposure_time * rate * signal

        noiseless.append(offset + expected / gain)
        if noise:
            total = np.zeros_like(offset)
            frame_saturated = np.zeros(offset.shape, dtype=bool)
            for _ in range(num_exposures):
                electrons = rng.poisson(expected) + rng.normal(
                    0.0, read_noise, offset.shape
                )
                counts, exposure_saturated = _digitize(
                    electrons, offset, gain, full_well, adc_max
                )
                counts = np.round(counts)
                total += counts
                frame_saturated |= exposure_saturated
            images.append(total / num_exposures)
        else:
            counts, frame_saturated = _digitize(
                expected, offset, gain, full_well, adc_max
            )
            images.append(counts)
        saturated.append(frame_saturated)

    image_dims = ["time", "detector_row", "detector_column"]
    l1a = frames.copy()
    l1a["image"] = (image_dims, np.stack(images))
    l1a["saturated"] = (image_dims, np.stack(saturated))
    simulation = xr.Dataset(
        {
            "image_noiseless": (image_dims, np.stack(noiseless)),
            "noise_enabled": ((), noise),
        }
    )
    if seed is not None:
        simulation["noise_seed"] = ((), np.int64(seed))
    tree = apply_metadata(
        xr.DataTree.from_dict({"/": l1a, "/simulation": simulation}), "l1a"
    )
    check(tree, "l1a", cal_db=cal_db)
    return tree


def _checked_scene(scene: xr.Dataset, cal_db: xr.Dataset) -> xr.Dataset:
    """The scene sorted by altitude, after checking its channels and altitudes."""
    num_channels = cal_db.sizes["spectral_channel"]
    if scene.sizes.get("spectral_channel") != num_channels:
        msg = (
            f"the scene has {scene.sizes.get('spectral_channel')} spectral channels, "
            f"the calibration database {num_channels}"
        )
        raise ValueError(msg)
    scene = scene.sortby("altitude")
    altitude = scene["altitude"].to_numpy()
    if not np.all(np.diff(altitude) > 0):
        msg = "scene altitudes must be distinct"
        raise ValueError(msg)

    half_width = float(cal_db["altitude_kernel_half_width"])
    needed = (
        float(cal_db["altitude"].min()) - half_width,
        float(cal_db["altitude"].max()) + half_width,
    )
    if altitude[0] > needed[0] or altitude[-1] < needed[1]:
        msg = (
            f"the scene covers {altitude[0]:.0f} to {altitude[-1]:.0f} m, but "
            f"L1A->L1B uses tangent altitudes from {needed[0]:.0f} to {needed[1]:.0f} m"
        )
        raise ValueError(msg)
    return scene


def _digitize(electrons, offset, gain, full_well, adc_max):
    """Counts and saturation flags for one exposure's electrons."""
    full = electrons >= full_well
    counts = offset + np.clip(electrons, None, full_well) / gain
    adc_full = counts >= adc_max
    return np.clip(counts, 0, adc_max), full | adc_full


def observation_sequence(
    cal_db: xr.Dataset,
    *,
    start: str | np.datetime64,
    num_sets: int,
    exposure_time: float | Sequence[float],
    spacecraft_latitude: float,
    spacecraft_longitude: float,
    spacecraft_altitude: float,
    boresight_azimuth: float,
    boresight_elevation: float,
    boresight_roll: float = 0.0,
    num_averaged_exposures: int = 1,
    frame_interval: float = 0.5,
    blank_between_sets: bool = True,
) -> xr.Dataset:
    """Frames stepping through every spectral channel ``num_sets`` times.

    A blank frame follows each observation set when ``blank_between_sets``. The
    spacecraft position and attitude are held fixed. ``exposure_time`` is one
    value or one per spectral channel; blank frames use the longest.
    """
    num_channels = cal_db.sizes["spectral_channel"]
    exposures = np.broadcast_to(np.asarray(exposure_time, dtype=float), (num_channels,))

    channels, sets = [], []
    for s in range(num_sets):
        channels += list(range(num_channels))
        sets += [s] * num_channels
        if blank_between_sets:
            channels.append(-1)
            sets.append(-1)
    channels, sets = np.array(channels), np.array(sets)
    num_frames = len(channels)

    interval = np.timedelta64(round(frame_interval * 1e9), "ns")
    time = np.datetime64(start, "ns") + np.arange(num_frames) * interval

    def constant(value):
        return ("time", np.full(num_frames, value, dtype=float))

    frames = xr.Dataset(
        {
            "spectral_channel": ("time", channels),
            "observation_set": ("time", sets),
            "exposure_time": (
                "time",
                np.where(
                    channels >= 0, exposures[np.maximum(channels, 0)], exposures.max()
                ),
            ),
            "num_averaged_exposures": (
                "time",
                np.full(num_frames, num_averaged_exposures),
            ),
            "spacecraft_latitude": constant(spacecraft_latitude),
            "spacecraft_longitude": constant(spacecraft_longitude),
            "spacecraft_altitude": constant(spacecraft_altitude),
            "boresight_azimuth_angle": constant(boresight_azimuth % 360.0),
            "boresight_elevation_angle": constant(boresight_elevation),
            "boresight_roll_angle": constant(boresight_roll),
        },
        coords={"time": time},
    )
    return apply_metadata(frames, "l1a")


def exposure_for_fill(
    cal_db: xr.Dataset, scene: xr.Dataset, fill: float = 0.6
) -> np.ndarray:
    """Exposure time per spectral channel that fills the brightest pixel to ``fill``.

    ``fill`` is a fraction of the saturation level, the lower of the full well and
    the ADC maximum. The brightest pixel is estimated from the largest I · (1 +
    dolp) of the scene within the L1B altitude grid and the largest pixel response.
    """
    gain = float(cal_db["detector_gain"])
    capacity = min(
        float(cal_db["detector_full_well"]),
        (saturation_counts(cal_db) - np.nanmax(pixel_array(cal_db, "dark_offset")))
        * gain,
    )
    grid = cal_db["altitude"].to_numpy()
    in_grid = _checked_scene(scene, cal_db).sel(altitude=slice(grid.min(), grid.max()))
    if in_grid.sizes["altitude"] == 0:
        msg = "the scene has no altitudes within the L1B altitude grid"
        raise ValueError(msg)
    intensity = in_grid["I"].transpose("spectral_channel", "altitude").to_numpy()
    polarized = (
        np.hypot(in_grid["Q"], in_grid["U"])
        .transpose("spectral_channel", "altitude")
        .to_numpy()
    )
    brightest = np.max(intensity + polarized, axis=-1)

    exposures = []
    for channel in range(cal_db.sizes["spectral_channel"]):
        peak_rate = (
            np.nanmax(electron_rate_per_radiance(cal_db, channel)) * brightest[channel]
        )
        dark_rate = (
            np.nanmax(pixel_array(cal_db, "dark_current")) * gain
            if "dark_current" in cal_db
            else 0.0
        )
        exposures.append(fill * capacity / (peak_rate + dark_rate))
    return np.array(exposures)


def exponential_scene(
    cal_db: xr.Dataset,
    *,
    radiance_at_zero: float | Sequence[float] = 0.05,
    scale_height: float = 7e3,
    q: float = 0.1,
    u: float = 0.02,
    altitude: np.ndarray | None = None,
) -> xr.Dataset:
    """A limb scene with exponential radiance and constant q and u.

    ``radiance_at_zero`` is one value or one per spectral channel, in
    W m-2 sr-1 nm-1.
    """
    altitude = np.arange(-5e3, 100.01e3, 100.0) if altitude is None else altitude
    num_channels = cal_db.sizes["spectral_channel"]
    radiance = np.broadcast_to(
        np.asarray(radiance_at_zero, dtype=float), (num_channels,)
    )
    intensity = radiance[:, np.newaxis] * np.exp(-altitude / scale_height)
    dims = ["spectral_channel", "altitude"]
    return xr.Dataset(
        {
            "I": (dims, intensity, {"units": RADIANCE_UNITS}),
            "Q": (dims, q * intensity, {"units": RADIANCE_UNITS}),
            "U": (dims, u * intensity, {"units": RADIANCE_UNITS}),
        },
        coords={"altitude": ("altitude", altitude, {"units": "m"})},
    )

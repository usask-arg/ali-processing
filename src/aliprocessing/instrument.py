"""Pieces of the calibration database model shared by the forward model and L1A→L1B.

See the ``model`` section of the calibration database schema for the equations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

PLANCK = 6.62607015e-34
SPEED_OF_LIGHT = 299792458.0

# A dark estimate for one L1A frame: (frame, cal_db) -> counts per pixel
DarkModel = Callable[[xr.Dataset, xr.Dataset], np.ndarray]

_PIXEL_DIMS = ("detector_row", "detector_column")


def pixel_array(cal_db: xr.Dataset, name: str) -> np.ndarray:
    """A per-pixel calibration variable with the detector dimensions last."""
    var = cal_db[name]
    leading = [dim for dim in var.dims if dim not in _PIXEL_DIMS]
    return var.transpose(*leading, *_PIXEL_DIMS).to_numpy()


def photon_weighted_throughput(cal_db: xr.Dataset) -> np.ndarray:
    """Λ_w = ∫ τ_w(λ) λ dλ / h c for each channel, in photons J-1 nm.

    Multiplied by etendue and a band-averaged spectral radiance it gives a
    photon (electron) rate.
    """
    wavelength_nm = cal_db["throughput_wavelength"].to_numpy()
    throughput = (
        cal_db["spectral_throughput"]
        .transpose("spectral_channel", "throughput_wavelength")
        .to_numpy()
    )
    photons_per_joule = wavelength_nm * 1e-9 / (PLANCK * SPEED_OF_LIGHT)
    return np.trapezoid(throughput * photons_per_joule, wavelength_nm, axis=-1)


def electron_rate_per_radiance(cal_db: xr.Dataset, channel: int) -> np.ndarray:
    """E_p · T_k(w) · Λ_w: electrons per second per unit band-averaged radiance.

    NaN for pixels outside every sub-image.
    """
    analyzer = pixel_array(cal_db, "pixel_analyzer")
    transmission = (
        cal_db["analyzer_transmission"]
        .transpose("spectral_channel", "analyzer")
        .to_numpy()[channel]
    )
    pixel_transmission = np.where(
        analyzer >= 0, transmission[np.maximum(analyzer, 0)], np.nan
    )
    return (
        pixel_array(cal_db, "pixel_etendue")
        * pixel_transmission
        * photon_weighted_throughput(cal_db)[channel]
    )


def instrument_sensitivities(
    cal_db: xr.Dataset, channel: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's response to Q and U in the instrument frame, a and b."""
    return tuple(
        cal_db[name].isel(spectral_channel=channel).transpose(*_PIXEL_DIMS).to_numpy()
        for name in ("q_sensitivity", "u_sensitivity")
    )


def rotate_sensitivities(
    a: np.ndarray, b: np.ndarray, roll_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate instrument-frame sensitivities into the Stokes frame for a roll.

    The roll turns the instrument up axis clockwise from the vertical, so an
    analyzer at theta is at theta + roll in the Stokes frame.
    """
    cos2, sin2 = np.cos(2 * roll_rad), np.sin(2 * roll_rad)
    return a * cos2 - b * sin2, a * sin2 + b * cos2


def stokes_sensitivities(
    cal_db: xr.Dataset, channel: int, roll_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each pixel's response to Q and U in the Stokes frame, a' and b'."""
    return rotate_sensitivities(*instrument_sensitivities(cal_db, channel), roll_rad)


def across_track_bin(azimuth: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Across-track bin of each pixel azimuth, or -1 outside every bin.

    Bins are half-open, [edges[j], edges[j + 1]). NaN azimuths are outside.
    """
    index = np.digitize(azimuth, edges) - 1
    return np.where((index >= 0) & (index < len(edges) - 1), index, -1)


def static_dark(frame: xr.Dataset, cal_db: xr.Dataset) -> np.ndarray:
    """dark_offset + t · dark_current, the default dark estimate."""
    exposure_time = float(frame["exposure_time"])
    dark = pixel_array(cal_db, "dark_offset")
    if "dark_current" in cal_db:
        dark = dark + exposure_time * pixel_array(cal_db, "dark_current")
    return dark


def detector_variance(
    image: np.ndarray, cal_db: xr.Dataset, num_averaged_exposures: int
) -> np.ndarray:
    """Variance of a mean of exposures, in counts²: shot, read and quantization noise.

    Shot noise covers the signal and dark electrons, (S - o) / g.
    """
    gain = float(cal_db["detector_gain"])
    read_noise = float(cal_db["detector_read_noise"])
    electrons_in_counts = np.clip(image - pixel_array(cal_db, "dark_offset"), 0, None)
    variance = electrons_in_counts / gain + (read_noise / gain) ** 2 + 1 / 12
    return variance / num_averaged_exposures


def saturation_counts(cal_db: xr.Dataset) -> float:
    """Highest count the ADC can report."""
    return float(2 ** int(cal_db["detector_adc_bits"]) - 1)

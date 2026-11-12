"""Pieces of the calibration database model shared by the forward model and L1A→L1B.

See the ``model`` section of the calibration database schema for the equations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

PLANCK = 6.62607015e-34
SPEED_OF_LIGHT = 299792458.0

# Throughput below this fraction of a channel's peak is treated as zero when
# checking that a spectrum covers the passbands
_NEGLIGIBLE_THROUGHPUT = 1e-6
_WAVELENGTH_TOLERANCE_NM = 1e-6

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


def channel_at_wavelength(
    cal_db: xr.Dataset, wavelengths_nm: np.ndarray, tolerance_nm: float = 0.01
) -> np.ndarray:
    """The spectral channel whose centre wavelength each wavelength is.

    Raises ValueError for a wavelength more than ``tolerance_nm`` from every
    channel centre.
    """
    centers = cal_db["center_wavelength"].to_numpy()
    wavelengths_nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    distance = np.abs(wavelengths_nm[:, np.newaxis] - centers)
    channel = np.argmin(distance, axis=1)
    unmatched = distance[np.arange(len(wavelengths_nm)), channel] > tolerance_nm
    if np.any(unmatched):
        msg = (
            f"wavelengths {wavelengths_nm[unmatched]} are not channel centres "
            f"{centers}"
        )
        raise ValueError(msg)
    return channel


def band_average(
    cal_db: xr.Dataset, wavelength_nm: np.ndarray, spectrum: np.ndarray
) -> np.ndarray:
    """Photon-weighted band average of a spectrum for each spectral channel.

    X_w = ∫ τ_w(λ) λ X(λ) dλ / ∫ τ_w(λ) λ dλ, the definition the calibration
    database uses for band-averaged radiance. The integrals are evaluated on the
    spectrum's own wavelength grid, which must resolve the passbands and cover
    every wavelength where spectral_throughput is greater than 0.

    Parameters
    ----------
    cal_db
        Calibration database.
    wavelength_nm
        Increasing wavelengths of the spectrum, in nm.
    spectrum
        Values with wavelength as the last dimension.

    Returns
    -------
    np.ndarray
        Shape (spectral_channel, *spectrum.shape[:-1]).
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    throughput_wavelength = cal_db["throughput_wavelength"].to_numpy()
    throughput = (
        cal_db["spectral_throughput"]
        .transpose("spectral_channel", "throughput_wavelength")
        .to_numpy()
    )

    # Coverage is needed where the throughput matters, not out in the tails of
    # passbands that never reach exactly zero
    significant = throughput > _NEGLIGIBLE_THROUGHPUT * throughput.max(
        axis=1, keepdims=True
    )
    passband = throughput_wavelength[np.any(significant, axis=0)]
    if (
        wavelength_nm[0] > passband.min() + _WAVELENGTH_TOLERANCE_NM
        or wavelength_nm[-1] < passband.max() - _WAVELENGTH_TOLERANCE_NM
    ):
        msg = (
            f"the spectrum covers {wavelength_nm[0]} to {wavelength_nm[-1]} nm, but "
            f"the passbands span {passband.min()} to {passband.max()} nm"
        )
        raise ValueError(msg)

    inside = (wavelength_nm >= throughput_wavelength[0]) & (
        wavelength_nm <= throughput_wavelength[-1]
    )
    wavelength_nm = wavelength_nm[inside]
    spectrum = spectrum[..., inside]
    averages = []
    for channel_throughput in throughput:
        weight = (
            np.interp(wavelength_nm, throughput_wavelength, channel_throughput)
            * wavelength_nm
        )
        averages.append(
            np.trapezoid(weight * spectrum, wavelength_nm, axis=-1)
            / np.trapezoid(weight, wavelength_nm)
        )
    return np.stack(averages)


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

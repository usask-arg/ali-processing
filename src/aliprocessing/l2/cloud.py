"""Cloud detection and the retrieval lower bound for the two-stage L2 retrieval.

Two signatures flag a cloud top, each computed on the L1B tangent altitudes of
one image:

- Colour: the spectral difference of vertical log-radiance gradients between a
  short and a long channel, times the stage-1 extinction, as in the OSIRIS L2
  cloud screen. Below a cloud top, radiance at the long channel rises faster
  with decreasing altitude than at the short one, while the stage-1 retrieval
  puts the cloud's extinction there.
- DoLP: the vertical gradient of ln(DoLP) at a long channel, from the
  measurements alone. Clouds depolarize, and at long wavelengths, where
  Rayleigh scattering is weak, DoLP collapses below a cloud top.

A level is cloudy when either score, scaled by a penalty above the tropopause
(as in OMPS-LP), exceeds its threshold, between the stage-1 lower bound and a
fixed ceiling. The cloud top is
the highest cloudy level, and the stage-2 lower bound is that plus padding, but
never below the stage-1 lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class CloudScreen:
    """Cloud diagnostics on the image's tangent altitudes, and the lower bound."""

    altitude: np.ndarray
    colour_index: np.ndarray
    colour_score: np.ndarray
    dolp_gradient: np.ndarray
    tropopause_penalty: np.ndarray
    colour_mask: np.ndarray
    dolp_mask: np.ndarray
    cloud_mask: np.ndarray
    cloud_top_altitude: float
    lower_altitude: float
    colour_wavelengths_nm: tuple[float, float]
    dolp_wavelength_nm: float

    def to_dataset(self) -> xr.Dataset:
        """The diagnostics as variables on a ``cloud_altitude`` dimension."""
        dim = "cloud_altitude"
        return xr.Dataset(
            {
                "cloud_colour_index": (dim, self.colour_index, {"units": "km-1"}),
                "cloud_colour_score": (dim, self.colour_score, {"units": "km-2"}),
                "cloud_dolp_gradient": (dim, self.dolp_gradient, {"units": "km-1"}),
                "cloud_tropopause_penalty": (dim, self.tropopause_penalty),
                "cloud_colour_mask": (dim, self.colour_mask.astype(np.int8)),
                "cloud_dolp_mask": (dim, self.dolp_mask.astype(np.int8)),
                "cloud_mask": (dim, self.cloud_mask.astype(np.int8)),
                "cloud_top_altitude": ((), self.cloud_top_altitude, {"units": "m"}),
                "retrieval_lower_altitude": ((), self.lower_altitude, {"units": "m"}),
            },
            coords={dim: (dim, self.altitude, {"units": "m"})},
            attrs={
                "cloud_colour_wavelengths_nm": list(self.colour_wavelengths_nm),
                "cloud_dolp_wavelength_nm": self.dolp_wavelength_nm,
            },
        )


def screen_clouds(
    radiance: xr.Dataset,
    dolp: xr.Dataset | None,
    extinction: xr.DataArray,
    settings: dict,
    stage1_lower_altitude: float,
    tropopause_altitude: float | None = None,
) -> CloudScreen:
    """Detect clouds in one image and set the stage-2 lower bound.

    Parameters
    ----------
    radiance, dolp
        The image's ``"I"`` and ``"dolp"`` spectra: ``radiance`` with dims
        (wavelength, los), ``tangent_altitude`` and ``sample_wavelengths_nm``.
        Without ``dolp`` only the colour signature is used.
    extinction
        Stage-1 aerosol extinction [m-1] on its retrieval altitude grid [m].
    settings
        The ``two_stage`` section of the retrieval configuration.
    stage1_lower_altitude
        Lowest tangent altitude of stage 1 [m]; the lower bound never goes below.
    tropopause_altitude
        Tropopause altitude [m], or None to apply no tropopause penalty.
    """
    cloud = settings["cloud"]
    altitude = radiance["tangent_altitude"].to_numpy()
    order = np.argsort(altitude)
    altitude = altitude[order]
    altitude_km = altitude / 1e3
    wavelengths = radiance["sample_wavelengths_nm"].to_numpy()
    intensity = radiance["radiance"].transpose("wavelength", "los").to_numpy()[:, order]

    short, long = _channels(wavelengths, cloud["colour_wavelengths_nm"])
    log_gradient = np.gradient(np.log(intensity), altitude_km, axis=1)
    colour_index = -(log_gradient[long] - log_gradient[short])
    extinction_per_km = 1e3 * np.interp(
        altitude, extinction[extinction.dims[0]].to_numpy(), extinction.to_numpy()
    )
    colour_score = colour_index * extinction_per_km

    dolp_channel = (
        int(np.argmax(wavelengths))
        if cloud["dolp_wavelength_nm"] is None
        else _nearest(wavelengths, cloud["dolp_wavelength_nm"])
    )
    if dolp is None:
        dolp_gradient = np.full(altitude.size, np.nan)
    else:
        polarization = (
            dolp["radiance"].transpose("wavelength", "los").to_numpy()[:, order]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            dolp_gradient = np.gradient(
                np.log(polarization[dolp_channel]), altitude_km, axis=0
            )

    penalty = tropopause_penalty(
        altitude, tropopause_altitude, cloud["tropopause_penalty_scale_m"]
    )
    # Only within stage 1's window: lower down the surface also changes DoLP
    searched = (altitude >= stage1_lower_altitude) & (
        altitude <= cloud["max_altitude_m"]
    )
    with np.errstate(invalid="ignore"):
        colour_mask = searched & (penalty * colour_score > cloud["colour_threshold"])
        dolp_mask = searched & (
            penalty * dolp_gradient > cloud["dolp_gradient_threshold"]
        )
    cloud_mask = colour_mask | dolp_mask

    cloud_top = float(altitude[cloud_mask].max()) if cloud_mask.any() else np.nan
    lower = stage1_lower_altitude
    if np.isfinite(cloud_top):
        lower = max(lower, cloud_top + settings["cloud_padding_m"])

    return CloudScreen(
        altitude=altitude,
        colour_index=colour_index,
        colour_score=colour_score,
        dolp_gradient=dolp_gradient,
        tropopause_penalty=penalty,
        colour_mask=colour_mask,
        dolp_mask=dolp_mask,
        cloud_mask=cloud_mask,
        cloud_top_altitude=cloud_top,
        lower_altitude=float(lower),
        colour_wavelengths_nm=(float(wavelengths[short]), float(wavelengths[long])),
        dolp_wavelength_nm=(
            float(wavelengths[dolp_channel]) if dolp is not None else np.nan
        ),
    )


def tropopause_penalty(
    altitude: np.ndarray, tropopause_altitude: float | None, scale: float | None
) -> np.ndarray:
    """exp(-0.5 (max(z - z_trop, 0) / scale)^2), or 1 without a tropopause or scale."""
    if tropopause_altitude is None or not scale or not np.isfinite(tropopause_altitude):
        return np.ones_like(altitude, dtype=float)
    above = np.maximum(np.asarray(altitude, dtype=float) - tropopause_altitude, 0.0)
    return np.exp(-0.5 * (above / scale) ** 2)


def _channels(wavelengths: np.ndarray, requested) -> tuple[int, int]:
    if requested is None:
        return int(np.argmin(wavelengths)), int(np.argmax(wavelengths))
    short, long = (_nearest(wavelengths, w) for w in requested)
    if wavelengths[short] >= wavelengths[long]:
        msg = f"colour_wavelengths_nm {requested} must be [short, long]"
        raise ValueError(msg)
    return short, long


def _nearest(wavelengths: np.ndarray, wavelength: float) -> int:
    return int(np.argmin(np.abs(wavelengths - wavelength)))

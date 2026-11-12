"""Cloud detection and the stage-2 lower bound, on synthetic profiles."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aliprocessing.l2.cloud import screen_clouds, tropopause_penalty
from aliprocessing.l2.config import load_retrieval_config

ALTITUDE = np.arange(4e3, 46001.0, 500.0)
WAVELENGTHS = np.array([525.0, 1560.0])
SETTINGS = load_retrieval_config()["two_stage"]
FLOOR = 5000.0


def _below(top, width_km=0.3):
    """1 below ``top``, 0 above, with a smooth edge."""
    return 1 / (1 + np.exp((ALTITUDE - top) / (width_km * 1e3)))


def _spectra(cloud_top=None, colour=True, depolarize=True):
    """The "I" and "dolp" spectra of one image, optionally with a cloud.

    Below the cloud top the radiance brightens more at the long channel than at
    the short one, and DoLP at the long channel collapses.
    """
    clear = np.exp(-ALTITUDE / 7e3)
    brightening = np.zeros((2, ALTITUDE.size))
    dolp = np.stack([np.full(ALTITUDE.size, 0.36), np.full(ALTITUDE.size, 0.47)])
    if cloud_top is not None:
        below = _below(cloud_top)
        if colour:
            brightening = np.stack([2 * below, 20 * below])
        if depolarize:
            dolp[1] = 0.47 * (1 - 0.95 * below)
    radiance = clear * (1 + brightening)

    def spectra(values):
        return xr.Dataset(
            {
                "radiance": (["wavelength", "los"], values),
                "tangent_altitude": ("los", ALTITUDE),
                "sample_wavelengths_nm": ("wavelength", WAVELENGTHS),
            }
        )

    return spectra(radiance), spectra(dolp)


def _extinction(cloud_top=None):
    """Stage-1 extinction: background aerosol, plus the cloud below its top,
    which stage 1 fits as aerosol."""
    grid = np.arange(0.0, 65001.0, 1000.0)
    per_km = np.full(grid.size, 1e-3)
    if cloud_top is not None:
        per_km += 0.05 / (1 + np.exp((grid - cloud_top) / 300.0))
    return xr.DataArray(per_km / 1e3, dims="altitude", coords={"altitude": grid})


def _screen(cloud_top=None, settings=SETTINGS, tropopause=None, **kwargs):
    radiance, dolp = _spectra(cloud_top, **kwargs)
    return screen_clouds(
        radiance, dolp, _extinction(cloud_top), settings, FLOOR, tropopause
    )


def test_clear_sky_has_no_cloud():
    screen = _screen()
    assert not screen.cloud_mask.any()
    assert np.isnan(screen.cloud_top_altitude)
    assert screen.lower_altitude == FLOOR


def test_cloud_raises_the_lower_bound():
    screen = _screen(cloud_top=12e3)
    assert screen.colour_mask.any()
    assert screen.dolp_mask.any()
    # The highest flagged level sits at or a little above the true top, where
    # the edge's gradient still crosses the threshold
    assert 12e3 <= screen.cloud_top_altitude <= 13.5e3
    assert (
        screen.lower_altitude == screen.cloud_top_altitude + SETTINGS["cloud_padding_m"]
    )


@pytest.mark.parametrize(
    ("colour", "depolarize", "fired"),
    [(True, False, "colour_mask"), (False, True, "dolp_mask")],
)
def test_either_signature_detects_a_cloud(colour, depolarize, fired):
    screen = _screen(cloud_top=12e3, colour=colour, depolarize=depolarize)
    other = "dolp_mask" if fired == "colour_mask" else "colour_mask"
    assert getattr(screen, fired).any()
    assert not getattr(screen, other).any()
    assert np.isfinite(screen.cloud_top_altitude)


def test_cloud_below_the_stage1_window_is_ignored():
    screen = _screen(cloud_top=3e3)
    assert not screen.cloud_mask.any()
    assert screen.lower_altitude == FLOOR


def test_nothing_is_detected_above_the_ceiling():
    screen = _screen(cloud_top=25e3)
    assert not screen.cloud_mask.any()


def test_tropopause_penalty_suppresses_clouds_well_above_it():
    assert np.isfinite(_screen(cloud_top=18e3).cloud_top_altitude)
    assert np.isnan(_screen(cloud_top=18e3, tropopause=10e3).cloud_top_altitude)
    # Just above the tropopause the penalty is mild
    assert np.isfinite(_screen(cloud_top=12e3, tropopause=11.5e3).cloud_top_altitude)


def test_tropopause_penalty_values():
    altitude = np.array([8e3, 10e3, 12e3, 14e3])
    np.testing.assert_allclose(
        tropopause_penalty(altitude, 10e3, 2e3),
        [1.0, 1.0, np.exp(-0.5), np.exp(-2.0)],
    )
    np.testing.assert_array_equal(tropopause_penalty(altitude, None, 2e3), 1.0)
    np.testing.assert_array_equal(tropopause_penalty(altitude, 10e3, None), 1.0)


def test_signature_channels_can_be_chosen():
    settings = {
        **SETTINGS,
        "cloud": {**SETTINGS["cloud"], "colour_wavelengths_nm": [1560, 525]},
    }
    with pytest.raises(ValueError, match=r"\[short, long\]"):
        _screen(cloud_top=12e3, settings=settings)

    screen = _screen(cloud_top=12e3)
    assert screen.colour_wavelengths_nm == (525.0, 1560.0)
    assert screen.dolp_wavelength_nm == 1560.0


def test_diagnostics_as_a_dataset():
    ds = _screen(cloud_top=12e3).to_dataset()
    assert ds["cloud_mask"].dtype == np.int8
    assert ds.sizes["cloud_altitude"] == ALTITUDE.size
    assert float(ds["retrieval_lower_altitude"]) > float(ds["cloud_top_altitude"])
    assert ds.attrs["cloud_dolp_wavelength_nm"] == 1560.0

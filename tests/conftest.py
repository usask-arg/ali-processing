from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from aliprocessing.l1b.data import L1bSpectra

# Includes one line of sight below the default lower bound and one above the default upper bound
TANGENT_ALTITUDES = np.array([-1000.0, 5000.0, 20000.0, 40000.0, 120000.0])
WAVELENGTHS = np.array([745.0, 1020.0])


@pytest.fixture
def make_spectra():
    def _make_spectra(
        radiance: np.ndarray | None = None,
        tangent_altitude: np.ndarray = TANGENT_ALTITUDES,
        wavelengths: np.ndarray = WAVELENGTHS,
        sza: float = 60.0,
        saa: float = 90.0,
        time: datetime = datetime(2025, 1, 1),
    ) -> L1bSpectra:
        nlos = len(tangent_altitude)
        if radiance is None:
            radiance = np.arange(len(wavelengths) * nlos, dtype=float).reshape(
                len(wavelengths), nlos
            )

        return L1bSpectra.from_np_arrays(
            radiance=radiance,
            radiance_noise=np.abs(radiance) * 0.01,
            tangent_altitude=tangent_altitude,
            tangent_latitude=np.linspace(10.0, 14.0, nlos),
            tangent_longitude=np.linspace(100.0, 104.0, nlos),
            sample_wavelengths_nm=wavelengths,
            time=time,
            observer_latitude=1.0,
            observer_longitude=2.0,
            observer_altitude=500000.0,
            sza=np.full(nlos, sza),
            saa=np.full(nlos, saa),
            los_azimuth_angle=np.zeros(nlos),
        )

    return _make_spectra

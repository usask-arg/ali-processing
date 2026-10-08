from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from aliprocessing.l1b.data import L1bImage
from aliprocessing.processing.l1b_to_l2 import process_l1b_to_l2_image

TANGENT_ALTITUDES = np.arange(5000.0, 45001.0, 2500.0)


@pytest.fixture
def por_image():
    altitude = np.arange(0.0, 65001.0, 1000.0)
    pressure = 101325.0 * np.exp(-altitude / 7000.0)
    # NaN pressures are dropped before building the ancillary
    pressure[-1] = np.nan

    return xr.Dataset(
        {
            "pressure": ("altitude", pressure),
            "temperature": ("altitude", np.full(len(altitude), 250.0)),
        },
        coords={"altitude": altitude},
    )


@pytest.mark.parametrize("pol_state", ["dolp", "aolp", "q"])
def test_process_l1b_to_l2_image_smoke(make_spectra, por_image, pol_state):
    # Exercises the full retrieval pipeline for a couple of iterations; checks that everything is
    # wired together and the outputs are well formed, not that the retrieval converges
    nlos = len(TANGENT_ALTITUDES)
    intensity = (
        np.exp(-TANGENT_ALTITUDES / 7000.0)[None, :] * np.array([1.0, 0.5])[:, None]
    )
    image = L1bImage(
        {
            "I": make_spectra(radiance=intensity, tangent_altitude=TANGENT_ALTITUDES),
            pol_state: make_spectra(
                radiance=np.full((2, nlos), 0.3), tangent_altitude=TANGENT_ALTITUDES
            ),
        }
    )

    state = process_l1b_to_l2_image(
        image, por_image, None, minimizer_kwargs={"max_nfev": 2}
    )

    assert 1 <= int(state["num_iterations"]) <= 2
    assert np.isfinite(float(state["cost"]))

    np.testing.assert_allclose(float(state["latitude"]), 12.0)
    np.testing.assert_allclose(float(state["longitude"]), 102.0)
    np.testing.assert_allclose(float(state["solar_zenith_angle"]), 60.0)
    np.testing.assert_allclose(float(state["solar_azimuth_angle"]), 90.0)
    np.testing.assert_allclose(float(state["solar_scattering_angle"]), 90.0)

    for key in ["I", pol_state]:
        simulated = state[f"simulated_l1_{key}_radiance"]
        assert simulated.shape == (2, nlos)
        assert np.all(np.isfinite(simulated))

    for var in ["o3_vmr", "lambertian_albedo"]:
        assert np.all(np.isfinite(state[var]))

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr
from conftest import TANGENT_ALTITUDES, WAVELENGTHS
from skretrieval.core.radianceformat import RadianceGridded

from aliprocessing.l1b.data import L1bFileWriter, L1bImage

# Lines of sight strictly inside the default (0, 100000) m bounds
IN_BOUNDS = slice(1, 4)


def test_from_np_arrays_builds_dataset(make_spectra):
    ds = make_spectra().ds

    assert ds["radiance"].dims == ("wavelength", "los")
    assert ds["radiance"].shape == (len(WAVELENGTHS), len(TANGENT_ALTITUDES))
    for var in [
        "tangent_altitude",
        "tangent_latitude",
        "tangent_longitude",
        "solar_zenith_angle",
        "relative_solar_azimuth_angle",
        "los_azimuth_angle",
    ]:
        assert ds[var].dims == ("los",)

    np.testing.assert_array_equal(ds["sample_wavelengths_nm"], WAVELENGTHS)
    np.testing.assert_array_equal(ds["tangent_altitude"], TANGENT_ALTITUDES)
    assert float(ds["spacecraft_altitude"]) == 500000.0
    assert ds["time"].to_numpy() == np.datetime64("2025-01-01")


def test_skretrieval_l1_filters_out_of_bounds_altitudes(make_spectra):
    spectra = make_spectra()
    l1 = L1bImage({"I": spectra, "dolp": make_spectra()}).skretrieval_l1()

    assert set(l1) == {"I", "dolp"}
    assert isinstance(l1["I"], RadianceGridded)

    data = l1["I"].data
    np.testing.assert_array_equal(
        data["tangent_altitude"], TANGENT_ALTITUDES[IN_BOUNDS]
    )
    np.testing.assert_array_equal(data["wavelength"], WAVELENGTHS)
    np.testing.assert_array_equal(
        data["radiance"], spectra.ds["radiance"].to_numpy()[:, IN_BOUNDS]
    )
    np.testing.assert_array_equal(
        data["radiance_noise"], spectra.ds["radiance_noise"].to_numpy()[:, IN_BOUNDS]
    )

    # tangent_altitude is usable as an index into the los dimension
    selected = data["radiance"].sel(tangent_altitude=20000.0)
    np.testing.assert_array_equal(selected, spectra.ds["radiance"].to_numpy()[:, 2])


def test_altitude_bounds_are_exclusive(make_spectra):
    image = L1bImage({"I": make_spectra()}, low_alt=5000.0, high_alt=40000.0)

    data = image.skretrieval_l1()["I"].data

    np.testing.assert_array_equal(data["tangent_altitude"], [20000.0])


def test_sk2_geometry_has_one_ray_per_in_bounds_los(make_spectra):
    image = L1bImage({"I": make_spectra(), "dolp": make_spectra()})

    geometry = image.sk2_geometry()

    assert set(geometry) == {"I", "dolp"}
    for viewing_geo in geometry.values():
        assert len(viewing_geo.observer_rays) == len(TANGENT_ALTITUDES[IN_BOUNDS])


def test_reference_values(make_spectra):
    image = L1bImage({"I": make_spectra(sza=60.0)})

    np.testing.assert_array_equal(image.sample_wavelengths()["I"], WAVELENGTHS)
    np.testing.assert_allclose(float(image.reference_cos_sza()["I"]), 0.5)
    np.testing.assert_allclose(float(image.reference_latitude()["I"]), 12.0)
    np.testing.assert_allclose(float(image.reference_longitude()["I"]), 102.0)


def test_append_information_to_l1_adds_tangent_altitude_index(make_spectra):
    image = L1bImage({"I": make_spectra()})
    n_in_bounds = len(TANGENT_ALTITUDES[IN_BOUNDS])
    l1 = {
        "I": RadianceGridded(
            xr.Dataset(
                {
                    "radiance": (
                        ["wavelength", "los"],
                        np.ones((len(WAVELENGTHS), n_in_bounds)),
                    )
                },
                coords={"wavelength": WAVELENGTHS},
            )
        )
    }

    image.append_information_to_l1(l1)

    np.testing.assert_array_equal(
        l1["I"].data["tangent_altitude"], TANGENT_ALTITUDES[IN_BOUNDS]
    )
    assert "tangent_altitude" in l1["I"].data.indexes


def test_file_writer_round_trip(make_spectra, tmp_path):
    times = [datetime(2025, 1, 1), datetime(2025, 1, 2)]
    images = [
        L1bImage(
            {
                "I": make_spectra(time=t, radiance=np.full((2, 5), float(i + 1))),
                "dolp": make_spectra(time=t, radiance=np.full((2, 5), 0.1 * (i + 1))),
            }
        )
        for i, t in enumerate(times)
    ]
    out_file = tmp_path / "l1b.nc"

    L1bFileWriter(images).save(out_file)

    # Every group is preserved, not just the last one written
    for key, scale in [("I", 1.0), ("dolp", 0.1)]:
        with xr.open_dataset(out_file, group=key) as ds:
            assert ds.sizes["time"] == len(times)
            np.testing.assert_array_equal(
                ds["time"].to_numpy(), np.array(times, dtype="datetime64[ns]")
            )
            np.testing.assert_allclose(ds["radiance"].isel(time=0), scale * 1.0)
            np.testing.assert_allclose(ds["radiance"].isel(time=1), scale * 2.0)

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import skretrieval as skr
import xarray as xr
from skretrieval.core.lineshape import DeltaFunction
from skretrieval.retrieval.processing import Retrieval

from aliprocessing.l1b.data import L1bImage
from aliprocessing.l2.ancillary import Ancillary
from aliprocessing.l2.config import altitude_grid, engine_kwargs, load_retrieval_config
from aliprocessing.l2.forwardmodel import ALIForwardModel
from aliprocessing.l2.optical import aerosol_median_radius_db


@Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_property(*args, **kwargs):
    return aerosol_median_radius_db()


def process_l1b_to_l2_image(
    l1b_image: L1bImage,
    por_image: xr.Dataset,
    cal_db: xr.Dataset | Path,
    config: dict | str | os.PathLike | None = None,
):
    """Retrieve one L1B image.

    ``cal_db`` is the calibration database, or a path to it; its spectral
    response functions set the band-averaged solar irradiance of each channel.
    ``config`` holds the retrieval settings: overrides of retrieval_config.yaml
    as a dict or a YAML file, or a complete configuration from
    load_retrieval_config. None uses the defaults.
    """
    config = load_retrieval_config(config)
    if not isinstance(cal_db, xr.Dataset):
        cal_db = xr.open_dataset(cal_db)
    good = ~np.isnan(por_image.pressure.to_numpy())
    anc = Ancillary(
        por_image.altitude.to_numpy()[good],
        por_image.pressure.to_numpy()[good],
        por_image.temperature.to_numpy()[good],
        cal_db=cal_db,
        **config["ancillary"],
    )

    # Plain floats, skretrieval uses the triplet wavelengths as dictionary keys
    sample_wavel = l1b_image.sample_wavelengths()["I"].to_numpy()

    measurement = config["measurement"]
    meas_vec = {
        str(w): skr.measvec.Triplet(
            [w],
            [1],
            list(measurement["altitude_range_m"]),
            list(measurement["normalization_range_m"]),
            normalize=measurement["normalize"],
            log_space=measurement["log_space"],
        )
        for w in sample_wavel
    }

    minimizer_kwargs = dict(config["minimizer"])
    state_kwargs = copy.deepcopy(config["state"])
    albedo = state_kwargs.get("surface", {}).get("lambertian_albedo")
    if albedo is not None and albedo.get("wavelengths") is None:
        albedo["wavelengths"] = np.atleast_1d(sample_wavel)
    state_kwargs["altitude_grid"] = altitude_grid(config)

    ret = Retrieval(
        observation=l1b_image,
        forward_model_cfg={
            "*": {
                "kwargs": {
                    "lineshape_fn": lambda _: DeltaFunction(),
                    "spectral_native_coordinate": "wavelength_nm",
                    "model_res_nm": config["forward_model"]["model_res_nm"],
                    "pol_states": l1b_image.reference_cos_sza().keys(),
                },
                "class": ALIForwardModel,
            },
        },
        measvec=meas_vec,
        minimizer=minimizer_kwargs.pop("minimizer"),
        ancillary=anc,
        l1_kwargs={},
        model_kwargs=engine_kwargs(config, len(sample_wavel)),
        minimizer_kwargs=minimizer_kwargs,
        target_kwargs={},
        state_kwargs=state_kwargs,
    )

    results = ret.retrieve()

    ref_lat = l1b_image.reference_latitude()["I"]
    ref_lon = l1b_image.reference_longitude()["I"]

    results["state"]["latitude"] = ref_lat
    results["state"]["longitude"] = ref_lon

    results["state"]["num_iterations"] = results["minimizer"]["minimizer"]["nfev"]
    results["state"]["cost"] = results["minimizer"]["minimizer"]["cost"]

    results["state"]["solar_zenith_angle"] = np.rad2deg(
        np.arccos(l1b_image.reference_cos_sza()["I"])
    )
    results["state"]["solar_azimuth_angle"] = (
        l1b_image.spectra["I"]._ds["relative_solar_azimuth_angle"].mean()
    )

    cos_scatter = np.sin(np.deg2rad(results["state"]["solar_zenith_angle"])) * np.cos(
        np.deg2rad(results["state"]["solar_azimuth_angle"])
    )

    results["state"]["solar_scattering_angle"] = np.rad2deg(np.arccos(cos_scatter))

    for key in results["simulated_l1"]:
        results["state"][f"simulated_l1_{key}_radiance"] = results["simulated_l1"][
            key
        ].data.radiance

    return results["state"]

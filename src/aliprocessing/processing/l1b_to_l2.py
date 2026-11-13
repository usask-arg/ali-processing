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
from aliprocessing.l2.cloud import screen_clouds
from aliprocessing.l2.config import altitude_grid, engine_kwargs, load_retrieval_config
from aliprocessing.l2.forwardmodel import ALIForwardModel
from aliprocessing.l2.optical import aerosol_median_radius_db

# The aerosol state element in the retrieval configuration
_AEROSOL = "stratospheric_aerosol"


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

    With ``two_stage.enabled``, stage 1 retrieves over the configured
    measurement window, clouds are detected from its extinction and the L1B
    (aliprocessing.l2.cloud), and stage 2 fits only from above the cloud top,
    starting from stage 1's aerosol. The result is stage 2's state, with the
    cloud diagnostics, the lower bound used and stage 1's aerosol added.
    """
    config = load_retrieval_config(config)
    if not isinstance(cal_db, xr.Dataset):
        cal_db = xr.open_dataset(cal_db)
    if not config["two_stage"]["enabled"]:
        return _retrieve(l1b_image, por_image, cal_db, config)

    two_stage = config["two_stage"]
    stage1_config = copy.deepcopy(config)
    stage1_config["minimizer"]["max_nfev"] = two_stage["stage1_max_nfev"]
    stage1 = _retrieve(l1b_image, por_image, cal_db, stage1_config)

    low, high = config["measurement"]["altitude_range_m"]
    tropopause = (
        float(por_image["tropopause_altitude"])
        if "tropopause_altitude" in por_image
        else None
    )
    screen = screen_clouds(
        l1b_image.spectra["I"].ds,
        l1b_image.spectra["dolp"].ds if "dolp" in l1b_image.spectra else None,
        stage1[f"{_AEROSOL}_extinction_per_m"],
        two_stage,
        stage1_lower_altitude=float(low),
        tropopause_altitude=tropopause,
    )

    stage2_config = copy.deepcopy(config)
    stage2_config["measurement"]["altitude_range_m"] = [screen.lower_altitude, high]
    aerosol = stage2_config["state"]["aerosols"][_AEROSOL]
    aerosol["initial_guess"] = np.concatenate(
        [
            stage1[f"{_AEROSOL}_{quantity}"].to_numpy()
            for quantity in aerosol["retrieved_quantities"]
        ]
    )
    if screen.lower_altitude >= high:
        msg = (
            f"the cloud top at {screen.cloud_top_altitude:.0f} m leaves no altitudes "
            f"below the top of the measurement window, {high} m"
        )
        raise ValueError(msg)
    state = _retrieve(l1b_image, por_image, cal_db, stage2_config)

    for quantity in aerosol["retrieved_quantities"]:
        name = f"{_AEROSOL}_{quantity}"
        state[f"{name}_first_pass"] = stage1[name]
    state["num_iterations_first_pass"] = stage1["num_iterations"]
    state["cost_first_pass"] = stage1["cost"]
    state = state.merge(screen.to_dataset(), combine_attrs="drop_conflicts")
    state.attrs.update(screen.to_dataset().attrs)
    state.attrs["two_stage"] = 1
    return state


def _retrieve(
    l1b_image: L1bImage, por_image: xr.Dataset, cal_db: xr.Dataset, config: dict
) -> xr.Dataset:
    """One retrieval with a loaded configuration."""
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

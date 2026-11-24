from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path

import numpy as np
import skretrieval as skr
import xarray as xr
from skretrieval.core.lineshape import DeltaFunction
from skretrieval.retrieval.processing import Retrieval

from aliprocessing.l1b.data import L1bImage, _circular_mean_deg
from aliprocessing.l2.ancillary import Ancillary
from aliprocessing.l2.cloud import screen_clouds
from aliprocessing.l2.config import altitude_grid, engine_kwargs, load_retrieval_config
from aliprocessing.l2.forwardmodel import ALIForwardModel
from aliprocessing.l2.optical import aerosol_median_radius_db
from aliprocessing.l2.retrieval import ALIRetrieval
from aliprocessing.schemas import apply_metadata, check

# The aerosol state element in the retrieval configuration
_AEROSOL = "stratospheric_aerosol"
# skretrieval's nominal_wavelength of an aerosol extinction profile when unset
_DEFAULT_AEROSOL_WAVELENGTH_NM = 745.0


@Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_property(*args, **kwargs):
    return aerosol_median_radius_db()


def process_l1b_to_l2(
    l1b: xr.Dataset,
    por: xr.Dataset,
    cal_db: xr.Dataset | Path,
    config: dict | str | os.PathLike | None = None,
    **image_kwargs,
) -> xr.Dataset:
    """Retrieve every observation set and across-track bin of an L1B granule.

    Parameters
    ----------
    l1b
        L1B granule.
    por
        Program of record: pressure and temperature on ``altitude``, and
        optionally tropopause_altitude. Its variables may also have the L1B's
        ``observation_set`` and ``across_track`` dimensions, to give each image
        its own profile.
    cal_db, config
        As for process_l1b_to_l2_image.
    image_kwargs
        Passed to L1bImage.from_l1b, for example ``max_dolp_noise``.

    Returns
    -------
    xr.Dataset
        The L2 granule, checked against the L2 schema.

    An image whose retrieval raises an error, for example when a cloud top
    leaves no altitudes to fit, does not stop the granule: a warning is issued
    and its profile is stored with ``retrieval_failed`` set (see
    failed_l2_profile). When every image fails, the first error is raised.
    """
    config = load_retrieval_config(config)
    if not isinstance(cal_db, xr.Dataset):
        cal_db = xr.open_dataset(cal_db)
    check(l1b, "l1b", cal_db=cal_db)

    profiles = []
    failures = {}
    for observation_set in range(l1b.sizes["observation_set"]):
        row = []
        for across_track in range(l1b.sizes["across_track"]):
            index = {"observation_set": observation_set, "across_track": across_track}
            por_image = por.isel(
                {dim: i for dim, i in index.items() if dim in por.dims}
            )
            image = None
            try:
                image = L1bImage.from_l1b(
                    l1b, observation_set, across_track, **image_kwargs
                )
                state = process_l1b_to_l2_image(image, por_image, cal_db, config)
                row.append(l2_profile(state, config))
            except Exception as error:
                warnings.warn(
                    f"The retrieval of observation set {observation_set}, "
                    f"across-track bin {across_track} failed: {error!r}",
                    stacklevel=2,
                )
                failures[observation_set, across_track] = (error, image, por_image)
                row.append(None)
        profiles.append(row)

    template = next(
        (profile for row in profiles for profile in row if profile is not None), None
    )
    if template is None:
        error = next(iter(failures.values()))[0]
        msg = "The retrieval failed for every image of the granule"
        raise RuntimeError(msg) from error
    for (observation_set, across_track), (_, image, por_image) in failures.items():
        time = (
            l1b["time"]
            .isel(observation_set=observation_set)
            .min("wavelength")
            .to_numpy()
        )
        profiles[observation_set][across_track] = failed_l2_profile(
            template, time, image, por_image
        )
    return l2_granule(profiles, l1b["across_track_angle"].to_numpy())


def l2_profile(state: xr.Dataset, config: dict) -> xr.Dataset:
    """One L2 profile from the state of process_l1b_to_l2_image.

    ``config`` is the loaded configuration it was retrieved with. The profile
    has the L2 variables and global attributes without the observation_set and
    across_track dimensions; l2_granule stacks profiles into a granule. The O3
    profile, held near its prior, is not stored.
    """
    aerosol = config["state"]["aerosols"][_AEROSOL]
    quantities = aerosol["retrieved_quantities"]
    two_stage = config["two_stage"]["enabled"]
    low, high = config["measurement"]["altitude_range_m"]
    status = int(state["optimizer_status"])
    altitude = state["altitude"].to_numpy()
    profile_dims, kernel_dims = ["altitude"], ["altitude", "altitude2"]

    def scalar(name):
        return ((), float(state[name]))

    def retrieved(state_name, name):
        """A retrieved profile and its prior, errors and averaging kernel."""
        return {
            name: (profile_dims, state[state_name].to_numpy()),
            f"{name}_prior": (profile_dims, state[f"{state_name}_prior"].to_numpy()),
            f"{name}_noise_error": (
                profile_dims,
                state[f"{state_name}_1sigma_error"].to_numpy(),
            ),
            f"{name}_solution_error": (
                profile_dims,
                state[f"{state_name}_1sigma_solution_error"].to_numpy(),
            ),
            f"{name}_averaging_kernel": (
                kernel_dims,
                state[f"{state_name}_averaging_kernel"].to_numpy(),
            ),
        }

    def retrieval_space(quantity):
        log_space = quantities[quantity].get(
            "log_space", aerosol.get("log_space", False)
        )
        return "log" if log_space else "linear"

    data_vars = {
        "latitude": scalar("latitude"),
        "longitude": scalar("longitude"),
        "solar_zenith_angle": scalar("solar_zenith_angle"),
        "scattering_angle": scalar("solar_scattering_angle"),
        **retrieved(f"{_AEROSOL}_extinction_per_m", "aerosol_extinction"),
        "lower_altitude_limit": (
            scalar("retrieval_lower_altitude") if two_stage else ((), float(low))
        ),
        "upper_altitude_limit": ((), float(high)),
        "cost": scalar("cost"),
        "optimizer_status": ((), np.int64(status)),
        "converged": ((), status > 0),
        "retrieval_failed": ((), False),
        "num_function_evaluations": ((), np.int64(state["num_iterations"])),
        # Placeholders: 0 is "not assessed" until the flags are defined
        "profile_quality_flag": ((), np.int64(0)),
        "aerosol_extinction_quality_flag": (
            profile_dims,
            np.zeros(len(altitude), np.int64),
        ),
    }
    coords = {
        "time": state["time"].to_numpy(),
        "altitude": altitude,
        "altitude2": altitude,
    }
    attrs = {
        "retrieval_stages": 2 if two_stage else 1,
        "aerosol_extinction_wavelength": float(
            aerosol.get("nominal_wavelength", _DEFAULT_AEROSOL_WAVELENGTH_NM)
        ),
        "aerosol_extinction_retrieval_space": retrieval_space("extinction_per_m"),
    }

    if two_stage:
        data_vars["aerosol_extinction_stage1"] = (
            profile_dims,
            state[f"{_AEROSOL}_extinction_per_m_first_pass"].to_numpy(),
        )
        data_vars["cloud_top_altitude"] = scalar("cloud_top_altitude")
    if "median_radius" in quantities:
        data_vars |= retrieved(f"{_AEROSOL}_median_radius", "aerosol_median_radius")
        attrs["aerosol_median_radius_retrieval_space"] = retrieval_space(
            "median_radius"
        )
    # A disabled albedo is described without errors
    if "lambertian_albedo_1sigma_error" in state:
        albedo = state["lambertian_albedo"]
        coords["albedo_wavelength"] = albedo[albedo.dims[0]].to_numpy()
        data_vars |= {
            "surface_albedo": ("albedo_wavelength", albedo.to_numpy()),
            "surface_albedo_noise_error": (
                "albedo_wavelength",
                state["lambertian_albedo_1sigma_error"].to_numpy(),
            ),
            "surface_albedo_solution_error": (
                "albedo_wavelength",
                state["lambertian_albedo_1sigma_solution_error"].to_numpy(),
            ),
        }
    if "tropopause_altitude" in state:
        data_vars["tropopause_altitude"] = scalar("tropopause_altitude")

    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def failed_l2_profile(
    template: xr.Dataset,
    time: np.datetime64,
    image: L1bImage | None = None,
    por_image: xr.Dataset | None = None,
) -> xr.Dataset:
    """The L2 profile of an image whose retrieval failed.

    It has the variables of ``template``, a profile from l2_profile of the same
    granule, with ``retrieval_failed`` set, ``optimizer_status`` -2,
    ``converged`` false, no function evaluations and every float variable NaN,
    except the reference latitude and longitude of ``image`` when it was read
    and the tropopause altitude of ``por_image`` when it gives one.
    """
    profile = template.map(
        lambda variable: (
            xr.full_like(variable, np.nan) if variable.dtype.kind == "f" else variable
        ),
        keep_attrs=True,
    )
    profile = profile.assign_coords(time=time)
    profile["optimizer_status"] = np.int64(-2)
    profile["converged"] = False
    profile["retrieval_failed"] = True
    profile["num_function_evaluations"] = np.int64(0)
    if image is not None:
        profile["latitude"] = float(image.reference_latitude()["I"])
        profile["longitude"] = float(image.reference_longitude()["I"])
    if (
        por_image is not None
        and "tropopause_altitude" in por_image
        and "tropopause_altitude" in profile
    ):
        profile["tropopause_altitude"] = float(por_image["tropopause_altitude"])
    return profile


def l2_granule(
    profiles: list[list[xr.Dataset]], across_track_angle: np.ndarray
) -> xr.Dataset:
    """Stack profiles from l2_profile into an L2 granule, checked against the schema.

    ``profiles[i][j]`` is the profile of observation set i and across-track bin
    j, whose centre is ``across_track_angle[j]``. All profiles must share one
    altitude grid, the albedo wavelengths and the retrieval settings in their
    global attributes.
    """
    flat = [profile for row in profiles for profile in row]
    if not flat:
        msg = "An L2 granule needs at least one profile"
        raise ValueError(msg)
    if any(profile.attrs != flat[0].attrs for profile in flat[1:]):
        msg = "All profiles in a granule must share their retrieval settings"
        raise ValueError(msg)

    # time is one value per observation set, so it is compared across the bins
    # and concatenated only along observation_set
    options = {"data_vars": "all", "join": "exact", "combine_attrs": "override"}
    rows = [
        xr.concat(row, dim="across_track", coords="minimal", compat="equals", **options)
        for row in profiles
    ]
    l2 = xr.concat(
        rows, dim="observation_set", coords=["time"], compat="equals", **options
    )
    l2["across_track_angle"] = ("across_track", np.asarray(across_track_angle, float))
    l2 = apply_metadata(l2, "l2")
    check(l2, "l2")
    return l2


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
    """One retrieval with a loaded configuration.

    Each ``*_1sigma_error`` in the state, from the measurement noise, has a
    ``*_1sigma_solution_error`` from the solution covariance next to it.
    """
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

    ret = ALIRetrieval(
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

    solution = results["solution_state"]
    for name in solution.data_vars:
        if name.endswith("_1sigma_error"):
            solution_name = name.removesuffix("_error") + "_solution_error"
            results["state"][solution_name] = solution[name]

    results["state"]["time"] = l1b_image.spectra["I"].ds["time"]
    if "tropopause_altitude" in por_image:
        results["state"]["tropopause_altitude"] = float(
            por_image["tropopause_altitude"]
        )

    ref_lat = l1b_image.reference_latitude()["I"]
    ref_lon = l1b_image.reference_longitude()["I"]

    results["state"]["latitude"] = ref_lat
    results["state"]["longitude"] = ref_lon

    results["state"]["num_iterations"] = results["minimizer"]["minimizer"]["nfev"]
    results["state"]["cost"] = results["minimizer"]["minimizer"]["cost"]
    results["state"]["optimizer_status"] = results["minimizer"]["minimizer"]["status"]

    results["state"]["solar_zenith_angle"] = np.rad2deg(
        np.arccos(l1b_image.reference_cos_sza()["I"])
    )
    # Averaged as angles: near backscatter the values straddle ±180
    results["state"]["solar_azimuth_angle"] = _circular_mean_deg(
        l1b_image.spectra["I"].ds["relative_solar_azimuth_angle"]
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

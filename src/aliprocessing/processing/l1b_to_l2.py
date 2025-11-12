from __future__ import annotations

from pathlib import Path

import numpy as np
import sasktran2 as sk
import skretrieval as skr
import xarray as xr
from skretrieval.core.lineshape import DeltaFunction
from skretrieval.retrieval.processing import Retrieval

from aliprocessing.l1b.data import L1bImage
from aliprocessing.l2.ancillary import Ancillary
from aliprocessing.l2.forwardmodel import ALIForwardModel
from aliprocessing.l2.optical import aerosol_median_radius_db


@Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_property(*args, **kwargs):
    return aerosol_median_radius_db()


def process_l1b_to_l2_image(
    l1b_image: L1bImage, por_image: xr.Dataset, cal_db: Path, **kwargs  # noqa: ARG001
):
    good = ~np.isnan(por_image.pressure.to_numpy())
    anc = Ancillary(
        por_image.altitude.to_numpy()[good],
        por_image.pressure.to_numpy()[good],
        por_image.temperature.to_numpy()[good],
        **kwargs.get("ancillary_cfg", {}),
    )

    sample_wavel = l1b_image.sample_wavelengths()["I"]

    # Triplets
    triplets = {
        str(w): {
            "wavelength": [w],
            "weights": [1],
            "altitude_range": [kwargs.get("low_measurement_bound", 10000), 40000],
            "normalization_range": [40000, 45000],
        }
        for w in sample_wavel
    }

    meas_vec = {}
    for name, t in triplets.items():
        meas_vec[name] = skr.measvec.Triplet(
            t["wavelength"],
            t["weights"],
            t["altitude_range"],
            t["normalization_range"],
            normalize=False,
            log_space=False,
        )

    model_kwargs = {
        "num_threads": len(l1b_image.sample_wavelengths()["I"]),
        "num_stokes": 3,
        "stokes_basis": sk.StokesBasis.Observer,
        "multiple_scatter_source": sk.MultipleScatterSource.DiscreteOrdinates,
        "num_streams": 8,
        "init_successive_orders_with_discrete_ordinates": True,
        "threading_model": sk.ThreadingModel.Wavelength,
        "num_successive_orders_iterations": 1,
        "num_sza": 1,
        "input_validation_mode": sk.InputValidationMode.Disabled,
    }
    model_kwargs.update(kwargs.get("model_kwargs", {}))

    minimizer_kwargs = {
        "minimizer": "scipy",
        "method": "trf",
        "xtol": 1e-15,
        "include_bounds": True,
        "max_nfev": 200,
        "ftol": 1e-6,
    }
    minimizer_kwargs.update(kwargs.get("minimizer_kwargs", {}))

    state_kwargs = {
        "absorbers": {
            "o3": {
                "prior_influence": 1e6,
                "tikh_factor": 2.5e4,
                "log_space": False,
                "min_value": 0,
                "max_value": 1e-1,
                "prior": {"type": "mipas", "value": 1e-6},
            },
        },
        "aerosols": {
            "stratospheric_aerosol": {
                "type": "extinction_profile",
                "nominal_wavelength": 745,
                "retrieved_quantities": {
                    "extinction_per_m": {
                        "prior_influence": kwargs.get(
                            "extinction_prior_influence", 1e-6
                        ),
                        "tikh_factor": kwargs.get("extinction_tikh_factor", 1e-2),
                        "min_value": 0,
                        "max_value": 1e-3,
                        "log_space": kwargs.get("extinction_log_space", False),
                    },
                    "median_radius": {
                        "prior_influence": kwargs.get(
                            "median_radius_prior_influence", 1e-2
                        ),
                        "tikh_factor": kwargs.get("median_radius_tikh_factor", 1e1),
                        "min_value": 10,
                        "max_value": kwargs.get("max_median_radius", 589.0),
                    },
                },
                "prior": {
                    "extinction_per_m": {"type": "testing"},
                    "median_radius": {"type": "constant", "value": 80},
                },
            },
        },
        "surface": {
            "lambertian_albedo": {
                "prior_influence": kwargs.get("albedo_prior_influence", 0),
                "tikh_factor": kwargs.get("albedo_tikh_factor", 1e-4),
                "log_space": False,
                "wavelengths": kwargs.get(
                    "albedo_wavelengths",
                    np.atleast_1d(l1b_image.sample_wavelengths()["I"]),
                ),
                "initial_value": 0.3,
                "out_bounds_mode": "extend",
            },
        },
        "altitude_grid": np.arange(0.0, 65001.0, 1000.0),
    }

    state_kwargs.update(kwargs.get("state_kwargs", {}))

    ret = Retrieval(
        observation=l1b_image,
        forward_model_cfg={
            "*": {
                "kwargs": {
                    "lineshape_fn": lambda _: DeltaFunction(),
                    "spectral_native_coordinate": "wavelength_nm",
                    "model_res_nm": 2,
                    "pol_states": l1b_image.reference_cos_sza().keys(),
                },
                "class": ALIForwardModel,
            },
        },
        measvec=meas_vec,
        minimizer=minimizer_kwargs.pop("minimizer"),
        ancillary=anc,
        l1_kwargs={},
        model_kwargs=model_kwargs,
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

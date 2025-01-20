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


@Retrieval.register_optical_property("stratospheric_aerosol")
def stratospheric_aerosol_optical_property(*args, **kwargs):
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(mode_width=1.6)

    return sk.database.MieDatabase(
        dist,
        refrac,
        np.array([470, 525, 745, 1020, 1500]),
        median_radius=np.arange(10, 400, 10.0),
    )


def process_l1b_to_l2_image(
    l1b_image: L1bImage, por_image: xr.Dataset, cal_db: Path, **kwargs  # noqa: ARG001
):
    good = ~np.isnan(por_image.pressure.to_numpy())
    anc = Ancillary(
        por_image.altitude.to_numpy()[good],
        por_image.pressure.to_numpy()[good],
        por_image.temperature.to_numpy()[good],
    )

    sample_wavel = l1b_image.sample_wavelengths()["I"]

    # Triplets
    triplets = {
        str(w): {
            "wavelength": [w],
            "weights": [1],
            "altitude_range": [0, 40000],
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
    }

    minimizer_kwargs = {
        "method": "trf",
        "xtol": 1e-15,
        "include_bounds": True,
        "max_nfev": 200,
        "ftol": 1e-6,
    }

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
                        "prior_influence": 1e-6,
                        "tikh_factor": 1e-2,
                        "min_value": 0,
                        "max_value": 1e-3,
                    },
                    "median_radius": {
                        "prior_influence": kwargs.get(
                            "median_radius_prior_influence", 1e-2
                        ),
                        "tikh_factor": 1e1,
                        "min_value": 10,
                        "max_value": 300,
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
                "prior_influence": 0,
                "tikh_factor": 1e-4,
                "log_space": False,
                "wavelengths": np.atleast_1d(l1b_image.sample_wavelengths()["I"]),
                "initial_value": 0.3,
                "out_bounds_mode": "extend",
            },
        },
        "altitude_grid": np.arange(0.0, 65001.0, 1000.0),
    }

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
        minimizer="scipy",
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

    return results["state"]

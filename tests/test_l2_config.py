from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk
import yaml

from aliprocessing.l2.config import altitude_grid, engine_kwargs, load_retrieval_config

EXTINCTION = (
    "state",
    "aerosols",
    "stratospheric_aerosol",
    "retrieved_quantities",
    "extinction_per_m",
)


def _get(config, path):
    for key in path:
        config = config[key]
    return config


def test_defaults_are_the_previous_settings():
    config = load_retrieval_config()
    assert config["minimizer"] == {
        "minimizer": "scipy",
        "method": "trf",
        "xtol": 1e-15,
        "include_bounds": True,
        "max_nfev": 200,
        "ftol": 1e-6,
        "verbose": 2,
    }
    assert config["measurement"]["altitude_range_m"] == [10000, 40000]
    assert config["forward_model"]["model_res_nm"] == 2
    extinction = _get(config, EXTINCTION)
    assert extinction["tikh_factor"] == 1e-2
    assert extinction["log_space"] is False
    assert config["state"]["absorbers"]["o3"]["tikh_factor"] == 2.5e4
    aerosol = config["state"]["aerosols"]["stratospheric_aerosol"]
    assert aerosol["prior"]["median_radius"] == {"type": "constant", "value": 80}


def test_altitude_grid_matches_the_previous_default():
    np.testing.assert_array_equal(
        altitude_grid(load_retrieval_config()), np.arange(0.0, 65001.0, 1000.0)
    )


def test_altitude_grid_joins_segments():
    config = load_retrieval_config(
        {
            "altitude_grid_m": [
                {"start": 0, "stop": 10000, "step": 5000},
                {"start": 10000, "stop": 12001, "step": 1000},
            ]
        }
    )
    np.testing.assert_array_equal(altitude_grid(config), [0, 5000, 10000, 11000, 12000])


def test_engine_settings_use_sasktran2_enums():
    kwargs = engine_kwargs(load_retrieval_config(), num_channels=5)
    assert kwargs["stokes_basis"] == sk.StokesBasis.Observer
    assert (
        kwargs["multiple_scatter_source"] == sk.MultipleScatterSource.DiscreteOrdinates
    )
    assert kwargs["threading_model"] == sk.ThreadingModel.Wavelength
    assert kwargs["input_validation_mode"] == sk.InputValidationMode.Disabled
    assert kwargs["num_threads"] == 5


def test_thread_count_can_be_set():
    config = load_retrieval_config({"engine": {"num_threads": 2}})
    assert engine_kwargs(config, num_channels=5)["num_threads"] == 2


def test_unknown_engine_enum_name_is_rejected():
    config = load_retrieval_config({"engine": {"stokes_basis": "Sideways"}})
    with pytest.raises(AttributeError, match="Sideways"):
        engine_kwargs(config, num_channels=5)


def test_overrides_are_merged_into_the_defaults():
    config = load_retrieval_config({"minimizer": {"max_nfev": 50}})
    assert config["minimizer"]["max_nfev"] == 50
    assert config["minimizer"]["method"] == "trf"


def test_unknown_settings_are_rejected():
    with pytest.raises(ValueError, match=r"minimizer\.max_nfe"):
        load_retrieval_config({"minimizer": {"max_nfe": 50}})
    with pytest.raises(ValueError, match="Unknown retrieval setting engines"):
        load_retrieval_config({"engines": {}})


def test_a_section_cannot_be_replaced_by_a_value():
    with pytest.raises(ValueError, match="must be a mapping"):
        load_retrieval_config({"minimizer": 3})


def test_state_and_ancillary_accept_new_keys():
    config = load_retrieval_config(
        {
            "state": {"absorbers": {"o3": {"initial_guess": 1e-6}}},
            "ancillary": {"some_skretrieval_option": True},
        }
    )
    o3 = config["state"]["absorbers"]["o3"]
    assert o3["initial_guess"] == 1e-6
    assert o3["tikh_factor"] == 2.5e4
    assert config["ancillary"] == {"some_skretrieval_option": True}


def test_log_space_extinction_is_one_setting():
    config = load_retrieval_config(
        {
            "state": {
                "aerosols": {
                    "stratospheric_aerosol": {
                        "retrieved_quantities": {
                            "extinction_per_m": {"log_space": True}
                        }
                    }
                }
            }
        }
    )
    extinction = _get(config, EXTINCTION)
    assert extinction["log_space"] is True
    assert extinction["tikh_factor"] == 1e-2


def test_a_complete_configuration_is_accepted_as_overrides():
    complete = load_retrieval_config({"minimizer": {"max_nfev": 7}})
    assert load_retrieval_config(complete) == complete


def test_loading_does_not_modify_the_defaults():
    load_retrieval_config({"minimizer": {"max_nfev": 50}})
    assert load_retrieval_config()["minimizer"]["max_nfev"] == 200


def test_loads_a_yaml_file(tmp_path):
    path = tmp_path / "retrieval.yaml"
    path.write_text(
        yaml.safe_dump({"measurement": {"altitude_range_m": [12000, 38000]}})
    )
    assert load_retrieval_config(path)["measurement"]["altitude_range_m"] == [
        12000,
        38000,
    ]

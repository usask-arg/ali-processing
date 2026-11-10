from __future__ import annotations

import copy

import numpy as np
import pytest
import xarray as xr

from aliprocessing import schemas
from aliprocessing.schemas import SchemaValidationError, check, validate


def _errors(issues):
    return [issue for issue in issues if issue.severity == "error"]


def _locations(issues):
    return {issue.location for issue in _errors(issues)}


@pytest.mark.parametrize("product", ["l1a", "l1b", "cal_db"])
def test_valid_dataset_has_no_issues(product, request):
    assert validate(request.getfixturevalue(product), product) == []


def test_unknown_product_is_rejected(l1a):
    with pytest.raises(ValueError, match="Unknown product"):
        validate(l1a, "l3")


def test_missing_required_variable(l1a):
    issues = validate(l1a.drop_vars("image"), "l1a")
    assert _locations(issues) == {"variables.image"}


def test_missing_optional_variable_is_allowed(l1a):
    assert validate(l1a.drop_vars(["saturated", "detector_temperature"]), "l1a") == []


def test_wrong_dimensions(l1a):
    l1a["image"] = l1a["image"].transpose("time", "detector_column", "detector_row")
    issues = validate(l1a, "l1a")
    assert _locations(issues) == {"variables.image"}
    assert "dims" in _errors(issues)[0].message


def test_wrong_dtype(l1a):
    l1a["exposure_time"] = l1a["exposure_time"].astype(int)
    assert _locations(validate(l1a, "l1a")) == {"variables.exposure_time"}


def test_float_spectral_channel_is_rejected(l1a):
    l1a["spectral_channel"] = l1a["spectral_channel"].astype(float)
    assert _locations(validate(l1a, "l1a")) == {"variables.spectral_channel"}


def test_missing_units(l1a):
    del l1a["exposure_time"].attrs["units"]
    issues = validate(l1a, "l1a")
    assert _locations(issues) == {"variables.exposure_time"}
    assert "units" in _errors(issues)[0].message


def test_wrong_units(l1a):
    l1a["spacecraft_altitude"].attrs["units"] = "km"
    assert _locations(validate(l1a, "l1a")) == {"variables.spacecraft_altitude"}


def test_value_below_valid_min(l1a):
    l1a["spectral_channel"][0] = -2
    assert "variables.spectral_channel" in _locations(validate(l1a, "l1a"))


def test_value_above_valid_max(l1b):
    l1b["relative_solar_azimuth_angle"][0, 0, 0, 0] = 270.0
    assert _locations(validate(l1b, "l1b")) == {
        "variables.relative_solar_azimuth_angle"
    }


def test_nan_is_not_a_range_violation(cal_db):
    # Gap pixels are NaN in the per-pixel calibration
    assert np.isnan(cal_db["q_sensitivity"]).any()
    assert validate(cal_db, "cal_db") == []


def test_scalar_time_is_rejected(l1a):
    l1a = l1a.isel(time=0)
    assert "variables.time" in _locations(validate(l1a, "l1a"))


def test_unknown_variable_is_a_warning(l1a):
    l1a["mystery"] = l1a["exposure_time"]
    issues = validate(l1a, "l1a")
    assert _errors(issues) == []
    assert [(issue.severity, issue.location) for issue in issues] == [
        ("warning", "variables.mystery")
    ]


def test_missing_required_attribute(l1a):
    del l1a.attrs["schema_version"]
    assert _locations(validate(l1a, "l1a")) == {"attributes.schema_version"}


def test_attribute_of_wrong_type(cal_db):
    cal_db.attrs["valid_from"] = 2026
    assert _locations(validate(cal_db, "cal_db")) == {"attributes.valid_from"}


def test_attribute_not_in_allowed_values(l1b):
    l1b.attrs["radiance_noise_model"] = "everything"
    assert _locations(validate(l1b, "l1b")) == {"attributes.radiance_noise_model"}


@pytest.fixture
def l1b_with_conditional_attribute(monkeypatch):
    """The L1B schema plus an attribute required only for one noise model.

    No schema uses required_when yet, so the validator's support for it is tested
    on a patched copy.
    """
    schema = copy.deepcopy(schemas.load_schema("l1b"))
    schema["attributes"]["detector_noise_source"] = {
        "dtype": "str",
        "required_when": {"radiance_noise_model": "detector"},
        "role": "processing",
        "description": "Test attribute.",
    }
    monkeypatch.setattr(schemas, "load_schema", lambda _product: schema)


@pytest.mark.usefixtures("l1b_with_conditional_attribute")
def test_required_when_condition_met(l1b):
    assert _locations(validate(l1b, "l1b")) == {"attributes.detector_noise_source"}


@pytest.mark.usefixtures("l1b_with_conditional_attribute")
def test_required_when_condition_not_met(l1b):
    l1b.attrs["radiance_noise_model"] = "none"
    assert validate(l1b, "l1b") == []


def test_check_raises_with_every_error(l1a):
    with pytest.raises(SchemaValidationError) as excinfo:
        check(l1a.drop_vars(["image", "exposure_time"]), "l1a")
    message = str(excinfo.value)
    assert "variables.image" in message
    assert "variables.exposure_time" in message


def test_check_passes_with_only_warnings(l1a):
    l1a["mystery"] = l1a["exposure_time"]
    check(l1a, "l1a")


@pytest.mark.parametrize("product", ["l1a", "l1b", "cal_db"])
def test_dataset_still_valid_after_netcdf_round_trip(product, request, tmp_path):
    path = tmp_path / f"{product}.nc"
    request.getfixturevalue(product).to_netcdf(path)
    with xr.open_dataset(path) as ds:
        assert validate(ds, product) == []


def _tree(root, **groups):
    return xr.DataTree.from_dict({"/": root, **{f"/{k}": v for k, v in groups.items()}})


def test_valid_group(l1a, l1a_simulation):
    assert validate(_tree(l1a, simulation=l1a_simulation), "l1a") == []


def test_root_of_tree_is_checked(l1a, l1a_simulation):
    tree = _tree(l1a.drop_vars("image"), simulation=l1a_simulation)
    assert _locations(validate(tree, "l1a")) == {"variables.image"}


def test_group_variable_error_is_located_in_group(l1a, l1a_simulation):
    l1a_simulation["image_noiseless"].attrs["units"] = "electron"
    tree = _tree(l1a, simulation=l1a_simulation)
    assert _locations(validate(tree, "l1a")) == {
        "groups.simulation.variables.image_noiseless"
    }


def test_unknown_group_is_a_warning(l1b, l1a_simulation):
    issues = validate(_tree(l1b, simulation=l1a_simulation), "l1b")
    assert [(issue.severity, issue.location) for issue in issues] == [
        ("warning", "groups.simulation")
    ]


def test_tree_still_valid_after_netcdf_round_trip(l1a, l1a_simulation, tmp_path):
    path = tmp_path / "l1a.nc"
    _tree(l1a, simulation=l1a_simulation).to_netcdf(path)
    with xr.open_datatree(path) as tree:
        assert validate(tree, "l1a") == []

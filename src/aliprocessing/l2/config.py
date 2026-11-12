"""Configuration for the ALI L1B -> L2 retrieval; defaults in retrieval_config.yaml."""

from __future__ import annotations

import copy
import os
from importlib import resources
from pathlib import Path

import numpy as np
import sasktran2 as sk
import yaml

# Sections passed to skretrieval, which may hold any key
_FREE_FORM = ("state", "ancillary")

# Engine settings given by the name of a sasktran2 enum member
_ENGINE_ENUMS = {
    "stokes_basis": sk.StokesBasis,
    "multiple_scatter_source": sk.MultipleScatterSource,
    "threading_model": sk.ThreadingModel,
    "input_validation_mode": sk.InputValidationMode,
}


def load_retrieval_config(overrides: dict | str | os.PathLike | None = None) -> dict:
    """The default retrieval settings with ``overrides`` merged in.

    ``overrides`` is a dict or a YAML file with the same layout as
    retrieval_config.yaml; a complete configuration also works. Raises ValueError
    for unknown keys outside the ``state`` and ``ancillary`` sections.
    """
    text = (
        resources.files(__package__)
        .joinpath("retrieval_config.yaml")
        .read_text("utf-8")
    )
    config = yaml.safe_load(text)
    if isinstance(overrides, str | os.PathLike):
        overrides = yaml.safe_load(Path(overrides).read_text("utf-8"))
    _merge(config, copy.deepcopy(overrides or {}), path="", free_form=False)
    return config


def altitude_grid(config: dict) -> np.ndarray:
    """Retrieval altitude grid [m]: the union of the configured ranges."""
    return np.unique(
        np.concatenate(
            [
                np.arange(segment["start"], segment["stop"], segment["step"])
                for segment in config["altitude_grid_m"]
            ]
        ).astype(float)
    )


def engine_kwargs(config: dict, num_channels: int) -> dict:
    """Engine settings with enum names converted and the thread count filled in."""
    kwargs = dict(config["engine"])
    for name, enum in _ENGINE_ENUMS.items():
        if name in kwargs:
            kwargs[name] = getattr(enum, kwargs[name])
    if kwargs.get("num_threads") is None:
        kwargs["num_threads"] = num_channels
    return kwargs


def _merge(base: dict, overrides: dict, path: str, free_form: bool) -> None:
    for key, value in overrides.items():
        location = f"{path}{key}"
        if key not in base and not free_form:
            msg = f"Unknown retrieval setting {location}"
            raise ValueError(msg)
        if isinstance(base.get(key), dict) and not isinstance(value, dict):
            msg = f"Retrieval setting {location} must be a mapping, got {value!r}"
            raise ValueError(msg)
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(
                base[key],
                value,
                f"{location}.",
                free_form=free_form or (path == "" and key in _FREE_FORM),
            )
        else:
            base[key] = value

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import xarray as xr
from skretrieval.core.radianceformat import RadianceBase


class Transformer:
    """
    These classes can be used to compose functions applied to a radiance for use in a `MeasurementVectorElement`.
    """

    def transform(self, l1_data: RadianceBase, wf=None, covariance=False):
        if "error" in l1_data.data.keys():
            if covariance:
                l1_data.data["covariance"] = self.covariance(l1_data)
        return l1_data

    def jacobian(self, l1_data):
        return np.diag(np.ones(len(l1_data.data.los)))

    def covariance(self, l1_data):
        if "covariance" in l1_data.data.keys():
            l1_covariance = l1_data.data["covariance"]
        else:
            l1_covariance = np.diag(l1_data.data["error"] ** 2)
        J = self.jacobian(l1_data)
        return xr.DataArray(
            J @ l1_covariance @ J.T,
            dims=["los", "los2"],
            coords=[l1_data.data.los.values, l1_data.data.los.values],
        )


class MeasurementVectorElement:
    """
    Applies a set of transforms to a level 1 measurement.
    """

    def __init__(self, name=None):
        self._transforms = {}
        self._name = name

    @property
    def transforms(self):
        return self._transforms

    @property
    def name(self):
        return self._name

    def add_transform(self, transform: Transformer, index: int | str = "post"):
        """
        Add a transform to the measurment vector element.

        Parameters
        ----------
        transform : Transformer
            transformation to apply to the measurements.
        index :
            Order in which the transform will be applied. If `post` (default) the transform will be appended to the
            current list. If `pre` transform be prepended. If an integer the transform will be placed at position
            `index` in the current list of transforms.
        """
        if type(index) == str:
            if self._transforms:
                if index.lower() == "pre":
                    index = sorted(self._transforms.keys())[0] - 1
                elif index.lower() == "post":
                    index = sorted(self._transforms.keys())[-1] + 1
            else:
                index = 0

        if type(index) == int:
            if index in self._transforms.keys():
                for key in sorted(self._transforms.keys(), reverse=True):
                    self._transforms[index + 1] = self._transforms[index]
                    if key <= index:
                        break

            self._transforms[index] = transform
        else:
            raise ValueError(
                f"index should be a `pre`, `post` or an integer, got {index} with type {type(index)}"
            )

    def remove_transform(self, index):
        del self._transforms[index]

    def transform(self, l1_data, covariance=False):
        """
        Transform the level 1 data into measurerment vector space.
        """
        for t in sorted(self._transforms.keys()):
            l1_data = self._transforms[t].transform(l1_data, covariance=covariance)
        return l1_data

    def meas_dict(
        self, l1_data: list[RadianceBase] | RadianceBase, covariance: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Transform the level 1 data into measurement vector space and return the measurement vector, error and weighting
        functions.

        Parameters
        ----------
        l1_data:
            level 1 data that the transforms will be applied to.
        covariance:
            Whether to compute the full covariance matrix or only the diagonal. Default False
        """
        data = self.transform(l1_data, covariance=covariance).data

        result = dict()

        result["y"] = data.radiance.values
        if result["y"].shape == ():
            result["y"] = np.array([result["y"]])
        result["y"][~np.isfinite(result["y"])] = np.nan

        if "error" in data.keys():
            result["y_error"] = data["error"].values ** 2

        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                result[key] = data[key]
        return result


class MeasurementVector:
    """
    A full measurement vector made up of a collection of measurement vector elements.
    """

    def __init__(
        self,
        meas_vec_elements: list[MeasurementVectorElement],
        covariance=False,
        drop_zero_error=True,
    ):
        self._elements = meas_vec_elements
        self._covariance = covariance
        self._drop_zero_error = drop_zero_error

    @property
    def elements(self) -> list[MeasurementVectorElement]:
        return self._elements

    def meas_dict(self, l1_data) -> dict[str, np.ndarray]:
        """
        Transform the level 1 data into the measurement vector space and return the measurement vector, error and weighting
        functions.

        Parameters
        ----------
        l1_data:
            level 1 data that the transforms will be applied to.
        """
        md = []
        for el in self._elements:
            md.append(el.meas_dict(l1_data, covariance=self._covariance))

        meas_dict = {}
        for key in md[0]:
            if len(md[0][key].shape) == 1:
                meas_dict[key] = np.hstack([m[key] for m in md])
            else:
                meas_dict[key] = np.vstack([m[key] for m in md])

        if self._drop_zero_error:
            if "y_error" in meas_dict:
                if np.nansum(meas_dict["y_error"]) == 0.0:
                    del meas_dict["y_error"]

        return meas_dict

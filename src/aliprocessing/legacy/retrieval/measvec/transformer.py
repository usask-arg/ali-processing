from __future__ import annotations

from copy import copy
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from ali_processing.legacy.retrieval.measvec import Transformer
from scipy.interpolate import UnivariateSpline
from skretrieval.core.radianceformat import RadianceBase, RadianceGridded


class SpectralRatio(Transformer):
    """
    Spectral ratio at two wavelengths.

    Parameters
    ----------
    wavel_1: float
        Wavelength of numerator.
    wavel_2: float
        Wavelength of numerator.
    """

    def __init__(self, wavel_1: float, wavel_2: float):
        self._w1 = wavel_1
        self._w2 = wavel_2

    def transform(self, l1_data: RadianceBase, wf=None, covariance=False):
        data_1 = l1_data.data.sel(wavelength=self._w1, method="nearest")
        data_2 = l1_data.data.sel(wavelength=self._w2, method="nearest")

        data = l1_data.data.sel(wavelength=self._w1, method="nearest").copy(deep=True)
        data["radiance"] = data_1["radiance"] / data_2["radiance"]

        if "error" in data.keys():
            data["error"] = np.sqrt(
                (data_1["error"] / data_2["radiance"]) ** 2
                + (data_2["error"] * data_1["radiance"] / data_2["radiance"] ** 2)
            )

        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                data[key] = data_1[key] / data_2.radiance - data_2[
                    key
                ] * data_1.radiance / (data_2.radiance**2)

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy


class AltitudeNormalization(Transformer):
    """
    Apply an altitude normalization

    Parameters
    ----------
    norm_alts : Tuple[float, float]
        Tuple consisting of low and high altitude limits of the normalization range.
    nan_above: bool
        Make values above the altitude normalization range NaN. Default False
    couple_altitude: bool
        Account for altitude coupling in the Jacobian. Default True

    Examples
    --------
    >>> from ali_processing.legacy.retrieval.measvec.transformer import AltitudeNormalization
    >>> from ali_processing.legacy.retrieval.measvec import MeasurementVectorElement
    >>> alt_norm = AltitudeNormalization((35000.0, 40000.0))
    >>> meas_vec_el = MeasurementVectorElement()
    >>> meas_vec_el.add_transform(alt_norm)
    """

    def __init__(
        self,
        norm_alts: tuple[float, float],
        nan_above=False,
        couple_altitudes: bool = True,
    ):
        self._norm_alts = norm_alts
        self._nan_above = nan_above
        self._couple_altitude = couple_altitudes

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data, covariance=False):
        tanalts = l1_data.tangent_locations().altitude
        norm_data = l1_data.data.where(
            (tanalts > self._norm_alts[0]) & (tanalts < self._norm_alts[1]), drop=True
        )
        data = l1_data.data.copy(deep=True)
        norm_rad = norm_data["radiance"].mean(dim="los")
        data["radiance"] = data["radiance"] / norm_rad

        if "error" in data.keys():
            norm_error = np.sqrt(
                1 / (len(norm_data.los) ** 2) * ((norm_data["error"]) ** 2).sum()
            )
            variance = (
                (l1_data.data["error"] / l1_data.data["radiance"]) ** 2
                + (norm_error / norm_rad) ** 2
            ) * (data["radiance"] ** 2)
            data["error"] = np.sqrt(variance)

            if covariance:
                S = self.covariance(l1_data)
                data["error"] = xr.DataArray(
                    np.sqrt(np.diag(S)), dims=["los"], coords=[l1_data.data.los.values]
                )
                data["covariance"] = S

        data["radiance"] = data["radiance"].where(tanalts < self._norm_alts[0])
        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                if self._couple_altitude:
                    data[key] = l1_data.data[key] / norm_rad + norm_data[key].mean(
                        dim="los"
                    ) * (l1_data.data["radiance"] / (norm_rad**2))
                else:
                    data[key] = l1_data.data[key] / norm_rad

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy

    def jacobian(self, l1_data):
        tanalts = l1_data.tangent_locations().altitude
        norm_data = l1_data.data.where(
            (tanalts > self._norm_alts[0]) & (tanalts < self._norm_alts[1]), drop=True
        )
        norm_rad = norm_data["radiance"].mean(dim="los")

        J = np.zeros((len(l1_data.data.los), len(l1_data.data.los)))
        N = len(norm_data.los)
        norm_idx = norm_data.los.values
        sum_norm_rad = float(norm_data.radiance.sum().values)
        rad = l1_data.data.radiance.values

        # TODO: vectorize
        # if self._couple_altitude:
        for i in range(len(l1_data.data.los)):
            for j in range(len(l1_data.data.los)):
                if j == i:
                    if j in norm_idx:
                        J[i, j] = N * (sum_norm_rad - rad[i]) / (sum_norm_rad**2)
                    else:
                        J[i, j] = 1 / norm_rad.values
                else:
                    if j in norm_idx:
                        J[i, j] = rad[i] / (sum_norm_rad**2)
                    else:
                        J[i, j] = 0

        return J


class AltitudeNormalizationShift(AltitudeNormalization):
    """
    Apply an altitude normalization that shifts the measurements rather than using a division.
    This can be useful for cases when

    Parameters
    ----------
    norm_alts : Tuple[float, float]
        Tuple consisting of low and high altitude limits of the normalization range.
    """

    def transform(self, l1_data, wf=None, covariance=False):
        tanalts = l1_data.tangent_locations().altitude
        norm_data = l1_data.data.where(
            (tanalts > self._norm_alts[0]) & (tanalts < self._norm_alts[1]), drop=True
        )
        data = l1_data.data.copy(deep=True)
        norm_rad = norm_data["radiance"].mean(dim="los")
        data["radiance"] = data["radiance"] - norm_rad

        if "error" in data.keys():
            norm_error = np.sqrt(((norm_data["error"]) ** 2).sum()) / len(
                norm_data["error"].values
            )
            variance = (l1_data.data["error"] ** 2) + (norm_error**2)
            data["error"] = np.sqrt(variance)

            if covariance:
                S = self.covariance(l1_data)
                data["error"] = xr.DataArray(
                    np.sqrt(np.diag(S)), dims=["los"], coords=[l1_data.data.los.values]
                )
                data["covariance"] = S

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy


class LogRadiance(Transformer):
    def transform(self, l1_data: RadianceBase, wf=None, covariance=False):
        data = l1_data.data.copy(deep=True)
        data["radiance"] = np.log(data["radiance"])

        if "error" in data.keys():
            data["error"] = l1_data.data["error"] / l1_data.data["radiance"]

        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                data[key] = l1_data.data[key] / l1_data.data["radiance"]

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy


class FrameSelect(Transformer):
    def __init__(self, index=0):
        self._index = index

    def transform(self, l1_data: list[RadianceBase], wf=None, covariance=False):
        return l1_data[self._index]


class WavelengthSelect(Transformer):
    """
    Select a single wavelength
    """

    def __init__(self, wavelength: float, method="nearest"):
        self._wavelength = wavelength
        self._method = method

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data, wf=None, covariance=False):
        data = l1_data.data.copy(deep=True)
        l1_copy = copy(l1_data)
        l1_copy.data = data.sel(wavelength=self._wavelength, method=self._method)
        return l1_copy


class LinearCombination(Transformer):
    r"""
    Compute a linear combination of radiances. `l1_data` is expected to be a List[RadianceBase]. Often useful
    for computing polarization states.

    .. math::
        y = \sum_{i=0}^{N} l1[key_i] * weight_i

    Parameters
    ----------
    weights: Dict[int, float]
        Dictionary in the form {frame_index: weight, ...}

    """

    def __init__(self, weights: dict[int, float]):
        self._weights = weights

    def transform(self, l1_data: list[RadianceBase], wf=None, covariance=False):
        idx0 = list(self._weights.keys())[0]

        data = l1_data[idx0].data.copy(deep=True)

        data["radiance"] = data["radiance"] * 0.0
        for idx, w in self._weights.items():
            data["radiance"] += w * l1_data[idx].data["radiance"]

        if "error" in data.keys():
            data["error"] = l1_data[idx0].data["error"] * 0.0
            for idx, w in self._weights.items():
                data["error"] += (w**2) * (l1_data[idx].data["error"] ** 2)
            data["error"] = np.sqrt(data["error"])

        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                data[key] = l1_data[idx0].data[key] * 0.0
                for idx, w in self._weights.items():
                    data[key] += w * l1_data[idx].data[key]

        l1_copy = copy(l1_data[idx0])
        l1_copy.data = data
        return l1_copy


class FrameRatio(Transformer):
    def __init__(self, index_1: int = 0, index_2: int = 1):
        self._index_1 = index_1
        self._index_2 = index_2

    def transform(self, l1_data: list[RadianceBase], wf=None, covariance=False):
        data_1 = l1_data[self._index_1].data
        data_2 = l1_data[self._index_2].data

        data = l1_data[self._index_1].data.copy(deep=True)
        data["radiance"].values = data_1["radiance"].values / data_2["radiance"].values

        if "error" in data.keys():
            data["error"].values = np.sqrt(
                (data_1["error"] / data_1["radiance"]) ** 2
                + (data_2["error"] / data_2["radiance"]) ** 2
            ) * np.abs(data["radiance"])
            data["error"] = data["error"].fillna(data["error"].max())

        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                data[key] = data_1[key] / data_2.radiance - data_2[
                    key
                ] * data_1.radiance / (data_2.radiance**2)

        l1_copy = copy(l1_data[self._index_1])
        l1_copy.data = data
        return l1_copy


class FrameRatios(Transformer):
    def __init__(self, index: list[tuple[int, int]]):
        self._index = index

    def transform(self, l1_data: list[RadianceBase], wf=None, covariance=False):
        data = []
        for i1, i2 in self._index:
            mv = FrameRatio(index_1=i1, index_2=i2)
            data.append(mv.transform(l1_data, covariance=covariance))
        return data


class LinearCombinations(Transformer):
    def __init__(self, weights: list[dict[int, float]]):
        self._weights = weights

    def transform(self, l1_data: list[RadianceBase], wf=None, covariance=False):
        data = []
        for weights in self._weights:
            mv = LinearCombination(weights)
            data.append(mv.transform(l1_data, covariance=covariance))
        return data


class SplineSmoothing(Transformer):
    def __init__(self, smoothing=200):
        self.smoothing = smoothing

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data: RadianceBase, covariance=False):
        data = l1_data.data.copy(deep=True)
        tanalts = l1_data.tangent_locations().altitude

        old_dims = data.radiance.dims
        good_data = data.radiance.transpose("los", ...).values
        good_data[~np.isfinite(good_data)] = 0.0
        good_err = data.error.transpose("los", ...).values
        good_err[~np.isfinite(good_err)] = np.max(good_err[np.isfinite(good_err)])

        if len(good_data.shape) > 1:
            new_data = []
            for gd, ge in zip(good_data.T, good_err.T, strict=False):
                spl = UnivariateSpline(x=tanalts, y=gd, w=1 / ge, s=self.smoothing)
                new_data.append(spl(tanalts))
            new_data = np.array(new_data).T
        else:
            spl = UnivariateSpline(
                x=tanalts, y=good_data, w=1 / good_err, s=self.smoothing
            )
            new_data = spl(tanalts)

        data_rad = data.transpose("los", ...)["radiance"]
        data_rad.values = new_data
        data["radiance"].values = data_rad.transpose(*old_dims).values

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy


class VerticalDerivative(Transformer):
    def __init__(self, smoothing=200):
        self.smoothing = smoothing

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data: RadianceBase, covariance=False):
        data = l1_data.data.copy(deep=True)

        tanalts = l1_data.tangent_locations().altitude
        # Differentiation matrix
        n = len(data.los)
        h = np.concatenate([np.diff(tanalts), [tanalts[-1] - tanalts[-2]]])
        D = np.diag(-np.ones(n) / h) + np.diag(np.ones(n - 1) / h[:-1], 1)

        # Fix the final row
        D[n - 1, n - 2] = -1 / h[-1]
        D[n - 1, n - 1] = 1 / h[-1]

        old_dims = data.radiance.dims
        good_data = data.radiance.transpose("los", ...).values
        good_data[~np.isfinite(good_data)] = 0.0

        new_data = D @ good_data
        data_rad = data.transpose("los", ...)["radiance"]
        data_rad.values = new_data

        # D @ np.diag(data.radiance.transpose('los', ...).values ** 2) @ D.T
        data["radiance"].values = data_rad.transpose(*old_dims).values

        if "error" in data.keys():
            good_err = data.error.transpose("los", ...).values
            good_err[~np.isfinite(good_err)] = np.max(good_err[np.isfinite(good_err)])
            new_error = np.sqrt((D**2) @ (good_err**2))
            data_err = data.transpose("los", ...)["error"]
            data_err.values = new_error
            data["error"].values = data_err.transpose(*old_dims).values

        wf_keys = [key for key in data.keys() if "wf" in key]
        if wf_keys:
            for key in wf_keys:
                old_dims = data[key].dims
                data_wf = data[key].transpose("los", ...)
                new_wf = D @ data_wf.values
                data_wf.values = new_wf
                data[key].values = data_wf.transpose(*old_dims).values

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy


class AltitudeBinning(Transformer):
    def __init__(self, altitudes: np.ndarray):
        self._altitudes = altitudes

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data: RadianceBase, wf=None, covariance=False):
        tanalts = l1_data.tangent_locations().altitude
        rad = []
        los_dims = [dim for dim in l1_data.data["los_vectors"].dims if dim != "xyz"]
        for idx, (min_alt, max_alt) in enumerate(
            zip(self._altitudes[:-1], self._altitudes[1:], strict=False)
        ):
            good = (tanalts > min_alt) & (tanalts < max_alt)
            tmp = l1_data.data.where(good).mean(dim=los_dims)
            tmp["los"] = idx
            if "error" in tmp:
                tmp["error"] = tmp["error"] / np.sqrt(np.sum(good.values))
            rad.append(tmp)
        rad = xr.concat(rad, dim="los")
        return RadianceGridded(rad)


class RowAverage(Transformer):
    """
    Reduce the dimension of the measurement data by averaging across a row from pixels `first_pixel` to `last_pixel`.
    If `first_pixel` is not given the entire row is used.
    """

    def __init__(
        self,
        dim="nx",
        to_gridded: bool = True,
        first_pixel: int = None,
        last_pixel: int = None,
    ):
        self._dim = dim
        self._to_gridded = to_gridded
        self._first_pixel = first_pixel
        self._last_pixel = last_pixel

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data: RadianceBase, wf=None, covariance=False):
        if self._first_pixel:
            num_cols = self._last_pixel - self._first_pixel
            data = l1_data.data.isel(
                {self._dim: slice(self._first_pixel, self._last_pixel)}
            ).mean(dim=self._dim)
        else:
            num_cols = l1_data.data[self._dim].shape[0]
            data = l1_data.data.mean(dim=self._dim)

        if "error" in data.keys():
            data["error"].values = data["error"] / np.sqrt(num_cols)

        l1_copy = copy(l1_data)
        l1_copy.data = data

        if self._to_gridded:
            return l1_data.to_gridded()

        # TODO: implement error and weighting function transforms
        return l1_copy


class Truncate(Transformer):
    def __init__(self, lower_bound: float = None, upper_bound: float = None):
        self._lowerbound = lower_bound
        self._upperbound = upper_bound

    def transform(
        self,
        l1_data: RadianceBase | list[RadianceBase],
        wf=None,
        covariance=False,
    ):
        if type(l1_data) is list:
            return [self._transform(l1, covariance=covariance) for l1 in l1_data]
        return self._transform(l1_data, covariance=covariance)

    def _transform(self, l1_data: RadianceBase, covariance=False):
        data = l1_data.data.copy(deep=True)
        tanalts = l1_data.tangent_locations().altitude
        if self._lowerbound is not None:
            data["radiance"] = data.radiance.where(tanalts > self._lowerbound)
        if self._upperbound is not None:
            data["radiance"] = data.radiance.where(tanalts < self._upperbound)

        l1_copy = copy(l1_data)
        l1_copy.data = data
        return l1_copy

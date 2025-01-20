from __future__ import annotations

import abc
from datetime import datetime
from pathlib import Path

import numpy as np
import sasktran2 as sk
import xarray as xr
from skretrieval.core.radianceformat import RadianceGridded
from skretrieval.retrieval.observation import Observation


class L1bSpectra:
    @classmethod
    def from_np_arrays(
        cls,
        radiance: np.array,
        radiance_noise: np.array,
        tangent_altitude: np.array,
        tangent_latitude: np.array,
        tangent_longitude: np.array,
        sample_wavelengths_nm: np.array,
        time: datetime,
        observer_latitude: float,
        observer_longitude: float,
        observer_altitude: float,
        sza: np.array,
        saa: np.array,
        los_azimuth_angle: np.array,
    ):
        ds = xr.Dataset()

        ds["radiance"] = xr.DataArray(radiance, dims=["wavelength", "los"])
        ds["radiance_noise"] = xr.DataArray(radiance_noise, dims=["wavelength", "los"])
        ds["tangent_altitude"] = xr.DataArray(tangent_altitude, dims=["los"])
        ds["tangent_latitude"] = xr.DataArray(tangent_latitude, dims=["los"])
        ds["tangent_longitude"] = xr.DataArray(tangent_longitude, dims=["los"])

        ds["time"] = time

        ds["sample_wavelengths_nm"] = xr.DataArray(
            sample_wavelengths_nm, dims=["wavelength"]
        )

        ds["spacecraft_latitude"] = observer_latitude
        ds["spacecraft_longitude"] = observer_longitude
        ds["spacecraft_altitude"] = observer_altitude

        ds["solar_zenith_angle"] = xr.DataArray(sza, dims=["los"])
        ds["relative_solar_azimuth_angle"] = xr.DataArray(saa, dims=["los"])
        ds["los_azimuth_angle"] = xr.DataArray(los_azimuth_angle, dims=["los"])
        return cls(ds)

    def __init__(self, ds: xr.Dataset, low_alt=0, high_alt=100000):
        self._ds = ds
        self._low_alt = low_alt
        self._high_alt = high_alt

    @property
    def ds(self):
        return self._ds


class L1bImage(Observation):
    def __init__(self, spectra: dict[xr.Dataset], low_alt=0, high_alt=100000):
        self._spectra = spectra
        self._low_alt = low_alt
        self._high_alt = high_alt

    @property
    def spectra(self):
        return self._spectra

    def sk2_geometry(self) -> dict[sk.ViewingGeometry]:
        result = {}
        for key, spectra in self._spectra.items():
            ds = spectra.ds
            viewing_geo = sk.ViewingGeometry()

            good_alt = (ds.tangent_altitude.to_numpy() > self._low_alt) & (
                ds.tangent_altitude.to_numpy() < self._high_alt
            )

            for i in range(len(ds.los[good_alt])):
                viewing_geo.add_ray(
                    sk.TangentAltitudeSolar(
                        ds["tangent_altitude"].to_numpy()[good_alt][i],
                        np.deg2rad(
                            ds["relative_solar_azimuth_angle"].to_numpy()[good_alt][i]
                        ),
                        float(ds["spacecraft_altitude"]),
                        np.cos(
                            np.deg2rad(ds["solar_zenith_angle"].to_numpy()[good_alt][i])
                        ),
                    )
                )
            result[key] = viewing_geo

        return result

    def skretrieval_l1(self, *args, **kwargs):
        result = {}
        for key, spectra in self._spectra.items():
            spectra_ds = spectra.ds
            ds = xr.Dataset()

            good_alt = (spectra_ds.tangent_altitude.to_numpy() > self._low_alt) & (
                spectra_ds.tangent_altitude.to_numpy() < self._high_alt
            )

            ds["radiance"] = xr.DataArray(
                spectra_ds["radiance"].to_numpy()[:, good_alt],
                dims=["wavelength", "los"],
            )
            ds["radiance_noise"] = xr.DataArray(
                spectra_ds["radiance_noise"].to_numpy()[:, good_alt],
                dims=["wavelength", "los"],
            )

            ds.coords["tangent_altitude"] = spectra_ds["tangent_altitude"][good_alt]
            ds.coords["wavelength"] = spectra_ds["sample_wavelengths_nm"]

            ds = ds.set_xindex("tangent_altitude")

            result[key] = RadianceGridded(ds)

        return result

    @abc.abstractmethod
    def sample_wavelengths(self) -> dict[np.array]:
        """
        The sample wavelengths for the observation in [nm]

        Returns
        -------
        dict[np.array]
        """
        l1 = self.skretrieval_l1()

        return {key: l1[key].data.wavelength for key in l1}

    @abc.abstractmethod
    def reference_cos_sza(self) -> dict[float]:
        """
        The reference cosine of the solar zenith angle for the observation

        Returns
        -------
        dict[float]
        """

        return {
            key: np.cos(np.deg2rad(ds.ds["solar_zenith_angle"].mean()))
            for key, ds in self._spectra.items()
        }

    @abc.abstractmethod
    def reference_latitude(self) -> dict[float]:
        """
        The reference latitude for the observation

        Returns
        -------
        dict[float]
        """
        return {
            key: ds.ds["tangent_latitude"].mean() for key, ds in self._spectra.items()
        }

    @abc.abstractmethod
    def reference_longitude(self) -> dict[float]:
        """
        The reference longitude for the observation

        Returns
        -------
        dict[float]
        """
        return {
            key: ds.ds["tangent_longitude"].mean() for key, ds in self._spectra.items()
        }

    def append_information_to_l1(self, l1: dict[RadianceGridded], **kwargs) -> None:
        """
        A method that allows for the observation to append information to the L1 data
        simulated by the forward model. Useful for adding things that are in the real L1 data
        to the simulations that may be useful inside the measurement vector.

        Parameters
        ----------
        l1 : dict[RadianceGridded]
        """
        for key, spectra in self._spectra.items():
            ds = spectra.ds
            good_alt = (ds.tangent_altitude.to_numpy() > self._low_alt) & (
                ds.tangent_altitude.to_numpy() < self._high_alt
            )
            l1[key].data.coords["tangent_altitude"] = (
                ["los"],
                ds["tangent_altitude"].to_numpy()[good_alt],
            )

            l1[key].data = l1[key].data.set_xindex("tangent_altitude")


class L1bFileWriter:
    def __init__(self, l1b_data: list[L1bImage]) -> None:
        self._keys = l1b_data[0].spectra.keys()

        self._data = {
            key: xr.concat([l1b.spectra[key] for l1b in l1b_data], dim="time")
            for key in self._keys
        }

    def _apply_global_attributes(self):
        pass

    def save(self, out_file: Path):
        for key in self._keys:
            self._data[key].to_netcdf(out_file.as_posix(), group=key)


class L1bDataSet:
    def __init__(self, ds: dict[xr.Dataset]):
        """
        Loads in a single L1b file and provides access to the data

        """
        self._ds = ds

    @classmethod
    def from_image(cls, image: L1bImage):
        return cls(xr.concat([l1b._ds for l1b in [image]], dim="time"), "", None)

    @classmethod
    def from_file(cls, file_path: Path):
        return cls(xr.open_dataset(file_path), file_path.stem, file_path.parent.parent)

    @property
    def ds(self):
        return self._ds

    def image(self, sample: int, row_reduction: int = 1, low_alt=0, high_alt=100000):
        return L1bImage(
            self._ds.isel(time=sample).isel(los=slice(None, None, row_reduction)),
            low_alt=low_alt,
            high_alt=high_alt,
        )

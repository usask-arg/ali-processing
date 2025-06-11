from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr

unimodal_folder = r"C:\Users\lar555\data\balloon\laramie\SizeDist_Stratosphere_Unimodal"
bimodal_folder = r"C:\Users\lar555\data\balloon\laramie\SizeDist_Stratosphere"
instruments = ["WPC", "LPC", "Dust"]


class ClimatologyAerosolOPC(sk.ClimatologyUserDefined):

    def __init__(
        self,
        mode="fine",
        bimodal=True,
        extend=True,
        altitudes: np.ndarray | None = None,
        date=None,
    ):
        if altitudes is None:
            altitudes = np.arange(0, 100000, 500)

        self._mode = mode.lower()
        self.bimodal = bimodal
        self.extend = extend
        self._species_name = "SKCLIMATOLOGY_AEROSOL_CM3"
        self.decay_below = 1e-4
        self.decay_above = 5e-4
        self.dataset = None
        self.date = date
        super().__init__(
            altitudes=altitudes,
            values={self._species_name: altitudes * 0},
            interp="linear",
            spline=False,
        )

    def get_parameter(
        self,
        species: str,
        latitude: float | None = None,
        longitude: float | None = None,
        altitudes: np.ndarray | None = None,
        mjd: float = 53000,
    ):
        if altitudes is None:
            altitudes = np.arange(0, 50001, 500.0)

        if self.date is None:
            time = pd.Timestamp("1858-11-17") + pd.Timedelta(mjd, "D")
        else:
            time = self.date

        if latitude is None:
            latitude = 41.0
        if longitude is None:
            longitude = 110.0
        self.dataset = read_file(get_filename_from_date(time, bimodal=self.bimodal))

        # if species not in self._values.keys():
        #     raise sk.SasktranError('species not supported')

        if species == "SKCLIMATOLOGY_AEROSOL_CM3":
            clim = self.dataset[f"number_density_{self._mode}"].to_numpy()
        elif species == "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS":
            clim = self.dataset[f"median_radius_{self._mode}"].to_numpy()
        elif species == "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH":
            clim = self.dataset[f"width_{self._mode}"].to_numpy()
        else:
            msg = f"Unrecognized species {species}"
            raise ValueError(msg)

        if type(clim) is np.ma.core.MaskedArray:
            clim = clim.data
        good = ~np.isnan(clim)
        if sum(good) == 0:
            msg = "number density profile is entirely nans"
            raise ValueError(msg)

        if species == "SKCLIMATOLOGY_AEROSOL_CM3":
            clim = np.interp(
                altitudes, self.dataset.altitude.to_numpy()[good] * 1000, clim[good]
            )
            if hasattr(altitudes, "__len__"):
                if self.extend:
                    # decay the aerosol above the top point faster than the background
                    msis = sk.MSIS90()
                    high_alts = (
                        altitudes > self.dataset.altitude.to_numpy()[good][-1] * 1000
                    )
                    air = msis.get_parameter(
                        "SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3",
                        latitude,
                        longitude,
                        altitudes[high_alts],
                        mjd,
                    )
                    air_top = msis.get_parameter(
                        "SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3",
                        latitude,
                        longitude,
                        self.dataset.altitude.to_numpy()[good][-1] * 1000,
                        mjd,
                    )
                    last = np.where(~high_alts)[0][-1]
                    clim[high_alts] = (
                        clim[high_alts][0]
                        * air
                        / air_top
                        * np.exp(
                            self.decay_above * -(altitudes[high_alts] - altitudes[last])
                        )
                    )

                    # exponentially decay the profile below the lower bound
                    low_alts = (
                        altitudes < self.dataset.altitude.to_numpy()[good][0] * 1000
                    )
                    first = np.where(~low_alts)[0][0]
                    clim[low_alts] = clim[first] * np.exp(
                        self.decay_below * (altitudes[low_alts] - altitudes[first])
                    )
                else:
                    clim[altitudes > self.dataset.altitude[-1].to_numpy() * 1000] = 0.0
                    clim[altitudes < self.dataset.altitude[0].to_numpy() * 1000] = 0.0

            else:
                if altitudes > self.dataset.altitude[-1].to_numpy() * 1000:
                    clim = 0.0
                if altitudes < self.dataset.altitude[0].to_numpy() * 1000:
                    clim = 0.0
        else:
            good = ~np.isnan(clim) & (clim > 0)
            if hasattr(altitudes, "__len__"):
                clim = np.interp(
                    altitudes,
                    self.dataset.altitude.to_numpy()[good] * 1000,
                    clim[good],
                    left=clim[good][0],
                    right=clim[good][-1],
                )
            else:
                if altitudes > self.dataset.altitude[-1].to_numpy() * 1000:
                    clim = 0.0
                if altitudes < self.dataset.altitude[0].to_numpy() * 1000:
                    clim = 0.0

        return clim

    def skif_object(self, **kwargs):

        engine = kwargs["engine"]
        reference_point = engine.model_parameters["referencepoint"]
        latitude = reference_point[0]
        longitude = reference_point[1]
        mjd = reference_point[3]
        opt_prop = kwargs["opt_prop"]

        rg = self.get_parameter(
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS",
            latitude,
            longitude,
            self._altitudes,
            mjd,
        )
        sg = self.get_parameter(
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH",
            latitude,
            longitude,
            self._altitudes,
            mjd,
        )
        n = self.get_parameter(
            "SKCLIMATOLOGY_AEROSOL_CM3", latitude, longitude, self._altitudes, mjd
        )
        opt_prop.particlesize_climatology[
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS"
        ] = rg
        opt_prop.particlesize_climatology["SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH"] = sg
        clim_values = {"SKCLIMATOLOGY_AEROSOL_CM3": n}
        userdef_clim = sk.ClimatologyUserDefined(self._altitudes, clim_values)
        return userdef_clim.skif_object()


class SpeciesAerosolOPC(sk.SpeciesAerosol):

    def __init__(
        self,
        altitudes: None | np.ndarray = None,
        mode="fine",
        species: str = "H2SO4",
        interp: str = "linear",
        spline: bool = False,
        extend=True,
        date=None,
        bimodal=True,
    ):
        if altitudes is None:
            altitudes = np.ndarray = np.arange(0, 100001, 500)

        particle_size_values = {
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": np.ones_like(altitudes)
            * 0.08,
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": np.ones_like(altitudes) * 1.6,
        }

        super().__init__(
            altitudes,
            {"SKCLIMATOLOGY_AEROSOL_CM3": np.ones_like(altitudes) * 0.0},
            particle_size_values,
            species=species,
            interp=interp,
            spline=spline,
        )
        self._climatology = ClimatologyAerosolOPC(
            mode=mode, extend=extend, altitudes=altitudes, date=date, bimodal=bimodal
        )


def read_file(filename):

    with Path.open(filename) as f:
        data = []
        reading = False
        for line in f.readlines():
            if reading:
                if "min" in line.lower():
                    pass
                    # units = line.strip()
                else:
                    data.append(np.array([float(x) for x in line.strip().split()]))
            else:
                if "sfcarea" in line.lower() and "tempk" in line.lower():
                    reading = True
                    header = line.strip().lower().split()
                if "balloon release" in line.lower():
                    hh = line.split(",")[1].split()[0]
                    dd = line.split(",")[2]
                    release_date = pd.Timestamp(f"{dd} {hh}")

    data = np.array(data)
    time = release_date + pd.to_timedelta(data[:, header.index("tim")], "minute")
    data[data < 0] = np.nan

    return xr.Dataset(
        {
            "number_density_fine": ("altitude", data[:, header.index("no1")]),
            "median_radius_fine": ("altitude", data[:, header.index("ro1")]),
            "width_fine": ("altitude", data[:, header.index("so1")]),
            "number_density_coarse": ("altitude", data[:, header.index("no2")]),
            "median_radius_coarse": ("altitude", data[:, header.index("ro2")]),
            "width_coarse": ("altitude", data[:, header.index("so2")]),
            "pressure": ("altitude", data[:, header.index("press")]),
            "temperature": ("altitude", data[:, header.index("tempk")]),
            "surface_area_density": ("altitude", data[:, header.index("sfcarea")]),
            "volume": ("altitude", data[:, header.index("volume")]),
            "time": ("altitude", time),
        },
        coords={"altitude": data[:, header.index("alt")]},
    )


def get_filename_from_date(date, bimodal=True):

    folder = bimodal_folder if bimodal else unimodal_folder

    files = []
    times = []
    for inst in instruments:
        for file in os.listdir(Path(folder) / inst):  # noqa: PTH208
            if file.split(".")[-1] == "szd":
                times.append(pd.Timestamp(file.split("_")[0]))
                files.append(Path(folder) / inst / file)

    idx = np.argmin(np.abs(pd.Timestamp(date) - pd.to_datetime(times)))
    return files[idx]


def opc_aerosol(date, mode="fine", altitude: np.ndarray | None = None):

    if altitude is None:
        altitude = np.arange(0.0, 100, 0.5)

    filename = get_filename_from_date(date)
    ds = read_file(filename)

    rg = np.interp(
        altitude, ds.altitude.to_numpy(), ds[f"median_radius_{mode}"].to_numpy()
    )
    sg = np.interp(altitude, ds.altitude.to_numpy(), ds[f"width_{mode}"].to_numpy())
    n = np.interp(
        altitude, ds.altitude.to_numpy(), ds[f"number_density_{mode}"].to_numpy()
    )

    particle_size = {
        "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": sg,
        "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": rg,
    }
    ps_clim = sk.ClimatologyUserDefined(altitude * 1000, particle_size)
    opt_prop = sk.MieAerosol(ps_clim, "H2SO4")

    aer_clim = sk.ClimatologyUserDefined(
        altitude * 1000, {"SKCLIMATOLOGY_AEROSOL_CM3": n}
    )
    return sk.Species(opt_prop, aer_clim, species="SKCLIMATOLOGY_AEROSOL_CM3")


def load_opc_profiles(instrument="WPC", bimodal=True, altitude_res=0.5):

    folder = bimodal_folder if bimodal else unimodal_folder

    alt_edges = np.arange(5.25, 35.5, altitude_res)
    alts = alt_edges[0:-1] + np.diff(alt_edges) / 2

    ds = []
    for file in os.listdir(Path(folder) / instrument):  # noqa: PTH208
        if file.split(".")[-1] == "szd":
            time = pd.Timestamp(file.split("_")[0])
            data = read_file(Path(folder) / instrument / file)
            data = data.sortby("altitude").sel(altitude=slice(10, 35))
            data = data.isel(altitude=slice(1, len(data.altitude.to_numpy()) - 1))

            # if len(data.altitude.values) != len(np.unique(data.altitude.values)):

            data = data.groupby_bins("altitude", bins=alt_edges).mean()
            data = data.rename({"altitude_bins": "altitude"})
            data["altitude"] = alts
            data["time"] = time
            ds.append(data)

    return xr.concat(ds, dim="time")


if __name__ == "__main__":

    filename = get_filename_from_date("2011-05-05")
    ds = read_file(filename)

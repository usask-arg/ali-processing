from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr


class ACEAtmosphere:

    def __init__(self, folder: str):

        self.folder = folder

    def load_species(
        self, species, latitude: float = 0.0, altitude: None | np.ndarray = None
    ):
        if altitude is None:
            altitude = np.arange(0.0, 100000.0, 1000.0)
        month = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        data = []
        for m in month:
            file = f"ACE_monthly_zm_{species.upper()}_hpa_lat_{m}_all.nc"
            ds = xr.open_dataset(Path(self.folder) / species.upper() / file)[
                species.upper()
            ]
            if latitude is not None:
                ds = ds.sel(lat=latitude, method="nearest")
            ds["month"] = m
            data.append(ds)

        data = xr.concat(data, dim="month")

        msis = sk.MSIS90()
        pressure = (
            msis.get_parameter(
                "SKCLIMATOLOGY_PRESSURE_PA",
                latitude=latitude,
                longitude=0.0,
                altitudes=altitude,
                mjd=53000,
            )
            / 100
        )
        nd = msis.get_parameter(
            "SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3",
            latitude=latitude,
            longitude=0.0,
            altitudes=altitude,
            mjd=53000,
        )
        data = data.mean(dim="month")
        data = data.interp(plev=pressure) * nd
        data["altitude"] = xr.DataArray(altitude, coords=[data.plev], dims=["plev"])
        return data.swap_dims({"plev": "altitude"}).fillna(0.0).reset_coords(drop=True)

    def load_v4_profile(
        self,
        species,
        min_lat=-90,
        max_lat=90,
        min_time="2000-01-01",
        max_time="2022-01-01",
        altitude: None | np.ndarray = None,
    ):

        if altitude is None:
            altitude = np.arange(0.0, 100000.0, 1000.0)

        data = xr.open_dataset(Path(self.folder) / f"ACEFTS_L2_v4p0_{species}.nc")
        time = [
            pd.Timestamp(f"{int(y)}-{int(m):02}-{int(d):02}") + pd.Timedelta(h, "h")
            for y, m, d, h in zip(
                data.year.to_numpy(),
                data.month.to_numpy(),
                data.day.to_numpy(),
                data.hour.to_numpy(),
                strict=False,
            )
        ]
        data["time"] = xr.DataArray(
            time, coords=[data.index.to_numpy()], dims=["index"]
        )
        data = data.swap_dims({"index": "time"})
        data = data.where(
            (data.latitude < max_lat) & (data.latitude > min_lat), drop=True
        )
        data[species] = data[species].where(data[species] > -999.0)

        data = data.sortby("time").sel(time=slice(min_time, max_time))
        mjd = (
            data.time.mean().to_numpy() - np.datetime64("1858-11-18")
        ) / np.timedelta64(1, "D")
        # data = data.mean(dim='time')
        data = data.max(dim="time")
        msis = sk.MSIS90()
        mean_lat = (min_lat + max_lat) / 2
        # pressure = msis.get_parameter('SKCLIMATOLOGY_PRESSURE_PA', latitude=mean_lat, longitude=0.0,
        #                               altitudes=altitude, mjd=mjd) / 100
        nd = msis.get_parameter(
            "SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3",
            latitude=mean_lat,
            longitude=0.0,
            altitudes=altitude,
            mjd=mjd,
        )
        data = data.rename({species: f"{species}_vmr"})
        data[species] = data[f"{species}_vmr"].interp(
            altitude=altitude / 1000
        ) * xr.DataArray(nd, coords=[altitude / 1000], dims=["altitude"])

        return data


if __name__ == "__main__":

    atmo = ACEAtmosphere(folder=r"C:\Users\lar555\data\acefts\v3.5\monthly")
    c2h2 = atmo.load_v4_profile("C2H2", min_lat=-5, max_lat=5)
    h2o = atmo.load_species("H2O", latitude=0)

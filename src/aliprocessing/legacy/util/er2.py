from __future__ import annotations

import os

import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr
from skretrieval.core import OpticalGeometry
from skretrieval.util import rotation_matrix


def load_er2_data(
    flight,
    folder=r"C:\Users\lar555\Documents\ACCP\ER2\telemetry",
    resample="400ms",
    load=True,
):
    if load:
        try:
            data_lores = xr.open_dataset(
                os.path.join(folder, f"IWG1.{flight}_{resample}.nc")
            )
            return data_lores
        except FileNotFoundError:
            load = False

    data = (
        pd.read_csv(
            os.path.join(folder, f"IWG1.{flight}.txt"),
            index_col=0,
            usecols=[1, 2, 3, 4, 8, 13, 14, 16, 17],
            names=[
                "time",
                "latitude",
                "longitude",
                "altitude",
                "speed",
                "heading",
                "track",
                "pitch_angle",
                "roll_angle",
            ],
            parse_dates=True,
            skiprows=11863,
        )
        .to_xarray()
        .sortby("time")
    )
    data = data.where(
        np.isfinite(data.latitude) & np.isfinite(data.roll_angle), drop=True
    )
    data_lores = data.resample(time=resample).mean()
    data_lores.to_netcdf(os.path.join(folder, f"IWG1.{flight}_{resample}.nc"))
    return data_lores


def er2_orientation(er2, instrument_pitch_deg=0.0):
    geo = sk.Geodetic()
    heading = float(er2.heading.values)
    pitch = float(er2.pitch_angle.values)
    roll = float(er2.roll_angle.values)
    geo.from_lat_lon_alt(
        float(er2.latitude.values),
        float(er2.longitude.values),
        float(er2.altitude.values),
    )
    pos = geo.location
    up = geo.local_up
    north = -geo.local_south
    look = north @ rotation_matrix(up, -heading * np.pi / 180)
    hor = np.cross(look, up)
    look = look @ rotation_matrix(hor, -(pitch + instrument_pitch_deg) * np.pi / 180)
    up = up @ rotation_matrix(look, roll * np.pi / 180)
    up = up @ rotation_matrix(hor, -(pitch + instrument_pitch_deg) * np.pi / 180)
    mjd = (er2.time.values - np.datetime64("1858-11-17")) / np.timedelta64(1, "D")
    # look = rotate(north, up, -heading, deg=True)
    # hor = np.cross(look, up)
    # look = rotate(look, hor, pitch, deg=True)
    # up = rotate(up, look, roll, deg=True)
    # up = rotate(up, hor, pitch, deg=True)
    return OpticalGeometry(observer=pos, look_vector=look, local_up=up, mjd=mjd)

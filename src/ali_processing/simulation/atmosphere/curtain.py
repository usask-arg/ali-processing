from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr
from skretrieval.core import OpticalGeometry


class SimulationAtmosphere:

    def __init__(
        self,
        filename,
        angular_resoluiton=0.5,
        vertical_resolution=0.5,
        cloud_scaling=1.0,
    ):
        """
        Code to create a sasktran atmosphere using a file containing OMPS, CALIPSO and ERA5 data.

        Parameters
        ----------
        filename : str
            name of the netcdf file
        angular_resoluiton : float
            angular resolution in degrees that the calipso, omps, and era5 datasets will be sampled at
        vertical_resolution : float
            vertical resolution in km that the calipso, omps, and era5 datasets will be sampled at

        Examples
        --------

        To setup a sasktran atmosphere with 2D climatologies::

            >>> atmo = SimulationAtmosphere(file)
            >>> atmo.sasktran_atmosphere(aerosol=True, cloud=True, h2o=False)


        If you would like to setup a 1D climatology you can sample the climatologies at the desired locations.
        Locations along the orbit are specified in terms of the angular position from 0.0 deg latitude::

            >>> atmo = SimulationAtmosphere(file)
            >>> angle = 20.0
            >>> lat = atmo.latitude(angle)
            >>> lon = atmo.longitude(angle)
            >>> mjd = atmo.mjd(angle)
            >>> profile = atmo.cloud.climatology.get_parameter('iceloud', lat, lon, altitudes, mjd)


        To obtain a limb satellite geometry that samples the atmosphere at an angle of 20deg, and altitude of 17.5km
        you can use::

            >>> atmo = SimulationAtmosphere(file)
            >>> optical_geometry = atmo.optical_geometry(orbit_angle=20.0, \
            >>>                                          altitude=17.5, \
            >>>                                          satellite_altitude=500.0)

        """
        self._file = filename
        self.angular_resolution = angular_resoluiton
        self.vertical_resolution = vertical_resolution
        self.max_altitude = 45
        self.min_altitude = 0
        self.max_angle = 90
        self.min_angle = 0

        self.cloud_scaling = cloud_scaling
        self.particle_size_2d = False
        self.cloud_effective_radius = 70
        self.max_cloud_extinction = None

        self._cloud = None
        self._aerosol = None
        self._water = None

        self._rg = 0.08
        self._sg = 1.6

        self._calipso_latitude = None
        self._calipso_longitude = None
        self._calipso_angles = None

    def angle_from_latitude(self, target_latitude):
        lat, lon, angles = self._calipso_position()
        return np.interp(target_latitude, lat, angles)

    def angle_from_mjd(self, target_mjd):
        calipso = xr.open_dataset(self._file, group="CALIPSO")
        time = calipso.time.to_numpy()
        mjd = (time - np.datetime64("1858-11-17")) / np.timedelta64(1, "D")
        lat, lon, angles = self._calipso_position()
        return np.interp(target_mjd, mjd, angles)

    @property
    def orbit_angle(self):
        return self.orbit_angle_bins[0:-1] + np.diff(self.orbit_angle_bins) / 2

    @property
    def altitude(self):
        return self.altitude_bins[0:-1] + np.diff(self.altitude_bins) / 2

    @property
    def orbit_angle_bins(self):
        return np.arange(
            self.min_angle,
            self.max_angle + self.angular_resolution / 2,
            self.angular_resolution,
        )

    @property
    def altitude_bins(self):
        return np.arange(
            self.min_altitude,
            self.max_altitude + self.vertical_resolution / 2,
            self.vertical_resolution,
        )

    @property
    def normal_vector(self):

        # idx = self.angle_from_latitude(self.ref_latitude)
        zero_angle = 0.0  # self.angle_from_latitude(0.0)

        geo = sk.Geodetic()
        geo.from_lat_lon_alt(self.latitude(zero_angle), self.longitude(zero_angle), 0.0)
        l0 = geo.location
        geo.from_lat_lon_alt(
            self.latitude(zero_angle + 1), self.longitude(zero_angle + 1), 0.0
        )
        l1 = geo.location
        orbit_dir = l1 - l0
        orbit_dir /= np.linalg.norm(orbit_dir)
        return np.cross(orbit_dir, l0 / np.linalg.norm(l0))

    @property
    def reference_vector(self):
        geo = sk.Geodetic()
        angle = self.angle_from_latitude(0.0)
        geo.from_lat_lon_alt(self.latitude(angle), self.longitude(angle), 0.0)
        return geo.location / np.linalg.norm(geo.location)

    @property
    def aerosol(self):

        if self._aerosol is None:
            opt_prop = self.aerosol_opt_prop()
            xsec = self.aerosol_xsec()
            extinction = self._downsample_omps()
            aerosol = extinction.transpose("angle", "altitude") / xsec * 1e-5
            aerosol = aerosol.where(np.isfinite(aerosol)).fillna(0.0)
            aer_clim = sk.ClimatologyUserDefined2D(
                self.orbit_angle,
                self.altitude * 1000,
                {"SKCLIMATOLOGY_AEROSOL_CM3": aerosol.to_numpy()},
                self.reference_vector,
                self.normal_vector,
            )
            self._aerosol = sk.Species(
                opt_prop, aer_clim, species="SKCLIMATOLOGY_AEROSOL_CM3"
            )
            # sk.GloSSAC()
        return self._aerosol

    def aerosol_curtain(self):
        return self._downsample_omps()

    def aerosol_opt_prop(self):

        ones = np.ones((len(self.orbit_angle), len(self.altitude)))
        particle_size = {
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": self._sg * ones,
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": self._rg * ones,
        }
        ps_clim = sk.ClimatologyUserDefined2D(
            self.orbit_angle,
            self.altitude * 1000,
            particle_size,
            self.reference_vector,
            self.normal_vector,
        )
        return sk.MieAerosol(ps_clim, "H2SO4")

    def aerosol_xsec(self):
        ones = np.ones((len(self.orbit_angle), len(self.altitude)))
        particle_size = {
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": self._sg * ones,
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": self._rg * ones,
        }
        ps_clim = sk.ClimatologyUserDefined2D(
            self.orbit_angle,
            self.altitude * 1000,
            particle_size,
            self.reference_vector,
            self.normal_vector,
        )
        opt_prop = sk.MieAerosol(ps_clim, "H2SO4")
        if self.particle_size_2d:
            xsec = np.ones((len(self.orbit_angle), len(self.altitude)))
            for angle_idx, angle in enumerate(self.orbit_angle):
                lat = self.latitude(angle)
                lon = self.longitude(angle)
                for idx, alt in enumerate(self.altitude):
                    xsec[angle_idx, idx] = opt_prop.calculate_cross_sections(
                        sk.MSIS90(),
                        latitude=lat,
                        longitude=lon,
                        altitude=alt * 1000,
                        mjd=53000.0,
                        wavelengths=750.0,
                    ).total[0]
            xsec = xr.DataArray(
                xsec,
                coords=[self.orbit_angle, self.altitude],
                dims=["angle", "altitude"],
            )
        else:
            xsec = np.ones((len(self.orbit_angle), len(self.altitude)))
            for idx, alt in enumerate(self.altitude):
                xsec[:, idx] = opt_prop.calculate_cross_sections(
                    sk.MSIS90(),
                    latitude=self.latitude(10),
                    longitude=self.longitude(10),
                    altitude=alt * 1000,
                    mjd=53000.0,
                    wavelengths=750.0,
                ).total[0]
        xsec[xsec == 0.0] = np.min(xsec[xsec > 0])
        return xsec

    def set_particle_size(
        self,
        rg: float | np.ndarray,
        sg: float | np.ndarray,
        altitude_m: np.ndarray = None,
    ):

        if isinstance(type(rg), float):
            self._rg = np.ones((len(self.orbit_angle), len(self.altitude))) * rg
            self._sg = np.ones((len(self.orbit_angle), len(self.altitude))) * sg
        elif isinstance(rg, np.ndarray):
            if len(rg.shape) == 1:
                rg = np.interp(self.altitude * 1000, altitude_m, rg)
                sg = np.interp(self.altitude * 1000, altitude_m, sg)
                self._rg = (
                    np.ones((len(self.orbit_angle), len(self.altitude)))
                    * rg[np.newaxis, :]
                )
                self._sg = (
                    np.ones((len(self.orbit_angle), len(self.altitude)))
                    * sg[np.newaxis, :]
                )
            else:
                self._rg = np.ones((len(self.orbit_angle), len(self.altitude))) * rg
                self._sg = np.ones((len(self.orbit_angle), len(self.altitude))) * sg

    @property
    def cloud(self):

        if self._cloud is None:
            opt_prop = sk.BaumIceCrystal(self.cloud_effective_radius)
            xsec = opt_prop.calculate_cross_sections(
                sk.MSIS90(),
                latitude=self.latitude(10),
                longitude=self.longitude(10),
                altitude=1000.0,
                mjd=53000.0,
                wavelengths=750.0,
            ).total[0]
            nd = self.cloud_curtain(extinction=False)
            nd = nd.fillna(0.0)
            if self.max_cloud_extinction is not None:
                max_nd = self.max_cloud_extinction / xsec * 1e-5
                nd = nd.clip(max=max_nd)
            cloud_clim = sk.ClimatologyUserDefined2D(
                self.orbit_angle,
                self.altitude * 1000,
                {"icecloud": nd.transpose("angle", "altitude").to_numpy()},
                self.reference_vector,
                self.normal_vector,
            )
            self._cloud = sk.Species(opt_prop, cloud_clim)

        return self._cloud

    def cloud_curtain(self, extinction=True, wavelength=750.0):
        cloud = self._downsample_calipso()
        opt_prop = sk.BaumIceCrystal(self.cloud_effective_radius)
        xsec = opt_prop.calculate_cross_sections(
            sk.MSIS90(),
            latitude=self.latitude(10),
            longitude=self.longitude(10),
            altitude=1000.0,
            mjd=53000.0,
            wavelengths=wavelength,
        ).total[0]
        p = opt_prop.calculate_phase_matrix(
            sk.MSIS90(),
            latitude=self.latitude(10),
            longitude=self.longitude(10),
            altitude=1000.0,
            mjd=53000.0,
            wavelengths=[wavelength],
            cosine_scatter_angles=[-1],
        )[0, 0]
        bs_to_ext = (1 / p[0, 0]) * np.pi * 4 * self.cloud_scaling
        if extinction:
            return cloud * bs_to_ext

        return cloud * bs_to_ext / xsec * 1e-5

    @property
    def water_vapour(self):
        if self._water is None:
            h2o = self._downsample_h2o()
            opt_prop = sk.HITRANChemical("h2o")
            clim = sk.ClimatologyUserDefined2D(
                self.orbit_angle,
                self.altitude * 1000,
                {"h2o": h2o.transpose("angle", "altitude").to_numpy()},
                self.reference_vector,
                self.normal_vector,
            )
            self._water = sk.Species(opt_prop, clim)
        return self._water

    @property
    def air(self):

        return sk.Species(sk.Rayleigh(), sk.MSIS90())

    def optical_geometry(
        self, orbit_angle, altitude=10.0, satellite_altitude=500.0, look_forward=True
    ):

        lat = self.latitude(orbit_angle)
        lon = self.longitude(orbit_angle)
        time = self.time(orbit_angle)
        mjd = (time - pd.Timestamp("1858-11-17")) / pd.Timedelta(1, "D")
        tp = sk.Geodetic()
        tp.from_lat_lon_alt(lat, lon, altitude * 1000)
        tp = tp.location
        normal = self.normal_vector

        look = np.cross(tp / np.linalg.norm(tp), normal)
        if not look_forward:
            look = -look
        re = 6372
        tp_to_sat = (
            np.sqrt((re + satellite_altitude) ** 2 - (re + altitude) ** 2) * 1000
        )
        obs = tp - look * tp_to_sat
        hor = np.cross(look, obs / np.linalg.norm(obs))
        up = np.cross(hor, look)

        geo = sk.Geodetic()
        geo.from_tangent_point(obs, look)
        lat_diff = lat - geo.latitude
        lon_diff = lon - geo.longitude
        if (np.abs(lat_diff) > 0.2) or (np.abs(lon_diff) % 360 > 0.2):
            # raise ValueError('could not match tp position')
            logging.debug("lat difference: {lat_diff}, lon difference {lon_diff}")

        if geo.longitude > 350:
            logging.debug("high longitude")

        return OpticalGeometry(observer=obs, look_vector=look, local_up=up, mjd=mjd)

    def sasktran_atmosphere(self, aerosol=True, cloud=True, h2o=True):

        sk_atmo = sk.Atmosphere()
        sk_atmo["air"] = self.air
        sk_atmo.brdf = 0.2

        if aerosol:
            sk_atmo["aerosol"] = self.aerosol

        if cloud:
            sk_atmo["icecloud"] = self.cloud

        if h2o:
            sk_atmo["h2o"] = self.water_vapour

        return sk_atmo

    def _downsample_h2o(self):
        era = xr.open_dataset(self._file, group="ERA5")
        angle_along_orbit = self._angle_along_orbit(
            np.asarray(era.latitude.to_numpy(), dtype=float),
            np.asarray(era.longitude.to_numpy(), dtype=float),
        )
        nd = (era.pressure * 100) / (era.temperature * 1.38064852e-23) / (100**3)
        m_air = 28.9647
        m_h2o = 18.02
        h2o = era.specific_humidity / (1 - era.specific_humidity) * nd * (m_air / m_h2o)

        h2o = xr.DataArray(
            h2o.where(h2o > 0).transpose("altitude", "time").to_numpy(),
            coords=(era.altitude.to_numpy(), angle_along_orbit),
            dims=["altitude", "angle"],
        ).fillna(0.0)

        # TODO: extend h2o clim above 40km using MSIS90?

        return self._downsample_to_atmo_grid(h2o).fillna(0.0)

    def latitude(self, orbit_angle):
        lat, lon, angles = self._calipso_position()
        return np.interp(orbit_angle, angles, lat)

    def longitude(self, orbit_angle):
        lat, lon, angles = self._calipso_position()
        lons = np.interp(orbit_angle, angles, lon)
        lon[lon < 0] = lon[lon < 0] + 360
        lons2 = np.interp(orbit_angle, angles, lon)
        try:
            lons2[lons2 > 180] = lons2[lons2 > 180] - 360
        except TypeError:
            if lons2 > 180:
                lons2 = lons2 - 360
        if np.abs(lons2 - lons) % 360.0 > 0.01:
            msg = f"Longitude are separated by more than 360 degrees: difference of {lons2-lons}"
            raise ValueError(msg)
        return lons

    def time(self, orbit_angle):
        mjd = self.mjd(orbit_angle)
        return pd.Timedelta(mjd, "D") + pd.Timestamp("1858-11-17")

    def mjd(self, orbit_angle):
        calipso = xr.open_dataset(self._file, group="CALIPSO")
        lat, lon, angles = self._calipso_position()
        time = calipso.time.to_numpy()

        mjds = (time - np.datetime64("1858-11-17")) / np.timedelta64(1, "D")
        return np.interp(orbit_angle, angles, mjds)

    def _calipso_position(self):
        if self._calipso_angles is None:
            calipso = xr.open_dataset(self._file, group="CALIPSO")
            self._calipso_latitude = np.asarray(
                calipso.latitude.to_numpy(), dtype=float
            )
            self._calipso_longitude = np.asarray(
                calipso.longitude.to_numpy(), dtype=float
            )
            self._calipso_longitude[667] = -179.99
            self._calipso_angles = self._angle_along_orbit(
                self._calipso_latitude, self._calipso_longitude, 0.0
            )
        return self._calipso_latitude, self._calipso_longitude, self._calipso_angles

    def _downsample_to_atmo_grid(self, array):

        array = array.groupby_bins(array.altitude, bins=self.altitude_bins).mean()
        array = array.groupby_bins(array.angle, bins=self.orbit_angle_bins).mean()
        array["altitude_bins"] = self.altitude
        array["angle_bins"] = self.orbit_angle
        return (
            array.sortby("angle_bins")
            .sortby("altitude_bins")
            .rename({"altitude_bins": "altitude", "angle_bins": "angle"})
        )

    def _angle_along_orbit(self, latitude, longitude, zero_lat=0.0):

        lat0 = zero_lat
        lon0 = np.interp(zero_lat, latitude, longitude)
        geo = sk.Geodetic()
        geo.from_lat_lon_alt(lat0, lon0, 0.0)
        xyz0 = geo.location / np.linalg.norm(geo.location)
        angle_along_orbit = np.ones_like(latitude)
        for idx, (lat, lon) in enumerate(zip(latitude, longitude, strict=False)):
            geo.from_lat_lon_alt(lat, lon, 0.0)
            xyz = geo.location / np.linalg.norm(geo.location)
            if lat < lat0:
                angle_along_orbit[idx] = -np.arccos(np.dot(xyz, xyz0))
            else:
                angle_along_orbit[idx] = np.arccos(np.dot(xyz, xyz0))

        return angle_along_orbit * 180 / np.pi

    def _downsample_calipso(self):

        calipso = xr.open_dataset(self._file, group="CALIPSO")
        angle_along_orbit = self._angle_along_orbit(
            np.asarray(calipso.latitude.to_numpy(), dtype=float),
            np.asarray(calipso.longitude.to_numpy(), dtype=float),
        )
        cloud = xr.DataArray(
            calipso.cloud_backscatter.where(calipso.cloud_backscatter > 0).to_numpy(),
            coords=(calipso.altitude.to_numpy(), angle_along_orbit),
            dims=["altitude", "angle"],
        ).fillna(0.0)

        return self._downsample_to_atmo_grid(cloud)

    def _downsample_omps(self):
        omps = xr.open_dataset(self._file, group="OMPS")
        angle_along_orbit = self._angle_along_orbit(
            np.asarray(omps.latitude.to_numpy(), dtype=float),
            np.asarray(omps.longitude.to_numpy(), dtype=float),
        )

        aerosol = xr.DataArray(
            omps.extinction.to_numpy().T,
            coords=(omps.altitude.to_numpy(), angle_along_orbit),
            dims=["altitude", "angle"],
        )
        # tropopause = xr.DataArray(
        #     omps.tropopause_altitude.to_numpy().T,
        #     coords=[angle_along_orbit],
        #     dims=["angle"],
        # )

        return aerosol.interp(altitude=self.altitude).interp(angle=self.orbit_angle)
        # tropopause = tropopause.interp(angle=self.orbit_angle)
        #
        # # fill nans then decay aerosol below tropopause and above upper bound
        # min_altitude = (aerosol > 0) * aerosol.altitude
        # min_altitude = (
        #     min_altitude.where(min_altitude > 0).min(dim="altitude").to_numpy()
        # )
        # max_altitude = (aerosol > 0) * aerosol.altitude
        # max_altitude = (
        #     max_altitude.where(max_altitude > 0).max(dim="altitude").to_numpy()
        # )
        # alts = aerosol.altitude.to_numpy()
        # for profile, min_alt, max_alt, trop in zip(
        #     aerosol.transpose("angle", "altitude").to_numpy(),
        #     min_altitude,
        #     max_altitude,
        #     tropopause.to_numpy(),
        #     strict=False,
        # ):
        #     min_idx = np.where(alts >= min_alt)[0][0]
        #     max_idx = np.where(alts <= max_alt)[0][-1]
        #     profile[alts < min_alt] = profile[min_idx]
        #     scale = np.exp(-1 * alts)
        #     scale *= profile[max_idx] / scale[max_idx]
        #     profile[alts > max_alt] = scale[alts > max_alt]
        #     alt_below_trop = alts - trop
        #     alt_below_trop[alt_below_trop > 0] = 0
        #     alt_below_trop[alt_below_trop < -3] = -3
        #     alt_below_trop *= 1 / 3
        #     alt_below_trop += 1
        #     profile *= alt_below_trop
        #
        # return aerosol

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr
from ali.util.config import Config

folder = Path.parent(__file__)
plt.style.use(Config.MATPLOTLIB_STYLE_FILE)
gem_file = r"C:\users\lar555\data\accp\GEM runs\GEM_201308251946_midlat_profile_data.nc"
gem_file_tropics = (
    r"C:\users\lar555\data\accp\GEM runs\GEM_201505160650_tropic_profile_data.nc"
)


class GEMAtmosphere:

    def __init__(self, tropics: bool = False, point: int = 0):
        """
        GEM atmosphere

        Parameters
        ----------
        tropics:
            Tropics or mid-latitude case. Defaults to False (mid-latitude)
        point:
            Which profile of the GEM atmosphere to use, 0-2, defaults to 0.
        """

        if tropics:
            self.gem_file = r"C:\users\lar555\data\accp\GEM runs\GEM_201505160650_tropic_profile_data.nc"
        else:
            self.gem_file = r"C:\users\lar555\data\accp\GEM runs\GEM_201308251946_midlat_profile_data.nc"

        self.point = point

    @property
    def latitude(self):
        data = xr.open_dataset(self.gem_file).sel(point=self.point)
        return float(data.lat.values)

    @property
    def time(self) -> np.datetime64:
        data = xr.open_dataset(self.gem_file).sel(point=self.point)
        ymd = Path.name(self.gem_file).split("_")[1]
        year = ymd[0:4]
        month = ymd[4:6]
        day = ymd[6:8]
        # hour = ymd[8:10]
        # min = ymd[10:]
        if type(data.time.to_numpy()) is np.timedelta64:
            ymd = np.datetime64(f"{year}-{month}-{day}") + data.time.to_numpy()
        else:
            ymd = np.datetime64(f"{year}-{month}-{day}") + np.timedelta64(
                int(data.time.to_numpy()[0]), "m"
            )
        return ymd

    @property
    def mjd(self):
        return self.time - np.datetime64("1858-11-18")

    @property
    def longitude(self):
        data = xr.open_dataset(self.gem_file).sel(point=self.point)
        return float(data.lon.to_numpy())

    def species_liquid_cloud(self, altitudes: None | np.ndarray = None, radius=10.0):
        if altitudes is None:
            altitudes = np.arange(0.0, 100000, 250.0)

        data = xr.open_dataset(self.gem_file).sel(point=self.point)
        alt = data.GZ.to_numpy()
        density = np.interp(altitudes / 1000, alt, data.liquid.to_numpy())

        volume_per_particle = 4 / 3 * np.pi * ((radius * 1e-6) ** 3)  # m^3
        mass_per_m3 = 1000000.0  # g
        mass_per_particle = volume_per_particle * mass_per_m3  # g
        profile = density / mass_per_particle * 1e-6  # cm^-3

        return sk.SpeciesAerosol(
            altitudes,
            {"SKCLIMATOLOGY_WATER_CM3": profile},
            {
                "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": np.ones_like(altitudes)
                * radius,
                "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": np.ones_like(altitudes) * 1.001,
            },
            species="WATER",
        )

    def species_ice_cloud(self, altitudes: None | np.ndarray = None, radius=22.0):
        if altitudes is None:
            altitudes = np.arange(0.0, 100000, 250.0)

        data = xr.open_dataset(self.gem_file).sel(point=self.point)
        alt = data.GZ.to_numpy()
        density = np.interp(altitudes / 1000, alt, data.solid.to_numpy())

        opt_prop = sk.BaumIceCrystal(radius * 2.0)

        volume_per_particle = 4 / 3 * np.pi * ((radius * 1e-6) ** 3)  # m^3
        mass_per_m3 = 1000000.0  # g
        mass_per_particle = volume_per_particle * mass_per_m3  # g
        profile = density / mass_per_particle * 1e-6  # cm^-3

        cloud_clim = sk.ClimatologyUserDefined(altitudes, {"icecloud": profile})
        return sk.Species(opt_prop, cloud_clim)

    def species_aerosol(
        self,
        altitudes: None | np.ndarray = None,
        radius: float = 0.08,
        width: float = 1.6,
        date: str | None = None,
        latitude: float | None = None,
    ) -> sk.SpeciesAerosol:
        """
        Return a 1D sasktran aerosol species at the desire altitudes. Uses the GloSSAC climatology for the profiles.

        Parameters
        ----------
        altitudes:
            Climatology altitudes in meters
        radius:
            Lognormal median radius in microns. Defaults to 0.08 microns
        width:
            Lognormal width. Defaults to 1.60.
        date:
            Date to use for the aerosol profile. Defaults to the GEM file time.
        latitude:
            Latitude to use for the aerosol profile. Defaults to the GEM file latitude.

        Returns
        -------
            sk.SpeciesAerosol
        """

        if altitudes is None:
            altitudes = np.arange(0.0, 100000, 250.0)

        if latitude is None:
            latitude = self.latitude

        if date is None:
            date = self.time

        mjd = (pd.Timestamp(date) - pd.Timestamp("1858-11-18")) / pd.Timedelta(1, "D")
        particle_size_values = {
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": np.ones_like(altitudes)
            * radius,
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": np.ones_like(altitudes) * width,
        }
        glossac = sk.SpeciesAerosolGloSSAC(
            extend=True, altitudes=altitudes, particle_size_values=particle_size_values
        )
        # glossac._climatology.decay_above = 1e-2
        ext = glossac.get_parameter(
            "SKCLIMATOLOGY_AEROSOL_EXTINCTIONPERKM",
            latitude=latitude,
            longitude=0,
            mjd=mjd,
            altitudes=altitudes,
        )
        xsec = glossac.optical_property.calculate_cross_sections(
            sk.MSIS90(),
            latitude=latitude,
            longitude=0.0,
            altitude=1000.0,
            mjd=mjd,
            wavelengths=glossac.climatology.wavelength_nm,
        ).total[0]
        aerosol = ext / xsec * 1e-5

        return sk.SpeciesAerosol(
            altitudes,
            {"SKCLIMATOLOGY_AEROSOL_CM3": aerosol},
            particle_size_values,
            species="H2SO4",
        )

    def species_water_vapour(self, altitudes: None | np.ndarray = None):

        if altitudes is None:
            altitudes = np.arange(0.0, 100000, 250.0)

        data = xr.open_dataset(self.gem_file).sel(point=self.point)
        alt = data.GZ.to_numpy()
        wv = data.HU.to_numpy()
        pressure = data.PP.to_numpy() * 100  # Pa
        temp = data.TT.to_numpy()

        k = 1.38064852e-23  # m2 kg s-2 K-1
        air_density = pressure / (k * temp) * 1e-6

        density = air_density * wv
        density_interp = np.exp(np.interp(altitudes / 1000, alt, np.log(density)))
        # wv_interp = np.exp(np.interp(altitudes / 1000, alt, np.log(wv)))

        opt_prop = sk.HITRANChemical("H2O")
        clim = sk.ClimatologyUserDefined(altitudes, {"water": density_interp})

        # fig, ax = plt.subplots(1, 2, figsize=(4, 4), dpi=200, sharey=True)
        # fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.97, wspace=0.08)
        # ax[0].plot(wv * 1e6, alt, label='gem altitudes')
        # ax[0].plot(wv_interp * 1e6, altitudes / 1000, label='interpolated')
        # ax[1].plot(density, alt, label='gem altitudes')
        # ax[1].plot(density_interp, altitudes / 1000, label='interpolated')
        #
        # ax[0].set_xlim(0, 10)
        # ax[1].set_xlim(0, 2e13)
        # ax[0].set_title('Mixing Ratio')
        # ax[1].set_title('Number Denstity')
        # ax[0].set_xlabel('ppm')
        # ax[1].set_xlabel('cm$^{-3}$')
        # ax[0].set_ylim(15, 22)
        # ax[0].set_ylabel('Altitude [km]')
        # ax[0].legend()
        return sk.Species(opt_prop, clim, species="water")


def make_figure():
    # data = xr.open_dataset(gem_file)
    data = xr.open_dataset(gem_file_tropics)

    blue = "#438cc4"
    grey = "#777777"
    red = "#9e3928"

    point = 0
    fig, ax = plt.subplots(1, 3, figsize=(4, 3), dpi=200, sharey=True)
    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.97, wspace=0.02)
    # ax.plot(data.sel(point=0).liquid.values, data.sel(point=0).GZ.values)
    ax[0].fill_betweenx(
        data.sel(point=point).GZ.to_numpy(),
        data.sel(point=point).liquid.to_numpy(),
        color=blue,
    )
    # ax.plot(data.sel(point=0).solid.values, data.sel(point=0).GZ.values, color='#444444')
    ax[0].fill_betweenx(
        data.sel(point=point).GZ.to_numpy(),
        data.sel(point=point).solid.to_numpy(),
        color=grey,
    )

    ax[0].set_ylabel("Altitude [km]")
    ax[0].set_xlabel("Concentration [g/m$^3$]")
    ax[1].set_xlabel("Mixing Ratio [kg/kg]")
    ax[2].set_xlabel("Degrees [K]")
    ax[0].set_xlim(0, 0.5)
    ax[0].set_ylim(0, 22.5)
    ax[1].set_xlim(1e-7, 1e-1)
    ax[2].set_xlim(180, 300)
    ax[0].set_xticks([0, 0.2, 0.4])
    ax[2].set_xticks([200, 250, 300])
    ax[1].set_xscale("log")
    ax[1].set_xticks([1e-6, 1e-4, 1e-2])

    bbox = {
        "facecolor": ax[0].get_facecolor(),
        "edgecolor": "none",
        "boxstyle": "square,pad=0.2",
    }
    ax[0].text(
        0.20,
        15.4,
        "Ice Content",
        color=grey,
        fontsize="small",
        fontweight="bold",
        bbox=bbox,
    )
    ax[0].text(
        0.21,
        6.45,
        "Water Content",
        color=blue,
        fontsize="small",
        fontweight="bold",
        bbox=bbox,
    )

    ax[1].plot(
        data.sel(point=point).HU.to_numpy(),
        data.sel(point=point).GZ.to_numpy(),
        color=blue,
    )

    ax[2].plot(
        data.sel(point=point).TT.to_numpy(),
        data.sel(point=point).GZ.to_numpy(),
        color=red,
    )

    ax[0].text(
        0.5,
        0.98,
        "Clouds",
        va="top",
        ha="center",
        transform=ax[0].transAxes,
        color="#444444",
        fontsize="x-large",
        bbox=bbox,
    )
    ax[1].text(
        0.5,
        0.98,
        "Water Vapour",
        va="top",
        ha="center",
        transform=ax[1].transAxes,
        color="#444444",
        fontsize="x-large",
        bbox=bbox,
    )
    ax[2].text(
        0.5,
        0.98,
        "Temperature",
        va="top",
        ha="center",
        transform=ax[2].transAxes,
        color="#444444",
        fontsize="x-large",
        bbox=bbox,
    )

    fig.savefig(Path(Config.DIAGNOSTIC_FIGURE_FOLDER) / "gem_profiles.png", dpi=450)


if __name__ == "__main__":
    # gem = GEMAtmosphere()
    # gem.point = 2
    # gem.species_water_vapour(np.arange(0.0, 100000, 250.0))
    make_figure()

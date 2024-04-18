from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr

from ali_processing.legacy.util.config import Config
from ali_processing.simulation.atmosphere.curtain import SimulationAtmosphere
from ali_processing.simulation.atmosphere.opc import load_opc_profiles


def aerosol_from_atmosphere(
    atmo: sk.Atmosphere,
    altitude: np.ndarray | None = None,
    wavelength: np.ndarray | list[float] | None = None,
    species_name="aerosol",
    latitude: float = 0.0,
    longitude: float = 0.0,
    mjd: float = 53000.0,
) -> xr.Dataset:
    if wavelength is None:
        wavelength = np.array([525.0, 750.0, 1020.0, 1500.0])

    if altitude is None:
        altitude = atmo[species_name].climatology.altitudes

    clim = atmo[species_name].climatology
    opt = atmo[species_name].optical_property

    species = atmo[species_name].climatology.supported_species()[0]
    n = clim.get_parameter(
        species, altitudes=altitude, latitude=latitude, longitude=longitude, mjd=mjd
    )

    opt_clim = atmo[species_name].optical_property.particlesize_climatology
    rg = opt_clim.get_parameter(
        "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS",
        altitudes=altitude,
        latitude=latitude,
        longitude=longitude,
        mjd=mjd,
    )
    sg = opt_clim.get_parameter(
        "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH",
        altitudes=altitude,
        latitude=latitude,
        longitude=longitude,
        mjd=mjd,
    )

    xsec = np.ones((len(altitude), len(wavelength)))
    for aidx, alt in enumerate(altitude):
        for widx, wavel in enumerate(wavelength):
            xsec[aidx, widx] = opt.calculate_cross_sections(
                sk.MSIS90(),
                latitude=latitude,
                longitude=longitude,
                altitude=alt,
                mjd=mjd,
                wavelengths=wavel,
            ).total[0]

    return xr.Dataset(
        {
            "number_density": (["altitude"], n),
            "extinction": (["altitude", "wavelength"], n[:, np.newaxis] * xsec * 1e5),
            "lognormal_median_radius": (["altitude"], rg),
            "lognormal_width": (["altitude"], sg),
        },
        coords={
            "altitude": (["altitude"], altitude),
            "wavelength": (["wavelength"], wavelength),
        },
    )


def ice_from_atmosphere(
    atmo: sk.Atmosphere,
    altitude: np.ndarray | None = None,
    wavelength: np.ndarray | list[float] | None = None,
    species_name="icecloud",
    latitude: float = 0.0,
    longitude: float = 0.0,
    mjd: float = 53000.0,
):
    if wavelength is None:
        wavelength = np.array([525.0, 750.0, 1020.0, 1500.0])

    if altitude is None:
        altitude = atmo[species_name].climatology.altitudes

    clim = atmo[species_name].climatology
    opt = atmo[species_name].optical_property

    species = atmo[species_name].climatology.supported_species()[0]
    n = clim.get_parameter(
        species, altitudes=altitude, latitude=latitude, longitude=longitude, mjd=mjd
    )

    xsec = np.ones((len(altitude), len(wavelength)))
    for aidx, alt in enumerate(altitude):
        for widx, wavel in enumerate(wavelength):
            xsec[aidx, widx] = opt.calculate_cross_sections(
                sk.MSIS90(),
                latitude=latitude,
                longitude=longitude,
                altitude=alt,
                mjd=mjd,
                wavelengths=wavel,
            ).total[0]

    return xr.Dataset(
        {
            "number_density": (["altitude"], n),
            "extinction": (["altitude", "wavelength"], n[:, np.newaxis] * xsec * 1e5),
            "effective_size_microns": (opt._effective_size_microns),
        },
        coords={
            "altitude": (["altitude"], altitude),
            "wavelength": (["wavelength"], wavelength),
        },
    )


def atmosphere_to_xarray(
    atmo: sk.atmosphere,
    altitudes: np.ndarray | None = None,
    aerosol_species: str = "aerosol",
    ice_species: str = "icecloud",
    water_species: str = "water",
    latitude: float = 0.0,
    longitude: float = 0.0,
    mjd: float = 53000.0,
):
    ds = {}

    aer = aerosol_from_atmosphere(
        atmo,
        altitude=altitudes,
        species_name=aerosol_species,
        latitude=latitude,
        longitude=longitude,
        mjd=mjd,
    )
    aer.attrs = {"title": "Sulphate aerosol"}
    aer.extinction.attrs = {"standard_name": "extinction", "units": "km-1"}
    aer.lognormal_median_radius.attrs = {
        "standard_name": "median_radius",
        "long_name": "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS",
        "units": "microns",
    }
    aer.lognormal_width.attrs = {
        "standard_name": "width",
        "long_name": "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH",
        "units": "unitless",
    }
    aer.number_density.attrs = {
        "standard_name": "concentration",
        "units": "cm-3",
        "long_name": "SKCLIMATOLOGY_AEROSOL_CM3",
    }
    ds[aerosol_species] = aer

    if "icecloud" in atmo.species:
        ice = ice_from_atmosphere(
            atmo,
            altitude=altitudes,
            species_name=ice_species,
            latitude=latitude,
            longitude=longitude,
            mjd=mjd,
        )
        ice.attrs["title"] = "Ice water cloud"
        ice.extinction.attrs = {"standard_name": "extinction", "units": "km-1"}
        ice.number_density.attrs = {
            "standard_name": "concentration",
            "units": "cm-3",
            "long_name": "SKCLIMATOLOGY_AEROSOL_CM3",
        }
        ice.altitude.attrs = {"units": "meters"}
        ds[ice_species] = ice

    if "water" in atmo.species:
        water = aerosol_from_atmosphere(
            atmo,
            species_name=water_species,
            altitude=altitudes,
            latitude=latitude,
            longitude=longitude,
            mjd=mjd,
        )
        water.attrs = {"title": "Liquid water cloud"}
        water.extinction.attrs = {"standard_name": "extinction", "units": "km-1"}
        water.number_density.attrs = {
            "standard_name": "concentration",
            "units": "cm-3",
            "long_name": "SKCLIMATOLOGY_AEROSOL_CM3",
        }
        water.altitude.attrs = {"units": "meters"}
        ds[water_species] = water

    ds["brdf"] = xr.Dataset({"lambertian_albedo": atmo.brdf.albedo})
    ds["brdf"].attrs = {
        "standard_name": "albedo",
        "long_name": "equivalent lambertian reflectance",
    }
    return ds


def create_atmosphere():
    atmosphere = sk.Atmosphere()
    atmosphere["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
    atmosphere["ozone"] = sk.Species(sk.O3OSIRISRes(), sk.Labow())
    atmosphere["aerosol"] = sk.SpeciesAerosolGloSSAC(extend=True)

    # Set the species to calculate the weighting function for
    # atmosphere.wf_species = 'ozone'

    return atmosphere


def aerosol_cross_section(wavelength, rg=0.08, sg=1.6):
    if isinstance(rg, float):
        alts = np.array([0.0, 100000.0])
        rg = np.ones_like(alts) * rg
        sg = np.ones_like(alts) * sg
        squeeze = True
    else:
        squeeze = False
        alts = np.linspace(0, 100000.0, len(rg))

    particle_size_values = {
        "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": rg,
        "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": sg,
    }
    particlesize_climatology = sk.ClimatologyUserDefined(alts, particle_size_values)
    optical_property = sk.MieAerosol(particlesize_climatology, "H2SO4")

    xsec = np.ones_like(alts)
    for aidx, alt in enumerate(alts):
        xsec[aidx] = optical_property.calculate_cross_sections(
            sk.MSIS90(),
            latitude=0.0,
            longitude=0.0,
            altitude=alt,
            mjd=53000.0,
            wavelengths=wavelength,
        ).total[0]
    if squeeze:
        xsec = xsec[0]
    return xsec * 1e5


def aerosol_phase_function(
    wavelength, rg=0.08, sg=1.6, angles: np.ndarray | None = None
):
    if angles is None:
        angles = np.arange(0, 180.1, 1.0)

    if isinstance(rg, float):
        alts = np.array([0.0, 100000.0])
        rg = np.ones_like(alts) * rg
        sg = np.ones_like(alts) * sg
        squeeze = True
    else:
        squeeze = False
        alts = np.linspace(0, 100000.0, len(rg))

    particle_size_values = {
        "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": rg,
        "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": sg,
    }
    particlesize_climatology = sk.ClimatologyUserDefined(alts, particle_size_values)
    optical_property = sk.MieAerosol(particlesize_climatology, "H2SO4")

    p = np.ones((len(alts), len(angles)))
    for aidx, alt in enumerate(alts):
        p[aidx, :] = optical_property.calculate_phase_matrix(
            sk.MSIS90(),
            latitude=0,
            longitude=10,
            altitude=alt,
            mjd=53000.0,
            wavelengths=[wavelength],
            cosine_scatter_angles=np.cos(angles * np.pi / 180),
        )[:, :, 0, 0].flatten()

    if squeeze:
        p = p[0]

    return p


def backscatter_to_extinction_ratio(wavelength, rg=0.08, sg=1.6):
    if isinstance(rg, float):
        rg = [rg]
        sg = [sg]

    p = np.ones_like(rg)
    for idx, (r, s) in enumerate(zip(rg, sg, strict=False)):
        p[idx] = aerosol_phase_function(
            wavelength, float(r), float(s), np.array([180.0])
        )[0]
    bs_to_ext = (1 / p) * np.pi * 4
    return bs_to_ext.flatten()


def aerosol_profile(
    lat,
    mjd,
    altitudes: np.ndarray | None = None,
    use_glossac=False,
    rg=0.08,
    sg=1.6,
):
    if altitudes is None:
        altitudes = np.linspace(500.0, 99500.0, 100)
    if use_glossac:
        particle_size_values = {
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": np.ones_like(altitudes) * rg,
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": np.ones_like(altitudes) * sg,
        }
        glossac = sk.SpeciesAerosolGloSSAC(
            extend=True, altitudes=altitudes, particle_size_values=particle_size_values
        )
        # glossac._climatology.decay_above = 1e-2
        ext = glossac.get_parameter(
            "SKCLIMATOLOGY_AEROSOL_EXTINCTIONPERKM",
            latitude=lat,
            longitude=0,
            mjd=mjd,
            altitudes=altitudes,
        )
        xsec = glossac.optical_property.calculate_cross_sections(
            sk.MSIS90(),
            latitude=lat,
            longitude=0.0,
            altitude=1000.0,
            mjd=mjd,
            wavelengths=750.0,
        ).total[0]
        aerosol = ext / xsec * 1e-5
    else:
        atmo = SimulationAtmosphere(Config.CALIPSO_OMPS_FILE)
        atmo.set_particle_size(rg, sg, altitudes)
        angle = atmo.angle_from_latitude(lat)
        lon = atmo.longitude(angle)
        mjd = atmo.mjd(angle)
        aerosol = atmo.aerosol.climatology.get_parameter(
            "SKCLIMATOLOGY_AEROSOL_CM3", lat, lon, altitudes, mjd
        )

    return aerosol


def cloud_profile(lat, mjd, altitudes: np.ndarray | None = None, cloud_scaling=1.0):
    if altitudes is None:
        altitudes = np.linspace(500.0, 99500.0, 100)
    atmo = SimulationAtmosphere(Config.CALIPSO_OMPS_FILE, cloud_scaling=cloud_scaling)
    angle = atmo.angle_from_latitude(lat)
    lon = atmo.longitude(angle)
    mjd = atmo.mjd(angle)
    profile = atmo.cloud.climatology.get_parameter("icecloud", lat, lon, altitudes, mjd)
    return profile * cloud_scaling


def cloud_cross_section(wavelength, reff=70.0):
    optical_property = sk.BaumIceCrystal(reff)
    xsec = optical_property.calculate_cross_sections(
        sk.MSIS90(),
        latitude=0.0,
        longitude=0.0,
        altitude=1000.0,
        mjd=53000.0,
        wavelengths=wavelength,
    ).total[0]
    return xsec * 1e5


def apriori_profile(
    lat: float, mjd: float, altitudes: np.ndarray | None = None, rg=0.08, sg=1.6
):
    if altitudes is None:
        altitudes = np.linspace(500.0, 99500.0, 100)

    date = pd.Timestamp("1858-11-17") + pd.Timedelta(mjd, "D")

    aerosol = []
    for year in np.arange(2000, 2015):
        day = (
            pd.Timestamp(f"{year}-{date.month:02}-01") - pd.Timestamp("1858-11-17")
        ) / pd.Timedelta(1, "D")
        aerosol.append(
            aerosol_profile(lat, day, altitudes, use_glossac=True, sg=sg, rg=rg)
        )
    aerosol = np.array(aerosol).mean(axis=0)
    aerosol[aerosol < 1e-10] = 0.0
    return aerosol * 0.5


def retrieval_atmo(
    lat: float,
    mjd: float,
    altitudes: np.ndarray | None = None,
    clouds: bool = True,
    cloud_top: float | None = None,
    cloud_optical_depth: float | None = None,
    cloud_effective_diameter: float | None = None,
    cloud_width: float | None = None,
):
    if altitudes is None:
        altitudes = np.linspace(500.0, 99500.0, 100)

    atmo = sk.Atmosphere()
    atmo["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
    atmo["ozone"] = sk.Species(sk.O3DBM(), sk.Labow())

    if clouds:
        if cloud_top is None:
            opt_prop = sk.BaumIceCrystal(70.0)
            clim = sk.ClimatologyUserDefined(
                altitudes, {"icecloud": cloud_profile(lat, mjd, altitudes) * 0.0}
            )
            atmo["icecloud"] = sk.Species(opt_prop, clim)
        else:
            baum_species = sk.SpeciesBaumIceCloud(
                particlesize_microns=cloud_effective_diameter,
                cloud_top_m=cloud_top,
                cloud_width_fwhm_m=cloud_width,
                vertical_optical_depth=cloud_optical_depth,
                vertical_optical_depth_wavel_nm=750.0,
            )

            # cloud_dens = baum_species.climatology.get_parameter('icecloud', latitude=0, longitude=0, mjd=54372,
            #                                                     altitudes=altitudes)
            atmo["icecloud"] = baum_species

    return atmo


def calipso_curtain_filename():
    return Config.CALIPSO_OMPS_FILE


def simulation_atmo(
    lat,
    mjd,
    altitudes: np.ndarray | None = None,
    rg=0.08,
    sg=1.6,
    opc_particle_size=False,
    clouds=True,
    use_glossac=False,
    two_dim=False,
    cloud_scaling=1.0,
    user_aerosol_profile: np.ndarray | None = None,
    user_cloud_profile: np.ndarray | None = None,
    clouds_as_sulphate=False,
):
    if altitudes is None:
        altitudes = np.linspace(500.0, 99500.0, 100)

    atmo = sk.Atmosphere()
    atmo["air"] = sk.Species(sk.Rayleigh(), sk.MSIS90())
    atmo["ozone"] = sk.Species(sk.O3DBM(), sk.Labow())

    if two_dim:
        # file = os.path.join(Config.CALIPSO_OMPS_FILE)
        file = Path(Config.CALIPSO_OMPS_FILE)
        sim_atmo = SimulationAtmosphere(file, cloud_scaling=cloud_scaling)
        sim_atmo.set_particle_size(rg, sg, altitudes)
        atmo["aerosol"] = sim_atmo.aerosol
    else:
        if user_aerosol_profile is None:
            user_aerosol_profile = aerosol_profile(
                lat, mjd, altitudes, use_glossac=use_glossac, rg=rg, sg=sg
            )
        atmo["aerosol"] = sk.SpeciesAerosol(
            altitudes=altitudes,
            aerosol_values={"aerosol": user_aerosol_profile},
            particle_size_values=particle_size(
                altitudes, rg=rg, sg=sg, opc_profile=opc_particle_size
            ),
        )

    if clouds:
        opt_prop = sk.BaumIceCrystal(70.0)

        if user_cloud_profile is None:
            user_cloud_profile = cloud_profile(
                lat, mjd, altitudes, cloud_scaling=cloud_scaling
            )

        if clouds_as_sulphate:
            ps = particle_size(altitudes)
            opt_prop_aer = sk.MieAerosol(
                sk.ClimatologyUserDefined(altitudes, ps), "H2SO4"
            )
            aer_xsec = aerosol_cross_section(1500.0)
            cloud_xsec = cloud_cross_section(1500.0, 70.0)
            user_cloud_profile *= cloud_xsec / aer_xsec
            clim = sk.ClimatologyUserDefined(
                altitudes, {"icecloud": user_cloud_profile}
            )
            atmo["icecloud"] = sk.Species(opt_prop_aer, clim)
        else:
            clim = sk.ClimatologyUserDefined(
                altitudes, {"icecloud": user_cloud_profile}
            )
            atmo["icecloud"] = sk.Species(opt_prop, clim)

    # atmo.wf_species = 'aerosol'
    return atmo


def particle_size(
    altitudes: np.ndarray | None = None,
    rg=0.08,
    sg=1.6,
    opc_profile: bool = False,
):
    if altitudes is None:
        altitudes = np.linspace(500.0, 99500.0, 100)

    if opc_profile:
        wpc = load_opc_profiles("WPC", bimodal=True, altitude_res=0.5).sel(
            time=slice("2000", "2021")
        )
        rg = wpc.median_radius_fine.mean(dim="time").to_numpy()
        sg = wpc.width_fine.mean(dim="time").to_numpy()
        good = rg > 0
        wpc_alts = wpc.altitude.to_numpy() * 1000.0
        rg = np.interp(
            altitudes, wpc_alts[good], rg[good], left=rg[good][0], right=rg[good][-1]
        )
        sg = np.interp(
            altitudes, wpc_alts[good], sg[good], left=sg[good][0], right=sg[good][-1]
        )
    return {
        "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": np.ones_like(altitudes) * rg,
        "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": np.ones_like(altitudes) * sg,
    }

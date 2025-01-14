from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import sasktran as sk
from ali.instrument.sensor_2channel import ALISensorDualChannel
from ali.instrument.simulator import ImagingSimulator
from ali.retrieval.aerosol import AerosolRetrieval
from ali.retrieval.measvec import MeasurementVector, MeasurementVectorElement
from ali.retrieval.measvec.transformer import (
    AltitudeNormalization,
    FrameRatio,
    FrameRatios,
    FrameSelect,
    LinearCombination,
    LinearCombinations,
    LogRadiance,
    RowAverage,
    SpectralRatio,
    SplineSmoothing,
    Truncate,
    VerticalDerivative,
    WavelengthSelect,
)
from ali.retrieval.statevector import StateVectorAerosolProfile, StateVectorCloudProfile
from ali.test.util.atmospheres import (
    aerosol_cross_section,
    aerosol_from_atmosphere,
    aerosol_profile,
    apriori_profile,
    atmosphere_to_xarray,
    cloud_cross_section,
    cloud_profile,
    ice_from_atmosphere,
    particle_size,
    retrieval_atmo,
    simulation_atmo,
)
from ali.test.util.rt_options import retrieval_rt_opts, simulation_rt_opts
from ali.util.config import Config
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.statevector import StateVector

plt.style.use(Config.MATPLOTLIB_STYLE_FILE)


def retrieve_cloud(
    optical_geometry,
    measurement_l1,
    lat,
    mjd,
    sensor_wavel,
    num_rows=640,
    num_cols=1,
    brdf=0.5,
    sim_atmo: sk.Atmosphere = None,
):

    np.random.seed(0)
    altitudes = np.arange(500.0, 45000.0, 200)
    atmo_ret = retrieval_atmo(lat, mjd, altitudes, clouds=False)
    atmo_ret.brdf = brdf

    l1_wavels = np.array([l1.data.wavelength.values for l1 in measurement_l1])
    unique_wavels = np.sort(np.unique(l1_wavels))
    short_wavel = unique_wavels[0]
    long_wavel = unique_wavels[-1]
    short_wavel_idx = np.argwhere(l1_wavels == short_wavel).flatten()
    long_wavel_idx = np.argwhere(l1_wavels == long_wavel).flatten()

    aerosol_mv = MeasurementVectorElement()
    aerosol_mv.add_transform(RowAverage(dim="nx"))
    aerosol_mv.add_transform(
        LinearCombination({short_wavel_idx[0]: 1, short_wavel_idx[0]: 1})
    )
    aerosol_mv.add_transform(
        AltitudeNormalization(norm_alts=(35000.0, 40000.0), couple_altitudes=True)
    )
    aerosol_mv.add_transform(LogRadiance())
    aerosol_mv.add_transform(Truncate(lower_bound=5500, upper_bound=35000))

    aerosol_mv2 = MeasurementVectorElement()
    aerosol_mv2.add_transform(RowAverage(dim="nx"))
    aerosol_mv2.add_transform(
        LinearCombination({long_wavel_idx[0]: 1, long_wavel_idx[1]: 1})
    )
    aerosol_mv2.add_transform(
        AltitudeNormalization(norm_alts=(35000.0, 40000.0), couple_altitudes=True)
    )
    aerosol_mv2.add_transform(LogRadiance())
    aerosol_mv2.add_transform(Truncate(lower_bound=5500, upper_bound=35000))

    cloud_mv1 = MeasurementVectorElement()
    cloud_mv1.add_transform(RowAverage(dim="nx"))
    cloud_mv1.add_transform(
        LinearCombinations(
            [
                {short_wavel_idx[0]: 1, short_wavel_idx[1]: -1},
                {short_wavel_idx[0]: 1, short_wavel_idx[1]: 1},
            ]
        )
    )
    cloud_mv1.add_transform(FrameRatio(index_1=0, index_2=1))
    cloud_mv1.add_transform(Truncate(lower_bound=5500, upper_bound=35000))

    cloud_mv = MeasurementVectorElement()
    cloud_mv.add_transform(RowAverage(dim="nx"))
    cloud_mv.add_transform(
        LinearCombinations(
            [
                {long_wavel_idx[0]: 1, long_wavel_idx[1]: -1},
                {long_wavel_idx[0]: 1, long_wavel_idx[1]: 1},
            ]
        )
    )
    # cloud_mv.add_transform(LinearCombinations([{0: 1, 2: -1}, {0: 1, 2: 1}]))
    # cloud_mv.add_transform(FrameRatios([(0, 1), (2, 3)]))
    cloud_mv.add_transform(FrameRatio(index_1=0, index_2=1))
    cloud_mv.add_transform(Truncate(lower_bound=5500, upper_bound=35000))

    # cloud_mv.add_transform(VerticalDerivative(smoothing=20))
    # cloud_mv.add_transform(SplineSmoothing(smoothing=300))
    cloud_vec = MeasurementVector([cloud_mv])

    meas_vec = MeasurementVector([aerosol_mv, aerosol_mv2, cloud_mv1, cloud_mv])

    ps_clim = sk.ClimatologyUserDefined(altitudes, particle_size(altitudes))
    aerosol_opt_prop = sk.MieAerosol(particlesize_climatology=ps_clim, species="H2SO4")
    # aerosol_apriori = aerosol_profile(lat, mjd, altitudes)
    # aerosol_apriori[aerosol_apriori == 0.0] = 1e-12

    aerosol_apriori = apriori_profile(lat, mjd, altitudes)
    aerosol_state_element = StateVectorAerosolProfile(
        altitudes_m=altitudes,
        values=np.log(aerosol_apriori),
        species_name="aerosol",
        optical_property=aerosol_opt_prop,
        lowerbound=5500,
        upperbound=35000,
        second_order_tikhonov_factor=np.array([30, 30, 50]),
        second_order_tikhonov_altitude=np.array([5000.0, 25000.0, 40000.0]),
    )
    aerosol_state_element.add_to_atmosphere(atmo_ret)

    cloud_opt_prop = sk.BaumIceCrystal(effective_size_microns=70)
    cloud_apriori = cloud_profile(lat, mjd, altitudes) * 100.0
    cloud_apriori[cloud_apriori < 1e-10] = 1e-10
    # cloud_profile[altitudes <= 22000] = 1e-8
    cloud_state_element = StateVectorCloudProfile(
        altitudes_m=altitudes,
        values=np.log(cloud_apriori),
        species_name="icecloud",
        optical_property=cloud_opt_prop,
        lowerbound=5500,
        upperbound=22000,
        second_order_tikhonov_factor=np.array([100, 100]),
        second_order_tikhonov_altitude=np.array([5000.0, 40000.0]),
    )
    cloud_state_element.add_to_atmosphere(atmo_ret)

    aerosol_ret = AerosolRetrieval(
        StateVector([aerosol_state_element, cloud_state_element]),
        meas_vec,
        cloud_vector=None,
        retrieval_altitudes=altitudes,
    )
    aerosol_ret.save_output = True
    aerosol_ret.rayleigh_norm = False

    cloud_mv.meas_dict(measurement_l1)
    ret_sensors = []
    opt_geom = []
    for wavel in sensor_wavel:
        if not hasattr(wavel, "__len__"):
            wavel = [wavel]
        ali = ALISensorDualChannel(
            wavelength_nm=wavel, num_rows=num_rows, num_columns=num_cols
        )
        ali.add_dark_current = False
        ali.add_noise = False
        ali.add_adc = False
        ali.turn_rotator_on()
        ret_sensors.append(ali)
        opt_geom.append(optical_geometry)

    ret_opts = retrieval_rt_opts(
        aerosol_ret,
        configure_for_cloud=True,
        cloud_lower_bound=0.0,
        cloud_upper_bound=18000.0,
    )
    forward_model = ImagingSimulator(
        sensors=ret_sensors,
        optical_axis=opt_geom,
        atmosphere=atmo_ret,
        options=ret_opts,
    )

    forward_model.store_radiance = False
    forward_model.grid_sampling = True
    forward_model.group_scans = False
    forward_model.dual_polarization = True

    rodgers = Rodgers(max_iter=10, lm_damping=0.01)
    aerosol_ret.configure_from_model(forward_model, measurement_l1)
    output = rodgers.retrieve(measurement_l1, forward_model, aerosol_ret)
    # aerosol_ret.cloud_detection(forward_model, measurement_l1)

    if type(measurement_l1) is list:
        tanalts = measurement_l1[0].tangent_locations().altitude / 1000
    else:
        tanalts = measurement_l1.tangent_locations().altitude / 1000

    sim_ds = atmosphere_to_xarray(sim_atmo)
    ret_ds = atmosphere_to_xarray(atmo_ret)
    save_dir = (
        r"C:\Users\lar555\PycharmProjects\ACCP\ali-retrieval\data\curtain_retrievals"
    )
    # save_state(aerosol_ret, output, measurement_l1, lat, save_dir)

    sim_ds = atmosphere_to_xarray(sim_atmo)
    ret_ds = atmosphere_to_xarray(atmo_ret, ice_species="icecloud")
    plot_retrieval(sim_ds, ret_ds)
    fig, ax = plot_extinction_results(aerosol_ret, output, tanalts, lat=lat)
    fig.savefig(os.path.join(save_dir, "2d_sim_1d_ret", f"lat_{lat}.png"), dpi=450)
    # fig, ax = plot_averaging_kernels(aerosol_ret, output, tanalts)
    # fig, ax = plot_cloud_results(measurement_l1)

    modelled_l1 = forward_model.calculate_radiance()

    yc_meas = cloud_vec.meas_dict(measurement_l1)
    yc_mod = cloud_vec.meas_dict(modelled_l1)

    fig, ax = plt.subplots(1, 1, figsize=(3, 4), dpi=200)
    ax.plot(yc_meas["y"], tanalts)
    ax.plot(yc_mod["y"], tanalts)
    ax.set_ylabel("Altitude [km]")
    ax.set_xlabel(
        "$\\left(\\frac{Q}{I}\\right)_{750} / \\left(\\frac{Q}{I}\\right)_{1550}$"
    )

    diff = np.abs(yc_meas["y"] - yc_mod["y"]) - np.sqrt(yc_meas["y_error"]) * 2
    # ax.plot(np.abs(yc_meas['y'] - yc_mod['y']) - np.sqrt(yc_meas['y_error'])*2, tanalts[:, 0])
    cloud = diff > 0
    cloud = np.convolve(cloud, np.array([1, 1, 1]), mode="same")
    cloud_altitude = np.max(tanalts[cloud == 3, 0])


def plot_retrieval(sim_atmo, ret_atmo, wavel=750.0):

    true_color = "#c2452f"
    ret_color = "#2177b5"
    fig, ax = plt.subplots(1, 2, figsize=(4, 3), dpi=200, sharey=True)
    fig.subplots_adjust(
        left=0.1, bottom=0.12, right=0.97, top=0.95, wspace=0.05, hspace=0.05
    )

    name = "aerosol"
    ax[0].plot(
        sim_atmo[name].extinction.sel(wavelength=wavel),
        sim_atmo[name].altitude / 1000,
        color=true_color,
    )
    ax[0].plot(
        ret_atmo[name].extinction.sel(wavelength=wavel),
        ret_atmo[name].altitude / 1000,
        color=ret_color,
    )

    name = "icecloud"
    ax[1].plot(
        sim_atmo[name].extinction.sel(wavelength=wavel),
        sim_atmo[name].altitude / 1000,
        color=true_color,
    )
    ax[1].plot(
        ret_atmo[name].extinction.sel(wavelength=wavel),
        ret_atmo[name].altitude / 1000,
        color=ret_color,
    )


def plot_extinction_results(
    aerosol_ret,
    output,
    tanalts,
    clouds=True,
    lat=33.0,
    mjd=54372,
    plot_error=True,
    nvec=1,
):
    aer_xsec = aerosol_cross_section(750.0)

    aer_state = aerosol_ret._state_vector.state_elements[0]
    cloud_state = aerosol_ret._state_vector.state_elements[1]
    # altitudes = aerosol_ret._retrieval_altitudes
    aer_altitude = aer_state._altitudes_m[aer_state.retrieval_alts()]
    cloud_altitude = cloud_state._altitudes_m[cloud_state.retrieval_alts()]
    nalts = len(aer_altitude)

    aerosol_final = np.exp(aer_state.state()) * aer_xsec
    aerosol_apriori = apriori_profile(lat, mjd, aer_altitude) * aer_xsec
    aerosol_true = aerosol_profile(lat, mjd, aer_altitude) * aer_xsec

    cloud_xsec = cloud_cross_section(750.0)
    cloud_apriori = cloud_profile(lat, mjd, cloud_altitude) * cloud_xsec
    cloud_final = np.exp(cloud_state.state()) * cloud_xsec
    cloud_true = cloud_profile(lat, mjd, cloud_altitude) * cloud_xsec

    true_color = "#c2452f"
    ret_color = "#2177b5"
    fig, ax = plt.subplots(1, 2, figsize=(4, 3), dpi=200, sharey=True)
    fig.subplots_adjust(
        left=0.1, bottom=0.12, right=0.97, top=0.95, wspace=0.05, hspace=0.05
    )

    ax[0].set_ylabel("Altitude [km]")
    ax[0].set_xlabel("Aerosol [$\\times10^{-3}$km$^{-1}$]")
    ax[1].set_xlabel("Cloud [$\\times10^{-3}$km$^{-1}$]")
    scale = 1000
    (l0,) = ax[0].plot(
        aerosol_apriori * scale,
        aer_altitude / 1000,
        color=ret_color,
        ls="--",
        lw=1,
        zorder=10,
    )
    (l1,) = ax[0].plot(
        aerosol_true * scale, aer_altitude / 1000, color=true_color, lw=1, zorder=11
    )
    for i in range(len(output["xs"])):
        (l2,) = ax[0].plot(
            np.exp(output["xs"][i][0:nalts]) * aer_xsec * scale,
            aer_altitude / 1000,
            color=ret_color,
            alpha=0.2,
            lw=0.5,
            zorder=9,
        )
    (l3,) = ax[0].plot(
        aerosol_final * scale, aer_altitude / 1000, color=ret_color, zorder=15
    )

    (l0,) = ax[1].plot(
        cloud_apriori * scale,
        cloud_altitude / 1000,
        color=ret_color,
        ls="--",
        lw=1,
        zorder=10,
    )
    (l1,) = ax[1].plot(
        cloud_true * scale, cloud_altitude / 1000, color=true_color, lw=1, zorder=11
    )
    for i in range(len(output["xs"])):
        (l2,) = ax[1].plot(
            np.exp(output["xs"][i][nalts:]) * cloud_xsec * scale,
            cloud_altitude / 1000,
            color=ret_color,
            alpha=0.2,
            lw=0.5,
            zorder=9,
        )
    (l3,) = ax[1].plot(
        cloud_final * scale, cloud_altitude / 1000, color=ret_color, zorder=15
    )
    # ax[0].set_xlim(0, .0028 * scale)

    # if plot_error:
    #     error = np.sqrt(np.diag(output['solution_covariance'])) * aer_xsec
    #     l4 = ax[0].fill_betweenx(altitudes / 1000, (aerosol_final + error) * scale, (aerosol_final - error) * scale,
    #                              color=ret_color, lw=0, alpha=0.3)
    # leg = ax[0].legend([l0, l1, l2, l3], ['A priori', 'True', 'Iterations', 'Retrieved'],
    #                    framealpha=1, facecolor=ax[0].get_facecolor(), edgecolor='none', fontsize='small')
    # leg.set_title('Retrieval State', prop={'size': 'small', 'weight': 'bold'})
    # ax[0].set_ylim(0, 40)

    n = len(tanalts)
    nvec = 4

    fig, ax = plt.subplots(1, nvec, figsize=(4, 3), dpi=200, sharey=True)
    fig.subplots_adjust(
        left=0.1, bottom=0.12, right=0.97, top=0.95, wspace=0.05, hspace=0.05
    )

    for i in range(nvec):
        ax[i].plot(
            output["y_meas"][(i * n) : ((i + 1) * n)], tanalts, color=true_color, lw=1
        )
    for i in range(len(output["ys"])):
        for j in range(nvec):
            ax[j].plot(
                output["ys"][i][(j * n) : ((j + 1) * n)],
                tanalts,
                color=ret_color,
                alpha=0.5,
                lw=0.5,
            )
    for j in range(nvec):
        ax[j].plot(
            output["ys"][-1][(j * n) : ((j + 1) * n)], tanalts, color=ret_color, lw=1
        )

    ax[0].set_xlim(0, 0.0028 * scale)
    # ax[1].set_xlim(0, 1.2)
    for a in ax:
        a.set_xlabel("Measurement Vector")

    ax[0].set_ylabel("Altitude [km]")
    ax[0].set_title("750 nm")
    ax[1].set_title("1550 nm")
    ax[2].set_title("Spectral Polarization")

    return fig, ax

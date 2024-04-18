from __future__ import annotations

from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr
from ali.retrieval.aerosol import AerosolRetrieval, ParticleSizeRetrieval
from ali.retrieval.extinction import ExtinctionRetrieval
from ali.retrieval.measvec import MeasurementVector, MeasurementVectorElement
from ali.retrieval.statevector import (
    StateVectorAerosolProfile,
    StateVectorProfileEffectiveRadius,
    StateVectorProfileParticleSize,
)
from ali.test.util.atmospheres import (
    aerosol_cross_section,
    apriori_profile,
    backscatter_to_extinction_ratio,
    particle_size,
    retrieval_atmo,
)
from ali.util.analysis import (
    decode_to_multiindex,
    encode_multiindex,
    resolution_from_averaging_kernel,
)
from ali.util.config import Config
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.statevector import StateVector

plt.style.use(Config.MATPLOTLIB_STYLE_FILE)


class LognormalRadiusRetrieval(ExtinctionRetrieval):
    """
    Retrieves the lognormal number density and median radius assuming a fixed distribution width.
    """

    def __init__(self, *args):

        super().__init__(*args)
        self._aerosol_ret = None
        self._particle_size_vector_wavelength: list[float] = [750.0, 1250.0]
        self.max_aerosol_iterations = 2
        self.tikhonov_factor = np.array([20, 50, 50]) * 2
        self.tikhonov_altitude = np.array([0000.0, 25000.0, 30000.0])
        self.particle_size_tikhonov_factor: np.ndarray = np.array([150, 300, 300]) * 2
        self.particle_size_tikhonov_altitude: np.ndarray = np.array(
            [5000.0, 25000.0, 30000.0]
        )
        self.aerosol_lm_damping = 0.01
        self.lm_damping = 0.01

    @property
    def particle_size_vector_wavelength(self) -> list[float]:
        return self._particle_size_vector_wavelength

    @particle_size_vector_wavelength.setter
    def particle_size_vector_wavelength(self, value: Union[float, list[float]]):

        if not hasattr(value, "__len__"):
            value = [value]
        self._particle_size_vector_wavelength = value

    @property
    def particle_size_measurement_vector(self) -> MeasurementVector:
        """
        Measurement vector used for the particle size retrieval.
        """
        return self._measurement_vector(self.particle_size_vector_wavelength)

    def median_radius_state_element(self):

        if self._aerosol_opt_prop is None:
            ps_clim = sk.ClimatologyUserDefined(
                self.altitudes, particle_size(self.altitudes)
            )
            self._aerosol_opt_prop = sk.MieAerosol(
                particlesize_climatology=ps_clim, species="H2SO4"
            )

        element = StateVectorProfileParticleSize(
            altitudes_m=self.altitudes,
            values=np.log(self.apriori_profile() * 0 + 0.08),
            species_name="aerosol",
            optical_property=self._aerosol_opt_prop,
            size_type="lognormal_medianradius",
            lowerbound=self.lower_bound,
            upperbound=self.upper_bound,
            second_order_tikhonov_factor=self.particle_size_tikhonov_factor,
            second_order_tikhonov_altitude=self.particle_size_tikhonov_altitude,
        )
        return element

    def retrieve(self):

        self.cloud_top = self.find_cloud_top_altitude()
        if self.use_cloud_for_lower_bound and (self.cloud_top > self.lower_bound):
            self.lower_bound = self.cloud_top + self.altitude_resolution

        self.retrieval_atmosphere = self.generate_retrieval_atmosphere()

        aerosol_element = self.aerosol_state_element()
        radius_element = self.median_radius_state_element()

        aerosol_element.add_to_atmosphere(self.retrieval_atmosphere)
        self._aerosol_ret = AerosolRetrieval(
            state_vector=StateVector([aerosol_element]),
            measurement_vector=self.aerosol_measurement_vector,
            retrieval_altitudes=self.altitudes,
        )
        self._aerosol_ret.save_output = True
        self._aerosol_ret.rayleigh_norm = False

        self._forward_model = self.forward_model(self._aerosol_ret)
        self._jacobian_altitudes = self._aerosol_ret.jacobian_altitudes
        if self.sun is not None:
            self._forward_model.sun = self.sun

        rodgers = Rodgers(
            max_iter=self.max_aerosol_iterations, lm_damping=self.aerosol_lm_damping
        )
        # self._aerosol_ret.configure_from_model(self._forward_model, self.measurement_l1)
        aer_output = rodgers.retrieve(
            self.measurement_l1, self._forward_model, self._aerosol_ret
        )
        aer_output["cloud_top_altitude"] = self.cloud_top

        self._retrieval = ParticleSizeRetrieval(
            state_vector=StateVector([aerosol_element, radius_element]),
            measurement_vector=self.particle_size_measurement_vector,
            retrieval_altitudes=self.altitudes,
        )
        radius_element.add_to_atmosphere(self.retrieval_atmosphere)
        rodgers = Rodgers(max_iter=self.max_iterations, lm_damping=self.lm_damping)
        self._retrieval.configure_from_model(self._forward_model, self.measurement_l1)
        output = rodgers.retrieve(
            self.measurement_l1, self._forward_model, self._retrieval
        )
        output["cloud_top_altitude"] = self.cloud_top

        if self.output_filename:
            self.save_output(
                rodgers,
                output,
                self.retrieval_atmosphere,
                self._retrieval,
                self.particle_size_measurement_vector,
            )

    @staticmethod
    def plot_results(
        output_filename,
        extinction_wavelength: float = 750.0,
        log_state: bool = True,
        plot_error: bool = True,
        plot_meas_vec: bool = True,
        plot_averaging_kernel: bool = True,
        plot_effective_radius: bool = True,
        aerosol_scale: Union[int, float] = 1000,
        plot_cloud: bool = True,
        kernel_kwargs: dict = {},
        figsize=(5, 6),
    ):
        try:
            true_state = xr.open_dataset(output_filename, group="true/aerosol")
            true_state["effective_radius"] = (
                true_state.lognormal_median_radius
                * np.exp(np.log(true_state.lognormal_width) ** 2 * 5 / 2)
            )
        except (KeyError, OSError) as e:
            true_state = None

        ret_data = xr.open_dataset(output_filename, group="retrieved/aerosol")
        ret_data = xr.merge(
            [ret_data, xr.open_dataset(output_filename, group="retrieved/cloud")]
        )

        ret_info = xr.open_dataset(output_filename, group="retrieval/state")
        ret_info = decode_to_multiindex(
            ret_info, ["ret_idx", "pert_idx", "measurement_idx"]
        )
        # ret_info = decode_to_multiindex(ret_info, 'pert_idx')
        # ret_info = decode_to_multiindex(ret_info, 'measurement_idx')

        aerosol_iterations = ret_info.target_state.sel(ret_state="aerosol")
        aerosol_initial = np.exp(ret_info.initial_state.sel(ret_state="aerosol"))
        radius_iterations = ret_info.target_state.sel(
            ret_state="aerosol_lognormal_medianradius"
        )
        radius_initial = np.exp(
            ret_info.initial_state.sel(ret_state="aerosol_lognormal_medianradius")
        )
        aerosol_iterations = np.exp(aerosol_iterations)
        radius_iterations = np.exp(radius_iterations)
        effrad_iterations = radius_iterations * np.exp(np.log(1.6) ** 2 * 5 / 2)
        effrad_initial = radius_initial * np.exp(np.log(1.6) ** 2 * 5 / 2)

        xsecs = []
        for iter in aerosol_iterations.iteration.values:
            rg = radius_iterations.sel(iteration=iter).values
            xsec = aerosol_cross_section(
                extinction_wavelength, rg=rg, sg=rg * 0.0 + 1.6
            )
            xsec = xr.DataArray(
                xsec, dims=["ret_alt"], coords=[radius_iterations.ret_alt.values]
            )
            xsec["iteration"] = iter
            xsecs.append(xsec)
        xsec_initial = aerosol_cross_section(
            extinction_wavelength,
            rg=radius_initial.values,
            sg=radius_initial * 0.0 + 1.6,
        )
        xsecs = xr.concat(xsecs, dim="iteration")
        aerosol_iterations *= xsecs.interp(ret_alt=radius_iterations.ret_alt.values)
        aerosol_initial *= xsec_initial
        aerosol_final = ret_data.extinction.sel(wavelength=extinction_wavelength)

        radius_final = ret_data.lognormal_median_radius
        effective_radius_final = radius_final * np.exp(np.log(1.6) ** 2 * 5 / 2)

        true_color = "#c2452f"
        ret_color = "#2177b5"
        num_plots = 2
        x = 4
        if plot_meas_vec:
            num_plots += 1
            x += 1.7
        if plot_averaging_kernel:
            num_plots += 2
            x += 1.7 * 2

        if plot_averaging_kernel:
            fig, ax = plt.subplots(
                1, num_plots - 2, figsize=figsize, dpi=200, sharey=True
            )
            ax = list(ax)
            hspace = 0.05 / 3
            fig.subplots_adjust(
                left=0.08, bottom=0.57, right=0.97, top=0.98, wspace=0.05
            )
            pos0 = ax[0].get_position()
            pos1 = ax[-1].get_position()

            ax.append(
                fig.add_axes(
                    [pos0.x0, 0.06, (pos1.x1 - pos0.x0) / 2 - hspace / 2, pos0.height]
                )
            )
            pos2 = ax[-1].get_position()
            ax.append(
                fig.add_axes(
                    [
                        pos2.x1 + hspace,
                        0.06,
                        (pos1.x1 - pos0.x0) / 2 - hspace / 2,
                        pos0.height,
                    ]
                )
            )
            ax[-2].set_ylabel("Altitude [km]")
            ax[-1].set_yticklabels([])
        else:
            fig, ax = plt.subplots(1, num_plots, figsize=figsize, dpi=200, sharey=True)
            fig.subplots_adjust(
                left=0.1, bottom=0.12, right=0.97, top=0.95, wspace=0.05
            )

        ax[0].set_ylabel("Altitude [km]")
        ax[0].set_xlabel("Extinction [$\\times10^{-3}$km$^{-1}$]")
        if plot_effective_radius:
            ax[1].set_xlabel(r"Effective Radius [$\mu$m]")
        else:
            ax[1].set_xlabel(r"Median Radius [$\mu$m]")
        if plot_meas_vec:
            ax[2].set_xlabel("Measurement Vectors")

        (l0,) = ax[0].plot(
            aerosol_initial * aerosol_scale,
            aerosol_initial.ret_alt / 1000,
            color=ret_color,
            ls="--",
            lw=1,
            zorder=10,
        )

        if plot_effective_radius:
            (l0,) = ax[1].plot(
                effrad_initial,
                effrad_initial.ret_alt / 1000,
                color=ret_color,
                ls="--",
                lw=1,
                zorder=10,
            )
        else:
            (l0,) = ax[1].plot(
                radius_initial,
                radius_initial.ret_alt / 1000,
                color=ret_color,
                ls="--",
                lw=1,
                zorder=10,
            )

        if true_state is not None:
            (l1,) = ax[0].plot(
                true_state.extinction.sel(wavelength=extinction_wavelength)
                * aerosol_scale,
                true_state.altitude / 1000,
                color=true_color,
                lw=1,
                zorder=11,
            )
            if plot_effective_radius:
                (l1,) = ax[1].plot(
                    true_state.effective_radius,
                    true_state.altitude / 1000,
                    color=true_color,
                    lw=1,
                    zorder=11,
                )
            else:
                (l1,) = ax[1].plot(
                    true_state.lognormal_median_radius,
                    true_state.altitude / 1000,
                    color=true_color,
                    lw=1,
                    zorder=11,
                )

        for i in range(1, len(ret_info.iteration)):
            (l2,) = ax[0].plot(
                aerosol_iterations.sel(iteration=i).values * aerosol_scale,
                aerosol_iterations.ret_alt / 1000,
                color=ret_color,
                alpha=0.2,
                lw=0.5,
                zorder=9,
            )
        (l3,) = ax[0].plot(
            aerosol_final.values * aerosol_scale,
            aerosol_final.altitude.values / 1000,
            color=ret_color,
            zorder=15,
        )

        if plot_effective_radius:
            for i in range(1, len(ret_info.iteration)):
                (l2,) = ax[1].plot(
                    effrad_iterations.sel(iteration=i).values,
                    radius_iterations.ret_alt / 1000,
                    color=ret_color,
                    alpha=0.2,
                    lw=0.5,
                    zorder=9,
                )
            (l3,) = ax[1].plot(
                effective_radius_final.values,
                radius_final.altitude.values / 1000,
                color=ret_color,
                zorder=15,
            )
        else:
            for i in range(1, len(ret_info.iteration)):
                (l2,) = ax[1].plot(
                    radius_iterations.sel(iteration=i).values,
                    radius_iterations.ret_alt / 1000,
                    color=ret_color,
                    alpha=0.2,
                    lw=0.5,
                    zorder=9,
                )
            (l3,) = ax[1].plot(
                radius_final.values,
                radius_final.altitude.values / 1000,
                color=ret_color,
                zorder=15,
            )

        # TODO: implement error plotting - need to properly select diagonal of each state element.
        if plot_error:
            coords = [
                ret_info.solution_covariance.sel(ret_state="aerosol").ret_alt.values
            ]
            Se = np.diag(
                ret_info.solution_covariance.sel(
                    ret_state="aerosol", pert_state="aerosol"
                ).values
            )
            Se = xr.DataArray(Se, dims=["ret_alt"], coords=coords)
            error = (
                np.sqrt(Se)
                * np.exp(
                    ret_info.sel(ret_state="aerosol").isel(iteration=-1).target_state
                )
            ) * xsecs.isel(iteration=-1).interp(ret_alt=Se.ret_alt.values)
            final = aerosol_final.interp(altitude=Se.ret_alt.values)
            l4 = ax[0].fill_betweenx(
                Se.ret_alt.values / 1000,
                (final.values + error.values) * aerosol_scale,
                (final.values - error.values) * aerosol_scale,
                color=ret_color,
                lw=0,
                alpha=0.3,
            )

            Se = np.diag(
                ret_info.solution_covariance.sel(
                    ret_state="aerosol_lognormal_medianradius",
                    pert_state="aerosol_lognormal_medianradius",
                ).values
            )
            Se = xr.DataArray(Se, dims=["ret_alt"], coords=coords)
            error = np.sqrt(Se) * radius_iterations.isel(iteration=-1)
            final = radius_final.interp(altitude=Se.ret_alt.values)
            if plot_effective_radius:
                error *= np.exp(np.log(1.6) ** 2 * 5 / 2)
                final *= np.exp(np.log(1.6) ** 2 * 5 / 2)
            l4 = ax[1].fill_betweenx(
                Se.ret_alt.values / 1000,
                (final.values + error.values),
                (final.values - error.values),
                color=ret_color,
                lw=0,
                alpha=0.3,
            )

        if true_state:
            leg = ax[0].legend(
                [l0, l2, l3, l1],
                ["A priori", "Iterations", "Retrieved", "True"],
                framealpha=1,
                facecolor=ax[0].get_facecolor(),
                edgecolor="none",
                fontsize="small",
            )
        else:
            leg = ax[0].legend(
                [l0, l2, l3],
                ["A priori", "Iterations", "Retrieved"],
                framealpha=1,
                facecolor=ax[0].get_facecolor(),
                edgecolor="none",
                fontsize="small",
            )

        leg.set_title("Retrieval State", prop={"size": "small", "weight": "bold"})

        ax[0].axhline(
            ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444"
        )
        ax[1].axhline(
            ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444"
        )
        ax[0].set_ylim(0, 40)

        if plot_meas_vec:
            ExtinctionRetrieval.plot_measurement_vectors(output_filename, ax=ax[2])

        if plot_averaging_kernel:
            ExtinctionRetrieval.plot_averaging_kernel(
                output_filename, "aerosol", ax=ax[-2], **kernel_kwargs
            )
            ExtinctionRetrieval.plot_averaging_kernel(
                output_filename,
                "aerosol_lognormal_medianradius",
                ax=ax[-1],
                **kernel_kwargs,
            )
            ax[-2].set_xlabel("Aerosol Averaging Kernels")
            ax[-1].set_xlabel("Radius Averaging Kernels")

        return fig, ax

    @staticmethod
    def plot_error(
        output_filename,
        extinction_wavelength: float = 750.0,
        log_state: bool = True,
        plot_error: bool = True,
        plot_meas_vec: bool = True,
        plot_averaging_kernel: bool = True,
        plot_effective_radius: bool = True,
        plot_backscatter: bool = False,
        aerosol_scale: Union[int, float] = 1000,
        plot_cloud: bool = True,
        kernel_kwargs: dict = {},
        figsize=(5, 6),
    ):

        try:
            true_state = xr.open_dataset(output_filename, group="true/aerosol")
            true_state["effective_radius"] = (
                true_state.lognormal_median_radius
                * np.exp(np.log(true_state.lognormal_width) ** 2 * 5 / 2)
            )
        except (KeyError, OSError) as e:
            true_state = None

        ret_data = xr.open_dataset(output_filename, group="retrieved/aerosol")
        ret_data = xr.merge(
            [ret_data, xr.open_dataset(output_filename, group="retrieved/cloud")]
        )

        ret_info = xr.open_dataset(output_filename, group="retrieval/state")
        ret_info = decode_to_multiindex(
            ret_info, ["ret_idx", "pert_idx", "measurement_idx"]
        )
        # ret_info = decode_to_multiindex(ret_info, 'pert_idx')
        # ret_info = decode_to_multiindex(ret_info, 'measurement_idx')

        aerosol_iterations = ret_info.target_state.sel(ret_state="aerosol")
        radius_iterations = ret_info.target_state.sel(
            ret_state="aerosol_lognormal_medianradius"
        )
        aerosol_iterations = np.exp(aerosol_iterations)
        radius_iterations = np.exp(radius_iterations)
        effrad_iterations = radius_iterations * np.exp(np.log(1.6) ** 2 * 5 / 2)

        xsecs = []
        for iter in aerosol_iterations.iteration.values:
            rg = radius_iterations.sel(iteration=iter).values
            xsec = aerosol_cross_section(
                extinction_wavelength, rg=rg, sg=rg * 0.0 + 1.6
            )
            xsec = xr.DataArray(
                xsec, dims=["ret_alt"], coords=[radius_iterations.ret_alt.values]
            )
            xsec["iteration"] = iter
            xsecs.append(xsec)
        xsecs = xr.concat(xsecs, dim="iteration")
        aerosol_iterations *= xsecs.interp(ret_alt=radius_iterations.ret_alt.values)
        aerosol_final = ret_data.extinction.sel(wavelength=extinction_wavelength)

        radius_final = ret_data.lognormal_median_radius
        effective_radius_final = radius_final * np.exp(np.log(1.6) ** 2 * 5 / 2)

        true_color = "#c2452f"
        ret_color = "#2177b5"

        if plot_backscatter:
            fig, axs = plt.subplots(3, 3, figsize=figsize, dpi=200, sharey=True)
            fig.subplots_adjust(
                left=0.1, bottom=0.08, right=0.97, top=0.95, wspace=0.05, hspace=0.25
            )
        else:
            fig, axs = plt.subplots(2, 3, figsize=figsize, dpi=200, sharey=True)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.95, wspace=0.05)

        ax = axs[:, 0]
        ax2 = axs[:, 1]
        ax3 = axs[:, 2]

        ax[0].set_ylabel("Altitude [km]")
        ax[1].set_ylabel("Altitude [km]")
        ax[0].set_xlabel("Extinction [$\\times10^{-3} $km$^{-1}$]")
        ax2[0].set_xlabel("Extinction Error [$\\times10^{-3} $km$^{-1}$]")
        ax3[0].set_xlabel("Extinction Error [%]")
        if plot_effective_radius:
            ax[1].set_xlabel(r"Effective Radius [$\mu$m]")
            ax2[1].set_xlabel(r"Effective Radius Error [$\mu$m]")
            ax3[1].set_xlabel("Effective Radius Error [%]")
        else:
            ax[1].set_xlabel(r"Median Radius [$\mu$m]")
        if plot_backscatter:
            ax[2].set_xlabel("Extinction to\nBackscatter [sr]")
            ax2[2].set_xlabel("Extinction to\nBackscatter Error [sr]")
            ax3[2].set_xlabel("Extinction to\nBackscatter Error [%]")
            ax[2].set_ylabel("Altitude [km]")

        (l0,) = ax[0].plot(
            aerosol_iterations.sel(iteration=0) * aerosol_scale,
            aerosol_iterations.ret_alt / 1000,
            color=ret_color,
            ls="--",
            lw=1,
            zorder=10,
        )

        if plot_effective_radius:
            (l0,) = ax[1].plot(
                effrad_iterations.sel(iteration=0),
                aerosol_iterations.ret_alt / 1000,
                color=ret_color,
                ls="--",
                lw=1,
                zorder=10,
            )
        else:
            (l0,) = ax[1].plot(
                radius_iterations.sel(iteration=0),
                aerosol_iterations.ret_alt / 1000,
                color=ret_color,
                ls="--",
                lw=1,
                zorder=10,
            )

        if plot_backscatter:
            bs_to_ext = xr.ones_like(radius_iterations)
            bs = []
            for iter in radius_iterations.iteration:
                bs.append(
                    backscatter_to_extinction_ratio(
                        532.0,
                        radius_iterations.sel(iteration=iter).values,
                        sg=radius_iterations.sel(iteration=iter).values * 0.0 + 1.6,
                    )
                )
            bs_to_ext[:, :] = np.array(bs)
            (l0,) = ax[2].plot(
                bs_to_ext.sel(iteration=0),
                bs_to_ext.ret_alt / 1000,
                color=ret_color,
                ls="--",
                lw=1,
                zorder=10,
            )

        if true_state is not None:
            (l1,) = ax[0].plot(
                true_state.extinction.sel(wavelength=extinction_wavelength)
                * aerosol_scale,
                true_state.altitude / 1000,
                color=true_color,
                lw=1,
                zorder=11,
            )

            true_ext = true_state.extinction.sel(
                wavelength=extinction_wavelength
            ).interp(altitude=aerosol_final.altitude.values)
            abs_error = (true_ext - aerosol_final) * aerosol_scale
            error = (true_ext - aerosol_final) / true_ext * 100
            ax2[0].plot(
                abs_error, error.altitude / 1000, color=true_color, lw=1, zorder=11
            )
            ax3[0].plot(error, error.altitude / 1000, color=true_color, lw=1, zorder=11)

            if plot_effective_radius:
                (l1,) = ax[1].plot(
                    true_state.effective_radius,
                    true_state.altitude / 1000,
                    color=true_color,
                    lw=1,
                    zorder=11,
                )

                true_reff = true_state.effective_radius.interp(
                    altitude=effrad_iterations.ret_alt
                )
                abs_error = true_reff - effrad_iterations.isel(iteration=-1)
                error = (
                    (true_reff - effrad_iterations.isel(iteration=-1)) / true_reff * 100
                )
                ax2[1].plot(
                    abs_error, error.altitude / 1000, color=true_color, lw=1, zorder=11
                )
                ax3[1].plot(
                    error, error.altitude / 1000, color=true_color, lw=1, zorder=11
                )
            else:
                (l1,) = ax[1].plot(
                    true_state.lognormal_median_radius,
                    true_state.altitude / 1000,
                    color=true_color,
                    lw=1,
                    zorder=11,
                )

            if plot_backscatter:
                true_med_rad = true_state.lognormal_median_radius.interp(
                    altitude=bs_to_ext.ret_alt
                )
                true_bs = xr.DataArray(
                    backscatter_to_extinction_ratio(
                        532.0, true_med_rad.values, true_med_rad.values * 0.0 + 1.6
                    ),
                    dims=["ret_alt"],
                    coords=[bs_to_ext.ret_alt.values],
                )

                (l1,) = ax[2].plot(
                    true_bs, true_bs.ret_alt / 1000, color=true_color, lw=1, zorder=11
                )

                abs_error = true_bs - bs_to_ext.isel(iteration=-1)
                error = (true_bs - bs_to_ext.isel(iteration=-1)) / true_bs * 100
                ax2[2].plot(
                    abs_error, error.ret_alt / 1000, color=true_color, lw=1, zorder=11
                )
                ax3[2].plot(
                    error, error.ret_alt / 1000, color=true_color, lw=1, zorder=11
                )

        for i in range(1, len(ret_info.iteration)):
            (l2,) = ax[0].plot(
                aerosol_iterations.sel(iteration=i).values * aerosol_scale,
                aerosol_iterations.ret_alt / 1000,
                color=ret_color,
                alpha=0.2,
                lw=0.5,
                zorder=9,
            )
        (l3,) = ax[0].plot(
            aerosol_final.values * aerosol_scale,
            aerosol_final.altitude.values / 1000,
            color=ret_color,
            zorder=15,
        )

        if plot_effective_radius:
            for i in range(1, len(ret_info.iteration)):
                (l2,) = ax[1].plot(
                    effrad_iterations.sel(iteration=i).values,
                    radius_iterations.ret_alt / 1000,
                    color=ret_color,
                    alpha=0.2,
                    lw=0.5,
                    zorder=9,
                )
            (l3,) = ax[1].plot(
                effective_radius_final.values,
                radius_final.altitude.values / 1000,
                color=ret_color,
                zorder=15,
            )
        else:
            for i in range(1, len(ret_info.iteration)):
                (l2,) = ax[1].plot(
                    radius_iterations.sel(iteration=i).values,
                    radius_iterations.ret_alt / 1000,
                    color=ret_color,
                    alpha=0.2,
                    lw=0.5,
                    zorder=9,
                )
            (l3,) = ax[1].plot(
                radius_final.values,
                radius_final.altitude.values / 1000,
                color=ret_color,
                zorder=15,
            )

        if plot_backscatter:
            for i in range(1, len(ret_info.iteration)):
                (l2,) = ax[2].plot(
                    bs_to_ext.sel(iteration=i).values,
                    bs_to_ext.ret_alt / 1000,
                    color=ret_color,
                    alpha=0.2,
                    lw=0.5,
                    zorder=9,
                )
            (l3,) = ax[2].plot(
                bs_to_ext.isel(iteration=-1).values,
                bs_to_ext.ret_alt.values / 1000,
                color=ret_color,
                zorder=15,
            )
            ax3[2].set_xlim(-50, 50)

        # TODO: implement error plotting - need to properly select diagonal of each state element.
        if plot_error:
            coords = [
                ret_info.solution_covariance.sel(ret_state="aerosol").ret_alt.values
            ]
            Se = np.diag(
                ret_info.solution_covariance.sel(
                    ret_state="aerosol", pert_state="aerosol"
                ).values
            )
            Se = xr.DataArray(Se, dims=["ret_alt"], coords=coords)
            error = (
                np.sqrt(Se)
                * np.exp(
                    ret_info.sel(ret_state="aerosol").isel(iteration=-1).target_state
                )
            ) * xsecs.isel(iteration=-1).interp(ret_alt=Se.ret_alt.values)
            final = aerosol_final.interp(altitude=Se.ret_alt.values)
            l4 = ax[0].fill_betweenx(
                Se.ret_alt.values / 1000,
                (final.values + error.values) * aerosol_scale,
                (final.values - error.values) * aerosol_scale,
                color=ret_color,
                lw=0,
                alpha=0.3,
            )
            # ax2[0].fill_betweenx(Se.ret_alt.values / 1000, -error.values * aerosol_scale,
            #                      error.values * aerosol_scale,
            #                      color=ret_color, lw=0, alpha=0.3)

            Se = np.diag(
                ret_info.solution_covariance.sel(
                    ret_state="aerosol_lognormal_medianradius",
                    pert_state="aerosol_lognormal_medianradius",
                ).values
            )
            Se = xr.DataArray(Se, dims=["ret_alt"], coords=coords)
            error = np.sqrt(Se) * radius_iterations.isel(iteration=-1)
            final = radius_final.interp(altitude=Se.ret_alt.values)
            if plot_effective_radius:
                error *= np.exp(np.log(1.6) ** 2 * 5 / 2)
                final *= np.exp(np.log(1.6) ** 2 * 5 / 2)
            l4 = ax[1].fill_betweenx(
                Se.ret_alt.values / 1000,
                (final.values + error.values),
                (final.values - error.values),
                color=ret_color,
                lw=0,
                alpha=0.3,
            )

        if true_state:
            leg = ax[0].legend(
                [l0, l2, l3, l1],
                ["A priori", "Iterations", "Retrieved", "True"],
                loc="upper right",
                framealpha=1,
                facecolor=ax[0].get_facecolor(),
                edgecolor="none",
                fontsize="small",
            )
        else:
            leg = ax[0].legend(
                [l0, l2, l3],
                ["A priori", "Iterations", "Retrieved"],
                loc="upper right",
                framealpha=1,
                facecolor=ax[0].get_facecolor(),
                edgecolor="none",
                fontsize="small",
            )

        leg.set_title("Retrieval State", prop={"size": "small", "weight": "bold"})

        ax[0].axhline(
            ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444"
        )
        ax[1].axhline(
            ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444"
        )
        ax[0].set_ylim(0, 40)
        # ax2[0].set_ylim(-1e)
        # ax2[1].set_ylim(-50, 50)

        ax3[0].set_xlim(-50, 50)
        ax3[1].set_xlim(-50, 50)
        ax[0].set_xlim(1e-6 * aerosol_scale, 1e-2 * aerosol_scale)
        ax[0].set_xscale("log")
        ax[0].set_xticks(10 ** np.arange(-6, -1.9) * aerosol_scale)


class LognormalEffectiveRadiusRetrieval(LognormalRadiusRetrieval):

    def median_radius_state_element(self):

        if self._aerosol_opt_prop is None:
            ps_clim = sk.ClimatologyUserDefined(
                self.altitudes, particle_size(self.altitudes)
            )
            self._aerosol_opt_prop = sk.MieAerosol(
                particlesize_climatology=ps_clim, species="H2SO4"
            )

        element = StateVectorProfileEffectiveRadius(
            altitudes_m=self.altitudes,
            values=np.log(self.apriori_profile() * 0 + 0.08),
            species_name="aerosol",
            optical_property=self._aerosol_opt_prop,
            lowerbound=self.lower_bound,
            upperbound=self.upper_bound,
            second_order_tikhonov_factor=self.particle_size_tikhonov_factor,
            second_order_tikhonov_altitude=self.particle_size_tikhonov_altitude,
        )
        return element

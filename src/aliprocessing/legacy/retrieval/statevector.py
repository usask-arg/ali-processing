from __future__ import annotations

from copy import copy

import numpy as np
import sasktran as sk
import xarray as xr
from skretrieval.legacy.retrieval.statevector.profile import StateVectorProfileLogND
from skretrieval.retrieval.tikhonov import two_dim_vertical_second_deriv


class StateVectorAerosolProfile(StateVectorProfileLogND):

    def __init__(
        self,
        altitudes_m: np.array,
        values: np.array,
        species_name: str,
        optical_property: sk.OpticalProperty,
        lowerbound: float,
        upperbound: float,
        second_order_tikhonov_factor: np.ndarray = np.zeros(2),
        second_order_tikhonov_altitude: np.ndarray = np.array((0.0, 100_000.0)),
        bounding_factor=1e4,
        zero_factor: float = 1e-20,
        max_update_factor: float = 5,
        scale_range: float = 200,
    ):
        """
        A state vector element which defines a vertical profile as the logarithm of number density.  The profile
        is bounded above and below by the upperbound and lowerbound through the use of the apriori.  A second
        order tikhonov smoothing is included.  Number dennsities less than a certain factor (zero_factor) are treated as
        identically equal to 0 within the RTM to prevent numerical precision problems

        Parameters
        ----------
        altitudes_m: np.array
            Altitudes of the profile, should match weighting function altitudes
        values: np.array
            log(number density) for the profile
        species_name: str
            Name of the species in the atmosphere
        optical_property: sk.OpticalProperty
            Optical property of the species
        lowerbound: float
            Lowerbound of the retrieval in m
        upperbound: float
            Upperbound of the retrieval in m
        second_order_tikhonov_factor: np.ndarray
            Altitude dependent second order tikhonov factor, should be the same size as second_order_tikhonov_altitude
        second_order_tikhonov_altitude: np.ndarray
            Altitudes of Tikhonov factors, should be the same size as second_order_tikhonov_factor
        bounding_factor: float, optional
            Bounding factor used for the apriori upper/lower bounding. Default 1e4
        zero_factor: float, optional
            Number densities less than this number are treated as identically 0.  Default 1e-20
        max_update_factor: float, optional
            Maximum factor to update any element of the state vector every iteration. Default 5
        scale_range: float
            The range in km that is used for scaling of the profile below the upper bound and above the lower bound.
            For example, if the `upper_bound` is 40km and `scale_range` is 5km, then the range between 35 and 40km will be
            used as the region to determine the scale factor above 40km.
        """

        self._second_order_tikhonov_factor = second_order_tikhonov_factor
        self._second_order_tikhonov_altitude = second_order_tikhonov_altitude

        super().__init__(
            altitudes_m=altitudes_m,
            values=values,
            species_name=species_name,
            optical_property=optical_property,
            lowerbound=lowerbound,
            upperbound=upperbound,
            second_order_tikhonov_factor=1.0,
            bounding_factor=bounding_factor,
            zero_factor=zero_factor,
            max_update_factor=max_update_factor,
        )

        self._second_order_tikhonov_factor = second_order_tikhonov_factor
        self._second_order_tikhonov_altitude = second_order_tikhonov_altitude
        self._upper_scale_range = scale_range
        self._lower_scale_range = scale_range
        self._upper_scale_alts = None
        self._lower_scale_alts = None
        self._compute_apriori_covariance()
        self._enabled = True

    @property
    def initial_state(self):
        return self._initial_state

    def max_value(self):
        return 1000.0

    def min_value(self):
        return 0.0

    def _compute_apriori_covariance(self):

        if type(self._second_order_tikhonov_factor * 1.0) == float:
            tikh_factor = self._second_order_tikhonov_factor
        else:
            tikh_factor = np.interp(
                self._altitudes_m[self.retrieval_alts()],
                self._second_order_tikhonov_altitude,
                self._second_order_tikhonov_factor,
            )

        # n = len(self._values[~self._zero_mask])
        n = len(self.state())
        # Link above

        # bound_alts = np.array([0, self._upperbound - 2000, self._upperbound + 2000, 100000.0])
        # bounds = np.array([0, 0, self._bounding_factor, self._bounding_factor])
        # bounding_factor = np.interp(self._altitudes_m, bound_alts, bounds)
        # gamma = two_dim_vertical_first_deriv(1, n, factor=bounding_factor)

        # gamma = two_dim_vertical_first_deriv(1, n, factor=self._bounding_factor)
        # bounded_altitudes = self._altitudes_m[~self._zero_mask] > self._upperbound
        # first_idx = np.nonzero(bounded_altitudes)[0][0]
        # bounded_altitudes[first_idx - 1] = True
        # gamma[~bounded_altitudes, :] = 0
        # self._inverse_Sa_bounding = gamma.T @ gamma
        #
        # # Link below
        # gamma = two_dim_vertical_first_deriv(1, n, factor=self._bounding_factor)
        # bounded_altitudes = self._altitudes_m[~self._zero_mask] < self._lowerbound
        # gamma[~bounded_altitudes, :] = 0
        #
        # self._inverse_Sa_bounding += gamma.T @ gamma
        self._inverse_Sa_bounding = np.zeros((n, n))

        gamma = two_dim_vertical_second_deriv(1, n, factor=tikh_factor)
        self._inverse_Sa = gamma.T @ gamma

    def retrieval_alts(self):
        return (self._altitudes_m < self._upperbound) & (
            self._altitudes_m > self._lowerbound
        )

    def state(self):
        return super().state()[self.retrieval_alts()]

    def update_state(self, x: np.array):
        m_factors = np.exp(x - (self._values[~self._zero_mask & self.retrieval_alts()]))
        m_factors = np.interp(
            self._altitudes_m,
            self._altitudes_m[self.retrieval_alts()],
            m_factors,
            left=m_factors[0],
            right=m_factors[-1],
        )

        # clip update factors to avoid too large of swings
        m_factors[m_factors > self._max_update_factor] = self._max_update_factor
        m_factors[m_factors < 1 / self._max_update_factor] = 1 / self._max_update_factor

        # compute the scale factor above the retrieval bounds
        m_factors[self._altitudes_m > self._upperbound] = np.mean(
            m_factors[self.upper_scale_alts]
        )

        # compute the scale factor below the retrieval bounds
        m_factors[self._altitudes_m <= self._lowerbound] = np.mean(
            m_factors[self.lower_scale_alts]
        )

        # apply scale factor in number density space and clip extreme values
        new_masked_values = m_factors * np.exp(self._values[~self._zero_mask])
        new_masked_values[new_masked_values > self.max_value()] = self.max_value()
        new_masked_values[new_masked_values < self.min_value()] = self.min_value()

        self._values[~self._zero_mask] = np.log(copy(new_masked_values))
        self._internal_update()

    @property
    def upper_scale_alts(self):
        """
        Altitudes in the upper altitude range used for profile scaling
        """
        good = (self._altitudes_m > self._upperbound - self._upper_scale_range) & (
            self._altitudes_m < self._upperbound
        )
        if sum(good) == 0:  # if no values in range, choose highest possible altitude
            good[np.where(self._altitudes_m < self._upperbound)[0][-1]] = True
        self._upper_scale_alts = good
        return self._upper_scale_alts

    @property
    def lower_scale_alts(self):
        """
        Altitudes in the lower altitude range used for profile scaling
        """
        good = (self._altitudes_m > self._lowerbound) & (
            self._altitudes_m < self._lowerbound + self._lower_scale_range
        )
        if sum(good) == 0:  # if no values in range, choose lowest possible altitude
            good[np.where(self._altitudes_m > self._lowerbound)[0][0]] = True
        self._lower_scale_alts = good
        return self._lower_scale_alts

    def propagate_wf(self, radiance) -> xr.Dataset:

        wf = radiance["wf_" + self.name()]
        # upper_scale_alts = (self._altitudes_m > (self._upperbound - self._upper_scale_range)) & \
        #                    (self._altitudes_m < self._upperbound)
        above_scale_alts = self._altitudes_m >= self._upperbound
        Ns = np.sum(self.upper_scale_alts)
        if Ns is None:
            Ns = 1
        if len(wf.shape) == 3:
            new_wf = wf * np.exp(self._values)[np.newaxis, np.newaxis, :]
            # wf_above = new_wf[:, self.upper_scale_alts].sum(axis=1)
            # new_wf[:, :, self.upper_scale_alts] = new_wf[:, :, self.upper_scale_alts] + wf_above[:, np.newaxis] / Ns
            good = ~self._zero_mask & self.retrieval_alts()
            return new_wf[:, :, good]
        else:
            new_wf = wf * np.exp(self._values)[np.newaxis, :]
            new_wf[:, above_scale_alts] *= (
                new_wf[:, above_scale_alts]
                * np.exp(self._values)[np.newaxis, above_scale_alts]
            )
            wf_above = new_wf[:, above_scale_alts].sum(axis=1)
            new_wf[:, self.upper_scale_alts] = (
                new_wf[:, self.upper_scale_alts] + wf_above[:, np.newaxis] / Ns
            )
            good = ~self._zero_mask & self.retrieval_alts()
            return new_wf[:, good]


class StateVectorProfileParticleSize(StateVectorAerosolProfile):

    def __init__(
        self,
        altitudes_m: np.array,
        values: np.array,
        species_name: str,
        optical_property: sk.OpticalProperty,
        size_type: str,
        lowerbound: float,
        upperbound: float,
        second_order_tikhonov_factor: np.ndarray = np.zeros(2),
        second_order_tikhonov_altitude: np.ndarray = np.array((0.0, 100_000.0)),
        bounding_factor=1e4,
        zero_factor: float = 1e-20,
        max_update_factor: float = 5,
    ):

        self._aerosol_species_name = species_name
        self._type = size_type
        self.min_radius = 0.04
        self.max_radius = 0.5
        self.min_width = 1.05
        self.max_width = 2.3

        if self._type == "lognormal_medianradius":
            self._speciesid = "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS"
        elif self._type == "lognormal_modewidth":
            self._speciesid = "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH"

        super().__init__(
            altitudes_m=altitudes_m,
            values=values,
            species_name=species_name,
            optical_property=optical_property,
            lowerbound=lowerbound,
            upperbound=upperbound,
            second_order_tikhonov_factor=second_order_tikhonov_factor,
            second_order_tikhonov_altitude=second_order_tikhonov_altitude,
            bounding_factor=bounding_factor,
            zero_factor=zero_factor,
            max_update_factor=max_update_factor,
        )

    def _internal_update(self):
        """
        Updates the internal climatology
        """
        altitudes, values = self._climatology_altitudes_and_density()
        self._optical_property.particlesize_climatology[self._speciesid] = values

    def name(self) -> str:
        return self._aerosol_species_name + "_" + self._type

    def max_value(self):
        if self._type == "lognormal_modewidth":
            return self.max_width
        if self._type == "lognormal_medianradius":
            return self.max_radius

    def min_value(self):
        if self._type == "lognormal_modewidth":
            return self.min_width
        if self._type == "lognormal_medianradius":
            return self.min_radius

    # def update_state(self, x: np.array):
    #
    #     m_factors = np.exp(x - (self._values[~self._zero_mask]))
    #     m_factors[m_factors > self._max_update_factor] = self._max_update_factor
    #     m_factors[m_factors < 1 / self._max_update_factor] = 1 / self._max_update_factor
    #     self._values[~self._zero_mask] = np.log(copy(m_factors * np.exp(self._values[~self._zero_mask])))
    #
    #     if self._type == 'lognormal_modewidth':
    #         self._values[self._values > np.log(self.max_width)] = np.log(self.max_width)
    #         self._values[self._values < np.log(self.min_width)] = np.log(self.min_width)
    #     if self._type == 'lognormal_medianradius':
    #         self._values[self._values > np.log(self.max_radius)] = np.log(self.max_radius)
    #         self._values[self._values < np.log(self.min_radius)] = np.log(self.min_radius)
    #     self._internal_update()

    def add_to_atmosphere(self, atmosphere: sk.Atmosphere):
        """
        Adds the species to the atmosphere

        Parameters
        ----------
        atmosphere: sk.Atmosphere

        """
        aer_wf = self._aerosol_species_name + "_" + self._type
        if atmosphere.wf_species is None:
            atmosphere.wf_species = [aer_wf]

        if aer_wf not in atmosphere.wf_species:
            old_wf_species = copy(atmosphere.wf_species)
            old_wf_species.append(aer_wf)
            atmosphere.wf_species = old_wf_species


class StateVectorCloudProfile(StateVectorAerosolProfile):

    def max_value(self):
        return 0.01


class StateVectorProfileEffectiveRadius(StateVectorProfileParticleSize):

    def __init__(
        self,
        altitudes_m: np.array,
        values: np.array,
        species_name: str,
        optical_property: sk.OpticalProperty,
        lowerbound: float,
        upperbound: float,
        second_order_tikhonov_factor: np.ndarray = np.zeros(2),
        second_order_tikhonov_altitude: np.ndarray = np.array((0.0, 100_000.0)),
        bounding_factor=1e4,
        zero_factor: float = 1e-20,
        max_update_factor: float = 5,
    ):

        self._aerosol_species_name = species_name
        self.min_radius = 0.04
        self.max_radius = 0.5
        self.min_width = 1.05
        self.max_width = 2.3

        super().__init__(
            altitudes_m=altitudes_m,
            values=values,
            species_name="lognormal_effective_radius",
            optical_property=optical_property,
            lowerbound=lowerbound,
            upperbound=upperbound,
            second_order_tikhonov_factor=second_order_tikhonov_factor,
            second_order_tikhonov_altitude=second_order_tikhonov_altitude,
            bounding_factor=bounding_factor,
            zero_factor=zero_factor,
            size_type="lognormal_medianradius",
            max_update_factor=max_update_factor,
        )

    def name(self):
        return self._aerosol_species_name + "_" + self._species_name

    def max_value(self):
        return 1000.0

    def min_value(self):
        return 0.0

    def lognormal_params_from_effective_radius(self, reff):

        # log(sg) = a * log(rm) + b from 2009-2019 LPC fine mode data
        p = np.array([-0.30773813, -0.40106275])

        rg_line = np.arange(0.04, 0.25, 0.001)
        sg_line = np.exp(p[0] * np.log(rg_line) + p[1])
        reff_line = rg_line * np.exp(5 / 2 * (np.log(sg_line) ** 2))

        rg = np.interp(reff, reff_line, rg_line, left=rg_line[0], right=rg_line[-1])
        sg = np.interp(reff, reff_line, sg_line, left=sg_line[0], right=sg_line[-1])

        return sg, rg

    def update_state(self, x: np.array):

        m_factors = np.exp(x - (self._values[~self._zero_mask]))
        m_factors[m_factors > self._max_update_factor] = self._max_update_factor
        m_factors[m_factors < 1 / self._max_update_factor] = 1 / self._max_update_factor
        new_values = np.log(copy(m_factors * np.exp(self._values[~self._zero_mask])))
        self._values[~self._zero_mask] = np.log(
            copy(m_factors * np.exp(self._values[~self._zero_mask]))
        )

        self._internal_update()

    def _internal_update(self):
        """
        Updates the internal climatology
        """
        altitudes, values = self._climatology_altitudes_and_density()
        sg, rg = self.lognormal_params_from_effective_radius(values)
        self._optical_property.particlesize_climatology[
            "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS"
        ] = sg
        self._optical_property.particlesize_climatology[
            "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH"
        ] = rg

    def add_to_atmosphere(self, atmosphere: sk.Atmosphere):
        """
        Adds the species to the atmosphere

        Parameters
        ----------
        atmosphere: sk.Atmosphere

        """
        rm_wf = self._aerosol_species_name + "_" + "lognormal_medianradius"
        sg_wf = self._aerosol_species_name + "_" + "lognormal_modewidth"
        if atmosphere.wf_species is None:
            atmosphere.wf_species = [rm_wf, sg_wf]

        if rm_wf not in atmosphere.wf_species:
            old_wf_species = copy(atmosphere.wf_species)
            old_wf_species.append(rm_wf)
            atmosphere.wf_species = old_wf_species

        if sg_wf not in atmosphere.wf_species:
            old_wf_species = copy(atmosphere.wf_species)
            old_wf_species.append(sg_wf)
            atmosphere.wf_species = old_wf_species

    def propagate_wf(self, radiance) -> xr.Dataset:
        # wf = radiance['wf_' + self.name()]

        altitudes, values = self._climatology_altitudes_and_density()
        sg, rm = self.lognormal_params_from_effective_radius(values)

        wf_rg = f"wf_{self._aerosol_species_name}_lognormal_medianradius"
        wf_sg = f"wf_{self._aerosol_species_name}_lognormal_modewidth"

        x = np.sqrt(np.log(values / rm))
        wf = radiance[wf_rg] * np.exp(-5 / 2 * (np.log(sg) ** 2)) + radiance[
            wf_sg
        ] * np.exp(np.sqrt(2 / 5) * x) / (np.sqrt(10) * values * x)

        if len(wf.shape) == 3:
            new_wf = wf * np.exp(self._values)[np.newaxis, np.newaxis, :]
            return new_wf[:, :, ~self._zero_mask]
        else:
            new_wf = wf * np.exp(self._values)[np.newaxis, :]
            return new_wf[:, ~self._zero_mask]

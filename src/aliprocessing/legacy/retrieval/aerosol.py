from __future__ import annotations

from copy import copy

# from ali.retrieval.measvec.cloud import CloudDetection
from typing import List, Union

import numpy as np
from ali_processing.legacy.retrieval.measvec import MeasurementVector
from skretrieval.core.radianceformat import RadianceBase
from skretrieval.retrieval import ForwardModel
from skretrieval.retrieval.statevector import StateVector, StateVectorElement
from skretrieval.retrieval.target import GenericTarget


class GenericTargetWithMeasVec(GenericTarget):

    def __init__(
        self, state_vector: StateVector, measurement_vector: MeasurementVector
    ):
        """
        Implements a generic abstract base target class that is composed of a `StateVector`.  A `MeasurementVector` will
        generally return a dictionary of `np.ndarray`, so the `propagate_wf` of the `state_elements` should be able
        to handle this type.

        Parameters
        ----------
        state_vector: StateVector
        """
        super().__init__(state_vector)
        self._meas_vec_def = measurement_vector

    def _internal_measurement_vector(self, l1_data: RadianceBase | list[RadianceBase]):

        # propagate the level 1 data through the measurement transforms
        result = self._meas_vec_def.meas_dict(l1_data)

        # propagate the jacobians through the state vector elements
        wf_keys = [k for k in result.keys() if "wf" in k]
        if wf_keys:
            all_jacobians = []
            for s in self._state_vector.state_elements:
                all_jacobians.append(s.propagate_wf(result))
            result["jacobian"] = np.hstack(all_jacobians)

        return result


class AerosolRetrieval(GenericTargetWithMeasVec):

    def __init__(
        self,
        state_vector: StateVector,
        measurement_vector: MeasurementVector,
        retrieval_altitudes: np.ndarray = None,
    ):
        """

        Parameters
        ----------
        state_vector : StateVector
            Vectors defining the various state elements that will be retrieved.
        measurement_vector : MeasurementVector
            Vectors defining the measurements that will be used in the inversion
        """

        self._min_altitude = 5500.0
        self._max_altitude = 35000.0
        self._retrieval_alt_res = 250.0
        if retrieval_altitudes is None:
            self._retrieval_altitudes = np.arange(
                self._min_altitude, self._max_altitude, self._retrieval_alt_res
            )
        else:
            self._retrieval_altitudes = retrieval_altitudes

        self._tangent_altitudes = None
        self.save_output = False
        self.output = []
        self.rayleigh_radiance = None
        self.y_rayleigh = None
        self.rayleigh_norm = True

        super().__init__(state_vector, measurement_vector)

    def _internal_measurement_vector(
        self, l1_data: RadianceBase | list[RadianceBase], rayleigh_norm=None
    ):

        if rayleigh_norm is None:
            rayleigh_norm = self.rayleigh_norm

        result = super()._internal_measurement_vector(l1_data)

        if rayleigh_norm:
            if self.y_rayleigh:
                result["y"] = result["y"] - self.y_rayleigh["y"]

        if self.save_output:
            self.output.append(result)
        return result

    def configure_from_model(self, forward_model, measurement_l1):
        if "aerosol" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["aerosol"].species
            old_aerosol = copy(
                forward_model.atmosphere.species["aerosol"].climatology[spec]
            )
            forward_model.atmosphere.species["aerosol"].climatology[spec] = (
                np.zeros_like(old_aerosol)
            )

        if "icecloud" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["icecloud"].species
            old_cloud = copy(
                forward_model.atmosphere.species["icecloud"].climatology[spec]
            )
            forward_model.atmosphere.species["icecloud"].climatology[spec] = (
                np.zeros_like(old_cloud)
            )

        self.rayleigh_radiance = forward_model.calculate_radiance()

        if "aerosol" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["aerosol"].species
            forward_model.atmosphere.species["aerosol"].climatology[spec] = old_aerosol

        if "icecloud" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["icecloud"].species
            forward_model.atmosphere.species["icecloud"].climatology[spec] = old_cloud

        self.y_rayleigh = self._internal_measurement_vector(
            self.rayleigh_radiance, rayleigh_norm=False
        )

    @property
    def min_altitude(self):
        return self._min_altitude

    @min_altitude.setter
    def min_altitude(self, value):
        self._min_altitude = value

    @property
    def max_altitude(self):
        return self._max_altitude

    @max_altitude.setter
    def max_altitude(self, value):
        self._max_altitude = value

    def initialize(
        self,
        forward_model: ForwardModel,
        meas_l1: RadianceBase | list[RadianceBase],
    ):

        super().initialize(forward_model, meas_l1)
        if type(meas_l1) is list:
            tangent_locations = meas_l1[0].tangent_locations()
        else:
            tangent_locations = meas_l1.tangent_locations()

        self._tangent_altitudes = tangent_locations.altitude

    @property
    def jacobian_altitudes(self):
        return self._retrieval_altitudes


class ParticleSizeRetrieval(GenericTargetWithMeasVec):

    def __init__(
        self,
        state_vector: StateVector,
        measurement_vector: MeasurementVector,
        retrieval_altitudes=None,
    ):

        self._min_altitude = 5500.0
        self._max_altitude = 35000.0
        self._retrieval_alt_res = 250.0
        if retrieval_altitudes is None:
            self._retrieval_altitudes = np.arange(
                self._min_altitude, self._max_altitude, self._retrieval_alt_res
            )
        else:
            self._retrieval_altitudes = retrieval_altitudes

        self._tangent_altitudes = None
        self.wavelength = 750.0
        self.save_output = False
        self.output = []
        self.rayleigh_radiance = None
        self.y_rayleigh = None

        super().__init__(state_vector, measurement_vector)

    def configure_from_model(self, forward_model, measurement_l1):
        if "aerosol" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["aerosol"].species
            old_aerosol = copy(
                forward_model.atmosphere.species["aerosol"].climatology[spec]
            )
            forward_model.atmosphere.species["aerosol"].climatology[spec] = (
                np.zeros_like(old_aerosol)
            )

        if "ice_cloud" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["ice_cloud"].species
            old_cloud = copy(
                forward_model.atmosphere.species["ice_cloud"].climatology[spec]
            )
            forward_model.atmosphere.species["ice_cloud"].climatology[spec] = (
                np.zeros_like(old_cloud)
            )

        self.rayleigh_radiance = forward_model.calculate_radiance()

        if "aerosol" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["aerosol"].species
            forward_model.atmosphere.species["aerosol"].climatology[spec] = old_aerosol

        if "ice_cloud" in forward_model.atmosphere.species.keys():
            spec = forward_model.atmosphere.species["ice_cloud"].species
            forward_model.atmosphere.species["ice_cloud"].climatology[spec] = old_cloud

        self.y_rayleigh = self._internal_measurement_vector(
            self.rayleigh_radiance, rayleigh_norm=False
        )

    def _internal_measurement_vector(
        self, l1_data: RadianceBase | list[RadianceBase], rayleigh_norm=True
    ):

        result = super()._internal_measurement_vector(l1_data)

        if rayleigh_norm:
            if self.y_rayleigh:
                result["y"] = result["y"] - self.y_rayleigh["y"]

        if self.save_output:
            self.output.append(result)

        return result

    @property
    def jacobian_altitudes(self):
        return self._retrieval_altitudes

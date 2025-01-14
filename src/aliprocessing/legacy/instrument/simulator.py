from __future__ import annotations

import time
from copy import copy
from typing import Dict, List, Union

import numpy as np
import sasktran as sk
import xarray as xr
from ali_processing.legacy.instrument.sensor import ALISensor
from ali_processing.legacy.instrument.sensor_2channel import ALISensorDualChannel
from ali_processing.legacy.util.sampling import Sampling
from sasktran import Atmosphere, SolarSpectrum
from skretrieval.core import OpticalGeometry
from skretrieval.core.radianceformat import RadianceGridded, RadianceSpectralImage

# from skretrieval.legacy.core.sensor.imager import SpectralImager
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph
from skretrieval.legacy.tomography.grids import OrbitalPlaneGrid
from skretrieval.legacy.tomography.sasktran_ext.engine import EngineHRTwoDim
from skretrieval.retrieval import ForwardModel


class ImagingSimulator(ForwardModel):
    """
    Given a sensor, position, and an atmosphere the `ImagingSimulator` generates a set of measurements.

    Parameters
    ----------

    sensors: List[ALISensor]
        A list of sensors that will be used to create the level 1 data. Should be derived from an imaging simulator.
    optical_axis: List[OpticalGeometry]
        The optical geometry of each measurement. Should be a list the same length as sensors.
    atmosphere: sk.Atmosphere
        The Sasktran atmosphere.
    options:
        A dictionary of radiative transfer options.
    frontend_radiance: List[xr.Dataset]
        A dataset containing the pre-calculated front-end radiances. This can be useful if radiance calculations take
        a substantial amount of time. If not provided front-end radiances will be calculated.
    grid: sk.OrbitalPlaneGrid
        An optional `OrbitalPlaneGrid` in the case that an orbit is being simulated.


    Examples
    --------

    >>> import sasktran as sk
    >>> from ali_processing.legacy.instrument.sensor import ALISensor
    >>> from ali_processing.legacy.instrument.simulator import ImagingSimulator
    >>> from ali_processing.legacy.util.geometry import optical_axis_from_geometry
    >>> import numpy as np
    >>> ali = ALISensor(np.array([750.0]))
    >>> geometry = sk.VerticalImage()
    >>> geometry.from_sza_ssa(sza=60, ssa=90, lat=45.0, lon=0, tanalts_km=image_center_alts, mjd=mjd, locallook=0, satalt_km=500)
    >>> optical_geometry = optical_axis_from_geometry(geometry)
    >>> atmosphere = sk.Atmosphere()
    >>> atmosphere['air'] = sk.Species(sk.Rayleigh(), sk.MSIS90())
    >>> sim = ImagingSimulator([ali], [optical_geometry], atmosphere)
    >>> l1 = sim.calculate_radiance()
    """

    def __init__(
        self,
        sensors: list[ALISensor | ALISensorDualChannel],
        optical_axis: list[OpticalGeometry],
        atmosphere: Atmosphere,
        options: dict = None,
        frontend_radiance: list[xr.Dataset] = None,
        grid: OrbitalPlaneGrid = None,
    ):

        self.sensors = sensors

        self.num_horizontal_samples = None
        self.num_vertical_samples = None
        self.wavelength_resolution = None
        self.solar_wavelength_resolution = 0.01
        self.group_scans = True
        self.dual_polarization = False
        self._grid = grid

        self.optical_axis = optical_axis
        # self.model_geometry = sk.Geometry()

        self.atmosphere = atmosphere
        self.options = options
        self.sun = None

        self.store_radiance = True
        self._frontend_radiance = frontend_radiance
        self._engine = None
        self._model_geometries = None
        self.random_seed = None

        try:
            self._integration_time = [s.exposure_time for s in sensors]
        except AttributeError:
            self._integration_time = 0.05

        self._save_diagnostics = False
        self.diagnostics = {"sensor": {"samples": []}}

        self.calculate_brdf_wf = False
        self.grid_sampling = False  # If true radiative transfer calculations are done on a (downsampled) sensor grid
        self.convert_single_col_to_gridded = False
        self.engine_diagnostics = {"referencepoint": [], "sun": []}

    @property
    def save_diagnostics(self):
        return self._save_diagnostics

    @save_diagnostics.setter
    def save_diagnostics(self, value: bool):
        self._save_diagnostics = value
        for sensor in self.sensors:
            sensor.save_diagnostics = value

    @property
    def integration_time(self):
        return self._integration_time

    @integration_time.setter
    def integration_time(self, value):
        self._integration_time = value
        if (type(value) is float) or (type(value) is int):
            value = [value for s in self.sensors]
        for sensor, v in zip(self.sensors, value, strict=False):
            sensor.exposure_time = v

    @property
    def model_geometries(self):

        if self._model_geometries is None:

            geom = []
            for idx, (sensor, optical_axis) in enumerate(
                zip(self.sensors, self.optical_axis, strict=False)
            ):
                if self.grid_sampling:
                    model_geometry = sk.Geometry()
                    # downsample the sensor using the simulator num_rows and num_cols parameters
                    model_geometry.lines_of_sight = sensor.measurement_geometry(
                        optical_axis,
                        num_rows=self.num_vertical_samples,
                        num_columns=self.num_horizontal_samples,
                    )
                    geom.append(model_geometry)
                else:
                    if type(sensor) is ALISensorDualChannel:
                        samples = Sampling(sensor.channel_1, optical_axis)
                    else:
                        samples = Sampling(sensor, optical_axis)
                    # samples.horizontal_spacing_ground = 10
                    samples.horizontal_spacing_ground = 5
                    samples.horizontal_spacing_toa = 1
                    samples.vertical_spacing_toa = 1
                    samples.vertical_spacing_ground = 0.4
                    samples.truncate_edges = True
                    geom.append(samples.meas_geom())
                    if type(sensor) is ALISensorDualChannel:
                        samples.set_weights(sensor.channel_1)
                        samples.set_weights(sensor.channel_2)
                    else:
                        samples.set_weights(sensor)
                    if self.save_diagnostics:
                        self.diagnostics["sensor"]["samples"].append(samples)

            self._model_geometries = geom
        return self._model_geometries

    @property
    def optical_geometries(self):

        geom = []
        for idx, (sensor, optical_axis) in enumerate(
            zip(self.sensors, self.optical_axis, strict=False)
        ):
            geom.append(sensor.optical_geometries(optical_axis))
        return geom

    @property
    def rtm_opts(self):

        opts = self.options.copy()
        if "polarization" in opts:
            del opts["polarization"]

        if "grid_spacing" in opts:
            del opts["grid_spacing"]

        if "configureforcloud" in opts:
            del opts["configureforcloud"]

        return opts

    @property
    def grid_spacing(self):
        if "manualraytracingshells" in self.options:
            return None
        grid_spacing = 1000.0
        if "grid_spacing" in self.options:
            grid_spacing = self.options["grid_spacing"]
        return grid_spacing

    @property
    def cloudconfiguration(self):
        configureforcloud = None
        if "configureforcloud" in self.options:
            configureforcloud = self.options["configureforcloud"]
        return configureforcloud

    def sensor_wavelengths(self, sensor=None):
        if sensor is None:
            if self.wavelength_resolution is not None:
                return np.unique(
                    np.concatenate(
                        [
                            sensor.required_wavelengths(self.wavelength_resolution)
                            for sensor in self.sensors
                        ]
                    )
                )
            else:
                return np.unique(
                    np.concatenate([sensor._wavelength_nm for sensor in self.sensors])
                )

        if self.wavelength_resolution:
            return sensor.required_wavelengths(self.wavelength_resolution)
        else:
            return sensor._wavelength_nm

    def apply_solar_spectrum(self, radiance, model_wavel):

        if self.wavelength_resolution is None:
            wr = 10
        else:
            wr = self.wavelength_resolution
        if len(model_wavel) == 1:
            wavel_left = [model_wavel - wr / 2]
            wavel_right = [model_wavel + wr / 2]
        else:
            wavel_diff = wr  # np.diff(model_wavel)
            wavel_left = model_wavel - wavel_diff / 2
            wavel_right = model_wavel + wavel_diff / 2

        irradiance = np.ones_like(model_wavel, dtype=float)
        solar_spectrum = SolarSpectrum("SAO2010")
        solar_spectrum_lw = SolarSpectrum("FONTELA_UVIS_3MICRON")

        try:
            wf = [key for key in radiance.keys() if "wf" in key]
        except IndexError:
            wf = False
        for widx, (lw, rw) in enumerate(zip(wavel_left, wavel_right, strict=False)):
            hr_wavel = np.arange(
                lw,
                rw + self.solar_wavelength_resolution / 2,
                self.solar_wavelength_resolution,
            )
            if all(hr_wavel >= 1000):
                irradiance[widx] = np.nanmean(solar_spectrum_lw.irradiance(hr_wavel))
            elif all(hr_wavel <= 1000):
                irradiance[widx] = np.nanmean(solar_spectrum.irradiance(hr_wavel))
            else:
                lw = hr_wavel[hr_wavel >= 1000]
                sw = hr_wavel[hr_wavel < 1000]
                swi = np.nanmean(solar_spectrum.irradiance(sw))
                lwi = np.nanmean(solar_spectrum_lw.irradiance(lw))
                irradiance[widx] = (lwi * len(lw) + swi * len(sw)) / len(hr_wavel)

            radiance["I"][widx, :] = radiance["I"][widx, :] * irradiance[widx]
            radiance["Q"][widx, :] = radiance["Q"][widx, :] * irradiance[widx]
            radiance["U"][widx, :] = radiance["U"][widx, :] * irradiance[widx]
            radiance["V"][widx, :] = radiance["V"][widx, :] * irradiance[widx]

            if wf:
                if type(wf) is list:
                    for w in wf:
                        if len(radiance[w].shape) == 2:
                            radiance[w][widx, :] = (
                                radiance[w][widx, :] * irradiance[widx]
                            )
                        else:
                            radiance[w][widx, :, :] = (
                                radiance[w][widx, :, :] * irradiance[widx]
                            )
                else:
                    radiance[wf][widx, :, :] = (
                        radiance[wf][widx, :, :] * irradiance[widx]
                    )

        return radiance

    def calculate_radiance(self):

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        if self.group_scans and len(self.sensors) > 1:
            radiance_list = self._calculate_frontend_radiance_grouped()
        else:
            radiance_list = self._calculate_frontend_radiance()

        stacked_data = []
        for idx, (sensor, optical_axis, radiance, model_geometry) in enumerate(
            zip(
                self.sensors,
                self.optical_axis,
                radiance_list,
                self.model_geometries,
                strict=False,
            )
        ):
            model_wavelengths = self.sensor_wavelengths(sensor)

            if hasattr(radiance, "weighting_function"):
                wf = radiance.weighting_function
                radiance = radiance.radiance
                # model_values = sensor.model_radiance(optical_axis, model_wavelengths, model_geometry, rad, wf)
            elif any(["wf_" in k for k in radiance.keys()]):
                wf = []
                for key in radiance.keys():
                    if "wf_" in key:
                        wf.append(radiance[key])
                # model_values = sensor.model_radiance(optical_axis, model_wavelengths, model_geometry, radiance, wf)
            else:
                wf = None

            if self.dual_polarization:
                [s.turn_rotator_on() for s in self.sensors]
                model_values_on = sensor.model_radiance(
                    optical_axis, model_wavelengths, model_geometry, radiance, wf
                )
                [s.turn_rotator_off() for s in self.sensors]
                model_values_off = sensor.model_radiance(
                    optical_axis, model_wavelengths, model_geometry, radiance, wf
                )
                if type(model_values_on) is list:
                    model_values = model_values_on + model_values_off
                else:
                    model_values = [model_values_on, model_values_off]
            else:
                model_values = sensor.model_radiance(
                    optical_axis, model_wavelengths, model_geometry, radiance, wf
                )

            if type(model_values) is list:
                for m in model_values:
                    stacked_data.append(m)
            else:
                stacked_data.append(model_values)

        if (self.sensors[0].num_columns == 1) and self.convert_single_col_to_gridded:
            stacked_data = [s.to_gridded() for s in stacked_data]
        return stacked_data

    def _calculate_frontend_radiance_grouped(self) -> list[xr.Dataset]:

        wavelengths = self.sensor_wavelengths()

        if self._grid is None:
            self._grid = OrbitalPlaneGrid(
                self.optical_axis,
                grid_altitudes=np.arange(0.0, 100_000.0, 1000.0),
                placementtype="uniform",
                extend=10,
            )
        self._engine = EngineHRTwoDim(
            self.model_geometries,
            self.atmosphere,
            self._grid,
            wavelength=wavelengths,
            grid_spacing=self.grid_spacing,
            common_options=self.rtm_opts,
            max_difference_seconds=1.0,
        )
        radiance = self._engine.calculate_radiance(full_stokes_vector=True)

        wf_names = [key for key in radiance.keys() if "wf_" in key]

        if wf_names:
            for wf_name in wf_names:
                los_idx = 0
                num_perts = radiance[wf_name][0].shape[1]
                for rad in radiance["radiance"]:
                    num_los = len(rad.I.los.values)

                    if len(radiance[wf_name][0].shape) == 2:
                        wf = np.array(
                            [
                                radiance[wf_name][0][
                                    los_idx : num_los + los_idx
                                ].toarray()
                            ]
                        )
                    else:
                        wf = radiance[wf_name][0][
                            :, los_idx : num_los + los_idx
                        ].toarray()

                    wf = xr.DataArray(
                        wf,
                        dims=["wavelength", "los", "perturbation"],
                        coords=[rad.wavelength, rad.los, np.arange(0, num_perts)],
                    )
                    rad[wf_name] = wf
                    los_idx += num_los

        return [
            self.apply_solar_spectrum(r, model_wavel=wavelengths)
            for r in radiance["radiance"]
        ]

    def _calculate_frontend_radiance(self) -> list[xr.Dataset]:

        if self._frontend_radiance is None:
            self._frontend_radiance = [None] * len(self.optical_axis)

        stacked_radiance = []
        self.engine_diagnostics = {"referencepoint": [], "sun": []}
        for idx, (sensor, optical_axis, model_geometry) in enumerate(
            zip(self.sensors, self.optical_axis, self.model_geometries, strict=False)
        ):

            model_wavelengths = self.sensor_wavelengths(sensor)

            if self._frontend_radiance[idx] is None:
                self._engine = sk.EngineHR(
                    model_geometry, self.atmosphere, options=self.rtm_opts
                )
                self._engine.wavelengths = model_wavelengths
                # self._engine.num_diffuse_profiles = 5
                if self.cloudconfiguration:
                    self._engine.configure_for_cloud(**self.cloudconfiguration)
                elif self.grid_spacing:
                    self._engine.grid_spacing = self.grid_spacing

                if self.sun is not None:
                    self._engine.geometry.sun = self.sun

                self._engine.polarization = "vector"
                t0 = time.perf_counter()
                radiance = self._engine.calculate_radiance(
                    full_stokes_vector=True, output_format="xarray"
                )
                if not self.calculate_brdf_wf and ("wf_brdf" in radiance.keys()):
                    radiance = radiance.drop("wf_brdf")
                radiance = self.apply_solar_spectrum(
                    radiance, model_wavel=model_wavelengths
                )

                self.engine_diagnostics["referencepoint"].append(
                    self._engine.model_parameters["referencepoint"]
                )
                self.engine_diagnostics["sun"].append(
                    self._engine.model_parameters["sun"]
                )

                t1 = time.perf_counter()
                if self.store_radiance:
                    self._frontend_radiance[idx] = radiance
            else:
                radiance = self._frontend_radiance[idx]
            stacked_radiance.append(radiance)
        return stacked_radiance


class OMPSImagingSimulator(ForwardModel):

    def __init__(
        self,
        sensors: list[Spectrograph],
        optical_axis: list[OpticalGeometry],
        atmosphere: Atmosphere,
        options: dict = None,
    ):

        self.sensors = sensors

        self.num_horizontal_samples = None
        self.num_vertical_samples = None
        self.wavelength_resolution = None

        self.optical_axis = optical_axis
        # self.model_geometry = sk.Geometry()

        self.atmosphere = atmosphere
        self.options = options
        del self.options["polarization"]
        self.sun = None
        self._engine = None
        self.polarization = False
        self.engine_diagnostics = {"referencepoint": [], "sun": []}
        self.hires_geometry = None

    def calculate_radiance(self):

        if self.hires_geometry is None:
            model_los = [
                sk.LineOfSight(
                    look_vector=oa.look_vector, observer=oa.observer, mjd=oa.mjd
                )
                for oa in self.optical_axis
            ]
            geometry = sk.Geometry()
            geometry.lines_of_sight = model_los
        else:
            geometry = sk.Geometry()
            geometry.lines_of_sight = [
                sk.LineOfSight(
                    look_vector=oa.look_vector, observer=oa.observer, mjd=oa.mjd
                )
                for oa in self.hires_geometry
            ]

        model_wavel = np.unique([s.wavelength_nm for s in self.sensors])
        self._engine = sk.EngineHR(
            geometry, self.atmosphere, options=self.options, wavelengths=model_wavel
        )
        radiance = self._engine.calculate_radiance(output_format="xarray")

        rp = self._engine.model_parameters["referencepoint"]
        sun = self._engine.model_parameters["sun"]
        self.engine_diagnostics["referencepoint"] = [rp for i in self.optical_axis]
        self.engine_diagnostics["sun"] = [sun for i in self.optical_axis]

        if "wf_brdf" in radiance.keys():
            radiance = radiance.drop("wf_brdf")

        if hasattr(radiance, "weighting_function"):
            wf = radiance.weighting_function
            radiance = radiance.radiance
            # model_values = sensor.model_radiance(optical_axis, model_wavelengths, model_geometry, rad, wf)
        elif any(["wf_" in k for k in radiance.keys()]):
            wf = []
            for key in radiance.keys():
                if "wf_" in key:
                    wf.append(radiance[key])
            radiance = radiance.radiance
            # model_values = sensor.model_radiance(optical_axis, model_wavelengths, model_geometry, radiance, wf)
        else:
            wf = None

        rad = []
        for oa, sensor in zip(self.optical_axis, self.sensors, strict=False):
            rad.append(
                sensor.model_radiance(
                    optical_geometry=oa,
                    model_wavel_nm=model_wavel,
                    model_geometry=geometry,
                    radiance=radiance,
                    wf=xr.merge(wf),
                )
            )
        # measurement_l1 = xr.Dataset({'radiance': (['los', 'wavelength'], np.array([radiance]).T),
        #                              'los_vector': (['los', 'xyz'], [l.look_vector for l in los]),
        #                              'observer_position': (['los', 'xyz'], [sat for i in range(len(opt_geom))])},
        #                             coords={'los': np.arange(0, len(opt_geom)), 'wavelength': [745.0],
        #                                     'xyz': ['x', 'y', 'z']})
        # measurement_l1 = RadianceGridded(measurement_l1)
        return [
            RadianceSpectralImage(
                xr.concat([r.data for r in rad], dim="los"), num_columns=1
            )
        ]

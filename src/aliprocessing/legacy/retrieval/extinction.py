from __future__ import annotations

import os
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sasktran as sk
import xarray as xr
from ali_processing.legacy.instrument.sensor_2channel import ALISensorDualChannel
from ali_processing.legacy.instrument.simulator import (
    ImagingSimulator,
    OMPSImagingSimulator,
)
from ali_processing.legacy.retrieval.aerosol import AerosolRetrieval
from ali_processing.legacy.retrieval.measvec import (
    MeasurementVector,
    MeasurementVectorElement,
)
from ali_processing.legacy.retrieval.measvec.transformer import (
    AltitudeNormalization,
    FrameRatio,
    FrameSelect,
    LinearCombination,
    LinearCombinations,
    LogRadiance,
    RowAverage,
    Truncate,
    WavelengthSelect,
)
from ali_processing.legacy.retrieval.statevector import StateVectorAerosolProfile
from ali_processing.legacy.util.analysis import (
    decode_to_multiindex,
    encode_multiindex,
    resolution_from_averaging_kernel,
)
from ali_processing.legacy.util.atmospheres import (
    aerosol_cross_section,
    apriori_profile,
    atmosphere_to_xarray,
    particle_size,
    retrieval_atmo,
)
from ali_processing.legacy.util.config import Config
from ali_processing.legacy.util.rt_options import retrieval_rt_opts
from matplotlib import ticker
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import DeltaFunction, Gaussian, Rectangle
from skretrieval.core.radianceformat import RadianceSpectralImage
from skretrieval.legacy.core.sensor.spectrograph import Spectrograph
from skretrieval.retrieval.rodgers import Rodgers
from skretrieval.retrieval.statevector import StateVector

plt.style.use(Config.MATPLOTLIB_STYLE_FILE)


class ExtinctionRetrieval:
    """
    Perform an aerosol extinction retrieval (varying number density with an assumed lognormal size distribution).

    This class is a wrapper around the MeasurementVector, StateVector, InstrumentSimulator and Rodgers' classes
    to help setup and retrieve aerosol extinction.


    Parameters
    ----------
    sensors : List[ALISensorDualChannel]
        List of ALI sensors
    optical_geometry : List[OpticalGeometry]
        List of optical geometries. Should be the same length as `sensors`.
    measurement_l1 : List[RadianceSpectralImage]
        Measured data. Should be in the same format that `ImagingSimulator` will produce given
        the `sensors` and `optical_geometry`
    output_filename : str
        Name of the output netcdf4 file that retrieval diagnostics and results are written to. If none is provided
        then nothing is written to file.

    Attributes
    ----------
    max_iterations : int
        maximum number of iterations that the `Rodgers` retrieval will perform.
    vertical_samples : int
        Number of vertical samples in the simulated radiative transfer. Larger values will provide higher accuracy
        calculations at the expense of computation time.
    brdf : float
        Equivalent Lambertian reflectance used in the forward model.
    use_cloud_for_lower_bound : bool
        Whether to set the lower bound of the retrieval based on the detected cloud top altitude.
    dual_polarization : bool
        Compute both polarizations for each sensor passed to the retrieval.
    tikhonov_factor : np.ndarray
        Factor used to the scale the second order Tikhonov regularization as a function of altitude. Should be an
        array the same size as `tikhonov_altitude`.
    tikhonov_altitude : np.ndarray
        Altitudes [meters] corresponding to `tikhonov_factor`.
    couple_normalization_altitudes : bool
        Include the coupling from altitude normalization in the jacobian calculation.
    simulation_atmosphere : sk.Atmosphere
        If set, this atmosphere will be included in the output file.
    sun : np.ndarray
        3 element array defining the sun location. If unset this is calculated by sasktran using the time.
    cloud_vector_normalization_range : Tuple[float, float]
        Normalization altitudes used for the cloud top detection algorithm.
    """

    def __init__(
        self,
        sensors: list[ALISensorDualChannel],
        optical_geometry: list[OpticalGeometry],
        measurement_l1: list[RadianceSpectralImage],
        output_filename: str = None,
    ):
        self.sensors = sensors
        self.optical_geometry = optical_geometry
        self.measurement_l1 = measurement_l1
        self.output_filename: str = output_filename

        # Measurement vector settings
        self.use_cloud_for_lower_bound: bool = False
        self.couple_normalization_altitudes: bool = True
        self.cloud_vector_wavelength: float = 1250.0
        self.cloud_vector_normalization_range: tuple[float, float] = (20000.0, 25000.0)

        # state element settings
        self.tikhonov_factor: np.ndarray = np.array([50.0, 100.0, 100.0])
        self.tikhonov_altitude: np.ndarray = np.array([5000.0, 25000.0, 30000.0])
        self.lm_damping = 0.01

        # A priori settings
        self._apriori_latitude: float = None
        self._apriori_longitude: float = None
        self._apriori_mjd: float = None
        self.sun: np.ndarray = None

        # Simulation settings
        self.dual_polarization: bool = True
        self.vertical_samples: int = 300
        self.single_scatter: bool = False

        # Retrieval settings
        self.max_iterations: int = 5
        self.simulation_atmosphere: sk.Atmosphere = None
        self.retrieval_atmosphere: sk.Atmosphere = None
        self.retrieve_cloud_top = True
        self.cloud_top: float = None
        self.ticfire_cloud: bool = False
        self.brdf: float = 0.3

        # Internal attributes
        self._aerosol_vector_wavelength: list[float] = [750.0]
        self._lower_bound: float = 5000.0
        self._upper_bound: float = 35000.0
        self._normalization_altitudes: tuple[float, float] = (35000.0, 45000.0)
        self._altitude_resolution = 200
        self._altitudes: np.ndarray = np.arange(0.0, 45000.1, self._altitude_resolution)
        self._jacobian_altitudes: np.ndarray = None
        self._forward_model: ImagingSimulator = None
        self._retrieval: AerosolRetrieval = None
        self._aerosol_opt_prop: sk.MieAerosol = None

    @property
    def altitudes(self):
        """
        Profile altitudes [meters] used for Jacobians
        """
        return self._altitudes

    @property
    def lower_bound(self) -> float:
        """
        Minimum altitude of the retrieval [meters]. Altitudes below this will use a scaled value of the apriori.
        """
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, value: float):
        self._lower_bound = value

    @property
    def normalization_altitudes(self) -> tuple[float, float]:
        """
        Normalization range used for the measurement vector [meters].
        """
        return self._normalization_altitudes

    @normalization_altitudes.setter
    def normalization_altitudes(self, value: tuple[float, float]):
        self._normalization_altitudes = value
        if self.upper_bound > self._normalization_altitudes[0]:
            self.upper_bound = self._normalization_altitudes[0]

    @property
    def upper_bound(self) -> float:
        """
        Maximum altitude of the retrieval [meters]. Altitudes above this will use a scaled value of the apriori.
        """
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, value: float):
        self._upper_bound = value

    @property
    def altitude_resolution(self) -> float:
        """
        Vertical resolution of the retrieval [meters]
        """
        return self._altitude_resolution

    @altitude_resolution.setter
    def altitude_resolution(self, value: float):
        """
        Vertical resolution of the retrieval [meters].
        """
        self._altitude_resolution = value
        self._altitudes = np.arange(0, 45000.1, self._altitude_resolution)

    @property
    def output_file_mode(self):
        if os.path.isfile(self.output_filename):
            return "a"
        else:
            return "w"

    @property
    def aerosol_vector_wavelength(self) -> list[float]:
        """
        Wavelengths used for the aerosol measurement vectors. A measurement vector is constructed for each wavelength.
        """
        return self._aerosol_vector_wavelength

    @aerosol_vector_wavelength.setter
    def aerosol_vector_wavelength(self, value: float | list[float]):
        if not hasattr(value, "__len__"):
            value = [value]
        self._aerosol_vector_wavelength = value

    @property
    def latitude(self) -> float:
        """
        Mean latitude of the observer tangent point locations
        """
        geo = sk.Geodetic()
        lats = []
        for opt in self.optical_geometry:
            geo.from_tangent_point(opt.observer, opt.look_vector)
            lats.append(geo.latitude)
        return float(np.mean(lats))

    @property
    def longitude(self) -> float:
        """
        Mean longitude of the observer tangent point locations
        """
        geo = sk.Geodetic()
        lons = []
        for opt in self.optical_geometry:
            geo.from_tangent_point(opt.observer, opt.look_vector)
            lons.append(geo.longitude)
        return float(np.mean(lons))

    @property
    def apriori_longitude(self) -> float:
        """
        Longitude used for the a priori profile selection
        """
        if self._apriori_longitude:
            return self._apriori_longitude
        return self.longitude

    @property
    def apriori_latitude(self) -> float:
        """
        Latitude used for the a priori profile selection
        """
        if self._apriori_latitude:
            return self._apriori_latitude
        return self.latitude

    @property
    def apriori_mjd(self) -> float:
        """
        Date in modified julian format used for the a priori profile selection
        """
        if self._apriori_mjd:
            return self._apriori_mjd
        return self.mjd

    @property
    def mjd(self) -> float:
        """
        Mean date of the observer tangent point locations in modified julian date (days since Nov 17, 1858)
        """
        return float(np.mean([o.mjd for o in self.optical_geometry]))

    @property
    def time(self) -> np.datetime64:
        """
        Mean time of the observer tangent point locations.
        """
        seconds_per_day = np.timedelta64(1, "D") / np.timedelta64(1, "us")
        seconds_since = np.timedelta64(int(self.mjd * seconds_per_day), "us")
        return np.datetime64("1858-11-17") + seconds_since

    @property
    def aerosol_measurement_vector(self) -> MeasurementVector:
        r"""
        Measurement vector used for aerosol retrieval

        .. math::

           I = I_{\parallel} + I_{\perp}

        .. math::
           y = \log(I) - \log(I_{norm})

        """
        return self._measurement_vector(self.aerosol_vector_wavelength)

    def _measurement_vector(self, wavelengths):
        aerosol_mvs = []

        l1_wavels = np.array([l1.data.wavelength.values for l1 in self.measurement_l1])
        unique_wavels = np.sort(np.unique(l1_wavels))

        for vec_wavel in wavelengths:
            aerosol_mv = MeasurementVectorElement()
            aerosol_mv.add_transform(RowAverage(dim="nx"))
            wavel = unique_wavels[np.argmin(np.abs(unique_wavels - vec_wavel))]
            wavel_idx = np.argwhere(l1_wavels == wavel).flatten()

            aerosol_mv.add_transform(
                LinearCombination({wavel_idx[0]: 1, wavel_idx[1]: 1})
            )
            aerosol_mv.add_transform(
                AltitudeNormalization(
                    norm_alts=self.normalization_altitudes,
                    couple_altitudes=self.couple_normalization_altitudes,
                )
            )
            aerosol_mv.add_transform(LogRadiance())
            aerosol_mv.add_transform(
                Truncate(lower_bound=self.lower_bound, upper_bound=self.upper_bound)
            )
            aerosol_mvs.append(aerosol_mv)

        return MeasurementVector(aerosol_mvs)

    @property
    def cloud_measurement_vector(self) -> MeasurementVector:
        r"""
        Measurement vector used for cloud top retrieval.

        .. math::

           y = \frac{Q}{I}

        Returns
        -------
        cloud_mv : MeasurementVector
            Cloud measurement vector

        """
        l1_wavels = np.array([l1.data.wavelength.values for l1 in self.measurement_l1])
        unique_wavels = np.sort(np.unique(l1_wavels))
        wavel = unique_wavels[
            np.argmin(np.abs(unique_wavels - self.cloud_vector_wavelength))
        ]
        wavel_idx = np.argwhere(l1_wavels == wavel).flatten()

        cloud_mv = MeasurementVectorElement()
        cloud_mv.add_transform(RowAverage(dim="nx"))
        cloud_mv.add_transform(
            LinearCombinations(
                [
                    {wavel_idx[0]: 1, wavel_idx[1]: -1},
                    {wavel_idx[0]: 1, wavel_idx[1]: 1},
                ]
            )
        )
        cloud_mv.add_transform(FrameRatio(index_1=0, index_2=1))
        return MeasurementVector([cloud_mv])

    def find_cloud_top_altitude(self) -> float:
        """
        Find the cloud top altitude by looking for a change in the polarization ratio (Q/I).

        Returns
        -------
        cloud_top : float
            Cloud top altitude [meters]

        """
        depol_y = self.cloud_measurement_vector.meas_dict(self.measurement_l1)
        tanalts, idx = self.tanalts_from_mv(
            self.cloud_measurement_vector, self.measurement_l1
        )

        min_alt = self.cloud_vector_normalization_range[0]
        max_alt = self.cloud_vector_normalization_range[1]
        cloud_top = 0.0
        for i in range(1):
            good = (tanalts > min_alt) & (tanalts < max_alt)
            below_norm = tanalts < min_alt

            depol_baseline = np.mean(depol_y["y"][good])
            baseline_std = np.std(depol_y["y"][good])
            depol_max = np.percentile(depol_y["y"][below_norm], 95)
            depol_min = np.percentile(depol_y["y"][below_norm], 5)
            # mid_point = (depol_max + depol_baseline) / 2
            mid_point = (depol_max + depol_min) / 2
            if np.abs(mid_point - depol_baseline) > (baseline_std * 2):
                incloud = (depol_y["y"] < mid_point) & (
                    tanalts < self.cloud_vector_normalization_range[0]
                )
                cloud_top = float(tanalts[np.where(incloud)[0][-1]])
                min_alt = cloud_top + self.altitude_resolution
            else:
                cloud_top = 0.0
                break
        return cloud_top

    def generate_retrieval_atmosphere(self) -> sk.Atmosphere:
        """
        Create the base atmosphere used in the retrieval. State Elements (e.g. 'aerosol') must still be added using the
        elements `add_to_atmosphere` method.
        """
        cloud_top = None
        cloud_optical_depth = None
        cloud_effective_diameter = None
        cloud_width = None
        clouds = False

        if self.ticfire_cloud:
            folder = r"C:\Users\lar555\PycharmProjects\ACCP\synergy\retrievals\GEM"
            file = os.path.join(
                folder, "GEM_201505160650_tropic_profile_data_TICFIREsimret.nc"
            )
            ticfire_data = xr.open_dataset(file, decode_times=False).isel(latitude=0)
            cloud_top = (
                17000.0  # float(ticfire_data['cloud_top_height'].values) * 1000.0
            )
            cloud_optical_depth = (
                float(ticfire_data["cloud_optical_depth"].values) * 0.5
            )
            cloud_effective_diameter = float(ticfire_data["effective_diameter"].values)
            cloud_width = 1000.0
            clouds = True

        atmo_ret = retrieval_atmo(
            self.apriori_latitude,
            self.apriori_mjd,
            self.altitudes,
            clouds=clouds,
            cloud_top=cloud_top,
            cloud_optical_depth=cloud_optical_depth,
            cloud_effective_diameter=cloud_effective_diameter,
            cloud_width=cloud_width,
        )
        atmo_ret.brdf = self.brdf
        return atmo_ret

    def retrieval_sensors(self) -> list[ALISensorDualChannel]:
        """
        ALI sensor models used for the retrieval.
        """
        ret_sensors = []
        for ali in self.sensors:
            ali.add_dark_current = False
            ali.add_noise = False
            ali.add_adc = False
            ali.straylight = 0.0
            # ali.turn_rotator_on()
            ret_sensors.append(ali)

        return ret_sensors

    def apriori_profile(self):
        """
        A priori number density profile used in the retrieval
        """
        apriori_altitudes = np.arange(0, 45001.0, self.altitude_resolution)
        values = apriori_profile(
            self.apriori_latitude, self.apriori_mjd, apriori_altitudes
        )
        decay_factor = -4
        scale_height_m = 7000
        upper_values = np.exp(decay_factor * apriori_altitudes / scale_height_m)
        above_ub = apriori_altitudes > self.upper_bound
        pinned_alt = np.where(above_ub)[0][0] - 1
        scale = values[pinned_alt] / upper_values[pinned_alt]
        values[apriori_altitudes > self.upper_bound] = (
            upper_values[apriori_altitudes > self.upper_bound] * scale
        )
        return values

    def aerosol_state_element(self):
        """
        Aerosol number density state vector element.
        """
        if self._aerosol_opt_prop is None:
            ps_clim = sk.ClimatologyUserDefined(
                self.altitudes, particle_size(self.altitudes)
            )
            self._aerosol_opt_prop = sk.MieAerosol(
                particlesize_climatology=ps_clim, species="H2SO4"
            )

        aerosol_state_element = StateVectorAerosolProfile(
            altitudes_m=self.altitudes,
            values=np.log(self.apriori_profile()),
            species_name="aerosol",
            optical_property=self._aerosol_opt_prop,
            lowerbound=self.lower_bound,
            upperbound=self.upper_bound,
            second_order_tikhonov_factor=self.tikhonov_factor,
            second_order_tikhonov_altitude=self.tikhonov_altitude,
        )

        return aerosol_state_element

    def forward_model(self, retrieval):
        """
        Generate the forward model used in the retrieval
        """
        ret_opts = retrieval_rt_opts(
            retrieval,
            configure_for_cloud=False,
            cloud_lower_bound=0.0,
            cloud_upper_bound=18000.0,
            single_scatter=self.single_scatter,
        )
        forward_model = ImagingSimulator(
            sensors=self.retrieval_sensors(),
            optical_axis=self.optical_geometry,
            atmosphere=self.retrieval_atmosphere,
            options=ret_opts,
        )
        forward_model.num_vertical_samples = self.vertical_samples
        forward_model.store_radiance = False
        forward_model.grid_sampling = True
        forward_model.group_scans = False
        forward_model.dual_polarization = self.dual_polarization
        forward_model.sun = self.sun
        return forward_model

    def retrieve(self):
        """
        Perform the aerosol extinction retrieval.
        """

        if self.retrieve_cloud_top:
            self.cloud_top = self.find_cloud_top_altitude()
        else:
            self.cloud_top = 0.0

        if self.use_cloud_for_lower_bound and (self.cloud_top > self.lower_bound):
            self.lower_bound = self.cloud_top

        self.retrieval_atmosphere = self.generate_retrieval_atmosphere()

        aerosol_element = self.aerosol_state_element()
        aerosol_element.add_to_atmosphere(self.retrieval_atmosphere)
        self._retrieval = AerosolRetrieval(
            state_vector=StateVector([aerosol_element]),
            measurement_vector=self.aerosol_measurement_vector,
            retrieval_altitudes=self.altitudes,
        )
        self._retrieval.save_output = True

        self._forward_model = self.forward_model(self._retrieval)
        self._jacobian_altitudes = self._retrieval.jacobian_altitudes
        if self.sun is not None:
            self._forward_model.sun = self.sun

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
                self.aerosol_measurement_vector,
            )
        return output

    def save_simulation_atmosphere(self):
        sim_ds = atmosphere_to_xarray(
            self.simulation_atmosphere,
            altitudes=self._retrieval.jacobian_altitudes,
            latitude=self.latitude,
            longitude=self.longitude,
            mjd=self.mjd,
        )
        sim_ds["aerosol"].to_netcdf(
            self.output_filename, group="true/aerosol", mode=self.output_file_mode
        )
        if "icecloud" in self.simulation_atmosphere.species.keys():
            sim_ds["icecloud"].to_netcdf(
                self.output_filename, group="true/ice_cloud", mode=self.output_file_mode
            )
        if "water" in self.simulation_atmosphere.species.keys():
            sim_ds["water"].to_netcdf(
                self.output_filename,
                group="true/water_cloud",
                mode=self.output_file_mode,
            )
        sim_ds["brdf"].to_netcdf(
            self.output_filename, group="true/BRDF", mode=self.output_file_mode
        )

    def save_retrieval_atmosphere(self, atmo_ret):
        ret_ds = atmosphere_to_xarray(
            atmo_ret, altitudes=self._retrieval.jacobian_altitudes
        )
        ret_ds["aerosol"].to_netcdf(
            self.output_filename, group="retrieved/aerosol", mode=self.output_file_mode
        )
        ctop = xr.Dataset({"cloud_top_altitude": self.cloud_top})
        ctop.attrs = {"standard_name": "cloud_top_altitude", "units": "km"}
        ctop.to_netcdf(
            self.output_filename, group="retrieved/cloud", mode=self.output_file_mode
        )
        ret_ds["brdf"].to_netcdf(
            self.output_filename, group="retrieved/BRDF", mode=self.output_file_mode
        )

    def save_optical_geometry(self):
        optgeom = xr.Dataset(
            {
                "observer": (
                    ["sensor", "xyz"],
                    np.array([o.observer for o in self.optical_geometry]),
                ),
                "look_vector": (
                    ["sensor", "xyz"],
                    np.array([o.look_vector for o in self.optical_geometry]),
                ),
                "local_up": (
                    ["sensor", "xyz"],
                    np.array([o.local_up for o in self.optical_geometry]),
                ),
            },
            coords={
                "time": [
                    pd.Timestamp("1858-11-18") + pd.Timedelta(o.mjd, "D")
                    for o in self.optical_geometry
                ],
                "sensor": np.arange(0, len(self.sensors)),
                "xyz": np.array(["x", "y", "z"]),
            },
        )

        rp = np.array(self._forward_model.engine_diagnostics["referencepoint"])
        if len(rp) > 0:
            mjd = rp[:, -1]
            lat = rp[:, 0]
            lon = rp[:, 1]
            alt = rp[:, 2]
            sun = np.array(self._forward_model.engine_diagnostics["sun"])
            engine_diag = xr.Dataset(
                {
                    "sun": (["sensor", "xyz"], sun),
                    "rp_latitude": (["sensor"], lat),
                    "rp_longitude": (["sensor"], lon),
                    "rp_altitude": (["sensor"], alt),
                    "rp_mjd": (["sensor"], mjd),
                },
                coords={
                    "sensor": np.arange(0, len(self.sensors)),
                    "xyz": np.array(["x", "y", "z"]),
                },
            )
            optgeom = xr.merge([optgeom, engine_diag])

        optgeom.to_netcdf(
            self.output_filename, group="optical_geometry", mode=self.output_file_mode
        )

    def save_level1_measurements(self):
        for idx, l1 in enumerate(self.measurement_l1):
            tp = l1.tangent_locations()
            data = l1.data
            xr.merge([data, tp]).to_netcdf(
                self.output_filename,
                group=f"level1/measurement{idx}",
                mode=self.output_file_mode,
            )

    def tanalts_from_mv(
        self, measurement_vector: MeasurementVector, level1: list[RadianceSpectralImage]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the tangent altitudes of the measurement vector elements.

        Parameters
        ----------
        measurement_vector : MeasurementVector
        level1 : List[RadianceSpectralImage]

        Returns
        -------
        tangent_altitude : np.ndarray
            Tangent altitudes of the measurement vector elements
        indexes : np.ndarray
            Index of the measurement vector elements
        """
        tas = []
        idxs = []
        for idx, element in enumerate(measurement_vector.elements):
            l1t = element.transform(level1)
            tanalts = l1t.tangent_locations().altitude
            tas.append(tanalts)
            idxs.append(np.ones(len(tanalts), dtype=int) * (idx + 1))
        tas = np.array(tas).flatten()
        idxs = np.array(idxs).flatten()
        return tas, idxs

    def get_output_state(self, output):
        Ns = 0
        states = []
        names = []
        altitudes = []
        aprioris = []
        for idx, element in enumerate(self._retrieval._state_vector.state_elements):
            good_alts = element.retrieval_alts()
            ret_alts = self._retrieval._retrieval_altitudes[good_alts]
            N = len(ret_alts)
            altitudes.append(ret_alts)
            aprioris.append(element.initial_state)
            xs = np.array(output["xs"])[:, Ns : (N + Ns)]
            names.append([element.name()] * N)
            states.append(xs)
            Ns += N

        index = pd.MultiIndex.from_arrays(
            [np.concatenate(altitudes), np.concatenate(names)],
            names=["ret_alt", "ret_state"],
        )
        state = xr.DataArray(
            np.concatenate(states, axis=1),
            name="target_state",
            dims=["iteration", "ret_idx"],
            coords=[np.arange(0, len(output["xs"])), index],
        )
        initial_state = xr.DataArray(
            np.concatenate(aprioris),
            name="initial_state",
            dims=["ret_idx"],
            coords=[index],
        )
        return xr.merge([state, initial_state])

    def save_retrieval_state_info(self, output, rodgers, retrieval, meas_vec):
        state = self.get_output_state(output)
        ret_alts = state.ret_idx.indexes["ret_idx"]
        pert_alts = ret_alts.set_names(["pert_alt", "pert_state"])
        averaging_kernel = xr.DataArray(
            output["averaging_kernel"],
            dims=["pert_idx", "ret_idx"],
            coords=[pert_alts, ret_alts],
            name="averaging_kernel",
        )
        y_meas_dict, y_meas, Sy, inv_Sy, good_meas = rodgers._measurement_parameters(
            retrieval, self.measurement_l1
        )

        tas, idxs = self.tanalts_from_mv(meas_vec, self.measurement_l1)
        multiidx = pd.MultiIndex.from_arrays([tas, idxs], names=["altitude", "vector"])
        good_meas_alts = tas[good_meas]
        good_meas_idx = pd.MultiIndex.from_arrays(
            [good_meas_alts, idxs[good_meas]], names=["altitude", "vector"]
        )

        gain_matrix = xr.DataArray(
            output["gain_matrix"],
            dims=["ret_idx", "measurement_idx"],
            coords=[ret_alts, good_meas_idx],
            name="gain_matrix",
        )
        ys = xr.DataArray(
            np.array(output["ys"]),
            dims=["iteration", "index"],
            coords=[np.arange(0, len(output["ys"])), multiidx],
            name="F",
        )

        y = xr.DataArray(y_meas_dict["y"], dims=["index"], coords=[multiidx], name="y")
        try:
            ye = xr.DataArray(
                y_meas_dict["y_error"],
                dims=["index"],
                coords=[multiidx],
                name="y_error",
            )
        except KeyError:
            ye = (y * 0).rename("y_error")
        depol_y = self.cloud_measurement_vector.meas_dict(self.measurement_l1)
        tas, idxs = self.tanalts_from_mv(
            self.cloud_measurement_vector, self.measurement_l1
        )
        cloud_multiidx = pd.MultiIndex.from_arrays(
            [tas, idxs], names=["cloud_altitude", "cloud_vector"]
        )
        if self.retrieve_cloud_top:
            yc = xr.DataArray(
                depol_y["y"],
                dims=["cloud_index"],
                coords=[cloud_multiidx],
                name="y_cloud",
            )
            try:
                yce = xr.DataArray(
                    depol_y["y_error"],
                    dims=["cloud_index"],
                    coords=[cloud_multiidx],
                    name="y_cloud_error",
                )
            except KeyError:
                yce = (yc * 0).rename("y_cloud_error")

        Sx = xr.DataArray(
            output["solution_covariance"],
            dims=["pert_idx", "ret_idx"],
            coords=[pert_alts, ret_alts],
            name="solution_covariance",
        )
        Se = xr.DataArray(
            output["error_covariance_from_noise"],
            dims=["pert_idx", "ret_idx"],
            coords=[pert_alts, ret_alts],
            name="error_covariance_from_noise",
        )

        forward_l1 = self._forward_model.calculate_radiance()
        y_ret_dict = retrieval.measurement_vector(forward_l1)
        jacobian_matrix = xr.DataArray(
            y_ret_dict["jacobian"][good_meas],
            dims=["measurement_idx", "ret_idx"],
            coords=[good_meas_idx, ret_alts],
            name="jacobian_matrix",
        )

        retrieval_info = xr.merge(
            [averaging_kernel, gain_matrix, Sx, Se, state, jacobian_matrix]
        )
        if self.retrieve_cloud_top:
            meas_vec_info = xr.merge([ys, y, ye, yc, yce])
        else:
            meas_vec_info = xr.merge([ys, y, ye])

        retrieval_encoded = encode_multiindex(retrieval_info, "ret_idx")
        retrieval_encoded = encode_multiindex(retrieval_encoded, "measurement_idx")
        retrieval_encoded = encode_multiindex(retrieval_encoded, "pert_idx")
        retrieval_encoded.to_netcdf(
            self.output_filename, group="retrieval/state", mode="a"
        )

        if self.retrieve_cloud_top:
            meas_vec_encoded = encode_multiindex(meas_vec_info, "cloud_index")
            meas_vec_encoded = encode_multiindex(meas_vec_encoded, "index")
        else:
            meas_vec_encoded = encode_multiindex(meas_vec_info, "index")
        meas_vec_encoded.to_netcdf(
            self.output_filename, group="retrieval/vectors", mode="a"
        )

    def save_output(self, rodgers, output, atmo_ret, retrieval, meas_vec):
        if self.simulation_atmosphere is not None:
            self.save_simulation_atmosphere()
        self.save_retrieval_atmosphere(atmo_ret)
        self.save_optical_geometry()
        self.save_level1_measurements()
        self.save_retrieval_state_info(output, rodgers, retrieval, meas_vec)

    @staticmethod
    def plot_results(
        output_filename,
        extinction_wavelength: float = 750.0,
        log_state: bool = True,
        plot_error: bool = True,
        plot_meas_vec: bool = True,
        plot_averaging_kernel: bool = True,
        aerosol_scale: int | float = 1000,
        plot_cloud: bool = True,
        figize: tuple[float, float] = None,
        kernel_kwargs: dict = {},
    ):
        """
        Plot the retrieved extinction and measurement vectors.

        Parameters
        ----------
        output_filename
            Retrieval output filename.
        extinction_wavelength
            Wavelength in nanometers to plot extinction at. Default = 750.0 nm.
        log_state
            Whether the statevector is in log space. Default True.
        plot_error
            Whether to plot the error bars on the retrieved parameters.
        plot_meas_vec
            Whether to plot the measurement vectors. Default True
        plot_cloud
            Whether to plot the cloud profile, if available. Default True
        plot_averaging_kernel
            Whether to plot the averaging kernel. Default True
        aerosol_scale
            Scale the aerosol values before plotting. Default 1000.
        figsize
            Size of the figure (width, height).
        kernel_kwargs
            Optional arguments provided to the `plot_averaging_kernel`. Has no effect if `plot_averaging_kernel=False`

        Returns
        -------
        fig : plt.Figure
            Figure used for plotting
        ax : plt.Axes or array of plt.Axes
            Axes used for plotting
        """

        if isinstance(extinction_wavelength, (float, int)):
            extinction_wavelength = [extinction_wavelength]

        try:
            true_state = xr.open_dataset(output_filename, group="true/aerosol")
        except (KeyError, OSError) as e:
            true_state = None

        try:
            ice_cloud = xr.open_dataset(output_filename, group="true/ice_cloud")
        except (KeyError, OSError) as e:
            ice_cloud = None

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

        aer_xsec = aerosol_cross_section(
            extinction_wavelength,
            rg=ret_data.lognormal_median_radius.values,
            sg=ret_data.lognormal_width.values,
        )
        aer_xsec = xr.DataArray(
            aer_xsec, dims=["ret_alt"], coords=[ret_data.altitude.values]
        )

        aerosol_iterations = ret_info.target_state.sel(ret_state="aerosol")
        aerosol_initial = ret_info.initial_state.sel(ret_state="aerosol")
        if log_state:
            aerosol_iterations = np.exp(aerosol_iterations)
        aerosol_iterations *= aer_xsec.interp(ret_alt=aerosol_iterations.ret_alt.values)
        aerosol_initial *= aer_xsec.interp(ret_alt=aerosol_iterations.ret_alt.values)
        # aerosol_final = ret_data.extinction.sel(wavelength=extinction_wavelength)

        true_color = "#c2452f"
        ret_color = "#2177b5"
        num_plots = 1
        x = 2.6
        if plot_meas_vec:
            num_plots += 1
            x += 1.8
        if plot_averaging_kernel:
            num_plots += 1
            x += 1.8
        if figize is None:
            figsize = (x, 4)
        fig, ax = plt.subplots(1, num_plots, figsize=figsize, dpi=200, sharey=True)
        fig.subplots_adjust(
            left=0.1 * 4 / x, bottom=0.12, right=0.97, top=0.9, wspace=0.05
        )

        ax[0].set_ylabel("Altitude [km]")
        ax[0].set_xlabel("Extinction [$\\times10^{-3}$km$^{-1}$]")
        if plot_meas_vec:
            ax[1].set_xlabel("Measurement Vector")

        (l0,) = ax[0].plot(
            aerosol_initial * aerosol_scale,
            aerosol_initial.ret_alt / 1000,
            color=ret_color,
            ls="--",
            lw=1,
            zorder=10,
        )

        if true_state is not None:
            for wavel in extinction_wavelength:
                (l1,) = ax[0].plot(
                    true_state.extinction.sel(wavelength=wavel, method="nearest")
                    * aerosol_scale,
                    true_state.altitude / 1000,
                    color=true_color,
                    lw=1,
                    zorder=11,
                )
                if (ice_cloud is not None) and plot_cloud:
                    ax[0].fill_betweenx(
                        ice_cloud.altitude / 1000,
                        ice_cloud.extinction.sel(wavelength=wavel, method="nearest")
                        * aerosol_scale,
                        ice_cloud.extinction.sel(wavelength=wavel, method="nearest")
                        * 0,
                        color="#444444",
                        lw=0,
                        alpha=0.1,
                        zorder=0,
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
        for wavel in extinction_wavelength:
            aerosol_final = ret_data.extinction.sel(wavelength=wavel, method="nearest")
            (l3,) = ax[0].plot(
                aerosol_final.values * aerosol_scale,
                aerosol_final.altitude.values / 1000,
                color=ret_color,
                zorder=15,
            )

        if plot_error:
            Se = np.diag(
                ret_info.solution_covariance.sel(
                    ret_state="aerosol", pert_state="aerosol"
                ).values
            )
            # Se = np.diag(ret_info.error_covariance_from_noise.sel(ret_state='aerosol', pert_state='aerosol').values)
            Se = xr.DataArray(
                Se,
                dims=["ret_alt"],
                coords=[
                    ret_info.solution_covariance.sel(ret_state="aerosol").ret_alt.values
                ],
            )
            error = np.exp(np.sqrt(Se)) * aer_xsec.interp(ret_alt=Se.ret_alt.values)
            final = aerosol_iterations.isel(iteration=-1)
            l4 = ax[0].fill_betweenx(
                final.ret_alt.values / 1000,
                (final.values + error.values) * aerosol_scale,
                (final.values - error.values) * aerosol_scale,
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

        # leg = ax[0].legend([l0, l2, l3], ['A priori', 'Iterations', 'Retrieved'],
        #                    framealpha=1, facecolor=ax[0].get_facecolor(), edgecolor='none', fontsize='small')
        leg.set_title("Retrieval State", prop={"size": "small", "weight": "bold"})

        ax[0].axhline(
            ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444"
        )
        ax[0].set_ylim(0, 40)

        if plot_meas_vec:
            ExtinctionRetrieval.plot_measurement_vectors(output_filename, ax[1])

        if plot_averaging_kernel:
            ax2 = ExtinctionRetrieval.plot_averaging_kernel(
                output_filename, "aerosol", ax=ax[-1], **kernel_kwargs
            )
            ax[-1].set_xlabel("Averaging Kernels")
            ax = np.concatenate([ax, [ax2]])

        return fig, ax

    @staticmethod
    def plot_measurement_vectors(
        output_filename,
        ax: plt.Axes = None,
        plot_error: bool = True,
        true_color="#c2452f",
        ret_color="#2177b5",
    ):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        y = xr.open_dataset(output_filename, group="retrieval/vectors")
        y = decode_to_multiindex(y, "index")
        vectors = np.unique(y.index.vector.values)
        for vector in vectors:
            ax.plot(
                y.y.sel(vector=vector),
                y.sel(vector=vector).altitude.values / 1000,
                color=true_color,
                lw=1,
                zorder=2,
            )
            if plot_error:
                yi = y.y.sel(vector=vector)
                ye = np.sqrt(y.y_error.sel(vector=vector))
                ax.fill_betweenx(
                    y.sel(vector=vector).altitude.values / 1000,
                    yi + ye,
                    yi - ye,
                    color=true_color,
                    alpha=0.2,
                    lw=0,
                    zorder=1,
                )

        for iteration in y.iteration.values:
            for vector in vectors:
                ax.plot(
                    y.F.sel(iteration=iteration, vector=vector).values,
                    y.sel(vector=vector).altitude.values / 1000,
                    color=ret_color,
                    alpha=0.5,
                    lw=0.5,
                    zorder=3,
                )

        for vector in vectors:
            ax.plot(
                y.F.isel(iteration=-1).sel(vector=vector).values,
                y.sel(vector=vector).altitude.values / 1000,
                color=ret_color,
                lw=1,
                zorder=5,
            )

        return ax

    @staticmethod
    def plot_averaging_kernel(
        output_filename,
        state,
        ax: plt.Axes = None,
        ret_alts: np.ndarray = np.array([10, 15, 20, 25, 30, 35]),
        alpha_scale: float = 6.0,
        alpha_exponent: float = 4.0,
        add_labels: bool = True,
        label_position: str = "left",
        fwhm_axis: bool = True,
        fwhm_label: str = "Vertical\nresolution [km]",
        fwhm_label_alt: float = 28,
    ):
        ret_info = xr.open_dataset(output_filename, group="retrieval/state")
        ret_info = decode_to_multiindex(
            ret_info, ["ret_idx", "pert_idx", "measurement_idx"]
        )
        # ret_info = decode_to_multiindex(ret_info, 'pert_idx')
        # ret_info = decode_to_multiindex(ret_info, 'measurement_idx')

        averaging_kernel = ret_info.averaging_kernel.sel(
            ret_state=state, pert_state=state
        )
        alts = averaging_kernel.ret_alt.values / 1000
        fwhm = resolution_from_averaging_kernel(
            averaging_kernel.rename({"pert_alt": "altitude"})
        )
        fwhm = fwhm.rename({"altitude": "pert_alt"}) / 1000.0

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if fwhm_axis:
            ax2 = ax.twiny()
            ax2.set_xlabel("FWHM [km]")
            ax2.set_facecolor(ax.get_facecolor())
            ax.set_facecolor("none")
            ax2.patch.set_visible(True)
            ax2.set_zorder(5)
            ax.set_zorder(10)
        else:
            ax2 = ax
        ax2.plot(fwhm, alts, color="#666666", zorder=3, lw=0.5)

        bbox = dict(
            edgecolor="none", facecolor=ax.get_facecolor(), boxstyle="square,pad=0.1"
        )
        for aidx, alt in enumerate(ret_alts):
            color = plt.cm.turbo(aidx / len(ret_alts))
            state_alts = ret_info.sel(ret_state=state).ret_alt.values / 1000
            A = ret_info.averaging_kernel.sel(ret_state=state, pert_state=state).sel(
                pert_alt=alt * 1000, method="nearest"
            )
            for ridx, ralt in enumerate(A.ret_alt.values[:-1] / 1000):
                ax.plot(
                    A[ridx : ridx + 2],
                    state_alts[ridx : ridx + 2],
                    color=color,
                    alpha=np.clip(1 - (np.abs(ralt - alt) / alpha_scale), 0, 1)
                    ** alpha_exponent,
                    zorder=5,
                    solid_capstyle="butt",
                )
            if add_labels:
                if label_position == "right":
                    x = float(A.sel(ret_alt=alt * 1000, method="nearest").values)
                    shift = 0.01
                    ha = "left"
                else:
                    x = float(A.min().values)
                    ha = "right"
                    shift = -0.01
                y = (
                    float(A.sel(ret_alt=alt * 1000, method="nearest").ret_alt.values)
                    / 1000
                )
                ax.text(
                    x + shift,
                    y,
                    f"{alt} km",
                    fontsize="small",
                    fontweight="bold",
                    color=color,
                    ha=ha,
                    va="center",
                    bbox=bbox,
                )

        if fwhm_label:
            x = float(fwhm.sel(pert_alt=fwhm_label_alt * 1000, method="nearest").values)
            ax2.text(
                x + 0.1,
                fwhm_label_alt,
                fwhm_label,
                fontsize="small",
                fontweight="bold",
                color="#666666",
                ha="left",
                va="center",
                bbox=bbox,
            )

        if fwhm_axis:
            l = ax.get_xlim()
            l2 = ax2.get_xlim()
            f = lambda x: l2[0] + (x - l[0]) / (l[1] - l[0]) * (l2[1] - l2[0])
            ticks = f(ax.get_xticks())
            ax2.xaxis.set_major_locator(ticker.FixedLocator(ticks))

        return ax2

    @staticmethod
    def plot_jacobian(
        output_filename,
        state,
        ax: plt.Axes = None,
        vectors: list | int = None,
        measurement_alts: np.ndarray = np.array([10, 15, 20, 25, 30, 35]),
    ):
        ret_info = xr.open_dataset(output_filename, group="retrieval/state")
        ret_info = decode_to_multiindex(ret_info, "ret_idx")
        ret_info = decode_to_multiindex(ret_info, "pert_idx")
        ret_info = decode_to_multiindex(ret_info, "measurement_idx")

        jacobian = ret_info.jacobian_matrix.sel(ret_state=state)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        if vectors is not None:
            if type(vectors) != list:
                vectors = [vectors]
        else:
            vectors = np.unique(ret_info.measurement_idx.vector)

        for aidx, alt in enumerate(measurement_alts):
            color = plt.cm.turbo(aidx / len(measurement_alts))
            for vector in vectors:
                K = jacobian.sel(vector=vector).sel(altitude=alt, method="nearest")
                ax.plot(K, K.ret_alt / 1000, color=color)

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
        aerosol_scale: int | float = 1000,
        fig: plt.Figure = None,
        axs: list[plt.axes] = None,
        plot_cloud: bool = True,
        kernel_kwargs: dict = {},
        figsize=(5, 4),
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
        aerosol_iterations = np.exp(aerosol_iterations)

        xsec = aerosol_cross_section(extinction_wavelength, rg=0.08, sg=1.6)
        aerosol_iterations *= xsec
        aerosol_final = ret_data.extinction.sel(wavelength=extinction_wavelength)

        true_color = "#c2452f"
        ret_color = "#2177b5"

        if axs is None:
            fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=200, sharey=True)
            fig.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.95, wspace=0.05)

        ax = axs[0]
        ax2 = axs[1]
        ax3 = axs[2]

        ax.set_ylabel("Altitude [km]")
        ax.set_xlabel("Extinction [$\\times10^{-3} $km$^{-1}$]")
        ax2.set_xlabel("Extinction Error [$\\times10^{-3} $km$^{-1}$]")
        ax3.set_xlabel("Extinction Error [%]")

        (l0,) = ax.plot(
            aerosol_iterations.sel(iteration=0) * aerosol_scale,
            aerosol_iterations.ret_alt / 1000,
            color=ret_color,
            ls="--",
            lw=1,
            zorder=10,
        )

        if true_state is not None:
            (l1,) = ax.plot(
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
            ax2.plot(
                abs_error, error.altitude / 1000, color=true_color, lw=1, zorder=11
            )
            ax3.plot(error, error.altitude / 1000, color=true_color, lw=1, zorder=11)

        for i in range(1, len(ret_info.iteration)):
            (l2,) = ax.plot(
                aerosol_iterations.sel(iteration=i).values * aerosol_scale,
                aerosol_iterations.ret_alt / 1000,
                color=ret_color,
                alpha=0.2,
                lw=0.5,
                zorder=9,
            )
        (l3,) = ax.plot(
            aerosol_final.values * aerosol_scale,
            aerosol_final.altitude.values / 1000,
            color=ret_color,
            zorder=15,
        )

        if true_state:
            leg = ax.legend(
                [l0, l2, l3, l1],
                ["A priori", "Iterations", "Retrieved", "True"],
                loc="upper right",
                framealpha=1,
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                fontsize="small",
            )
        else:
            leg = ax.legend(
                [l0, l2, l3],
                ["A priori", "Iterations", "Retrieved"],
                loc="upper right",
                framealpha=1,
                facecolor=ax.get_facecolor(),
                edgecolor="none",
                fontsize="small",
            )

        leg.set_title("Retrieval State", prop={"size": "small", "weight": "bold"})

        ax.axhline(ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444")
        ax.axhline(ret_data.cloud_top_altitude / 1000, lw=0.5, ls="--", color="#444444")
        ax.set_ylim(0, 40)

        ax3.set_xlim(-50, 50)
        ax3.set_xlim(-50, 50)
        ax.set_xlim(1e-6 * aerosol_scale, 1e-2 * aerosol_scale)
        ax.set_xscale("log")
        ax.set_xticks(10 ** np.arange(-6, -1.9) * aerosol_scale)
        return fig, ax


class OMPSExtinctionRetrieval(ExtinctionRetrieval):
    def __init__(
        self,
        sensors,
        optical_geometry: list[OpticalGeometry],
        measurement_l1: list[RadianceSpectralImage],
        output_filename: str = None,
        latitude_25km: float = 0.0,
        orbit: int = 53544,
    ):
        super().__init__(sensors, optical_geometry, measurement_l1, output_filename)
        self.orbit = orbit
        self.latitude_25km = latitude_25km
        self.orbit = orbit
        self.hires_geometry = None

    def retrieval_sensors(self) -> list[Spectrograph]:
        """
        ALI sensor models used for the retrieval.
        """
        sensors = [
            Spectrograph(
                wavelength_nm=[745.0],
                pixel_shape=DeltaFunction(),
                vert_fov=Rectangle(0.0003),
                horiz_fov=DeltaFunction(),
            )
            for geom in self.optical_geometry
        ]
        return sensors

    def forward_model(self, retrieval):
        """
        Generate the forward model used in the retrieval
        """
        ret_opts = retrieval_rt_opts(
            retrieval,
            configure_for_cloud=False,
            cloud_lower_bound=0.0,
            cloud_upper_bound=18000.0,
            single_scatter=self.single_scatter,
        )
        forward_model = OMPSImagingSimulator(
            sensors=self.retrieval_sensors(),
            optical_axis=self.optical_geometry,
            atmosphere=self.retrieval_atmosphere,
            options=ret_opts,
        )
        forward_model.hires_geometry = self.hires_geometry
        # forward_model.num_vertical_samples = self.vertical_samples
        # forward_model.store_radiance = False
        # forward_model.grid_sampling = True
        # forward_model.group_scans = False
        # forward_model.dual_polarization = self.dual_polarization
        # forward_model.sun = self.sun
        return forward_model

    def _measurement_vector(self, wavelengths):
        aerosol_mvs = []

        for vec_wavel in wavelengths:
            aerosol_mv = MeasurementVectorElement()
            aerosol_mv.add_transform(RowAverage(dim="nx"))
            aerosol_mv.add_transform(FrameSelect(0))
            aerosol_mv.add_transform(WavelengthSelect(vec_wavel))
            aerosol_mv.add_transform(
                AltitudeNormalization(
                    norm_alts=self.normalization_altitudes,
                    couple_altitudes=self.couple_normalization_altitudes,
                )
            )
            aerosol_mv.add_transform(LogRadiance())
            aerosol_mv.add_transform(
                Truncate(lower_bound=self.lower_bound, upper_bound=self.upper_bound)
            )
            aerosol_mvs.append(aerosol_mv)

        return MeasurementVector(aerosol_mvs)

    def generate_retrieval_atmosphere(self) -> sk.Atmosphere:
        atmo_file = os.path.join(
            r"C:\Users\lar555\data\omps\nasa\L1", f"o{self.orbit}.h5"
        )
        atmo_data = xr.open_dataset(atmo_file, group="Data")
        atmo_data_geo = xr.open_dataset(atmo_file, group="Geo")
        lat_idx = np.argmin(np.abs(atmo_data_geo.Latitude.values - self.latitude_25km))
        atmo_data_unbounded = xr.open_dataset(atmo_file, group="DataUnbounded")
        atmo = sk.Atmosphere()
        # atmo['air'] = sk.Species(sk.Rayleigh(), sk.MSIS90())
        background = sk.ClimatologyUserDefined(
            altitudes=np.arange(500, 99999.1, 1000.0),
            values={
                "SKCLIMATOLOGY_PRESSURE_PA": atmo_data.BackgroundPressure[:, lat_idx],
                "SKCLIMATOLOGY_AIRNUMBERDENSITY_CM3": atmo_data.BackgroundNumDen[
                    :, lat_idx
                ],
                "SKCLIMATOLOGY_TEMPERATURE_K": atmo_data.BackgroundTemperature[
                    :, lat_idx
                ],
            },
        )
        ozone_cm3 = atmo_data_unbounded.OzoneNumDen[:, lat_idx]
        ozone_cm3[np.isnan(ozone_cm3)] = 0.0
        ozone = sk.ClimatologyUserDefined(
            altitudes=np.arange(500, 99999.1, 1000.0),
            values={"SKCLIMATOLOGY_O3_CM3": ozone_cm3},
        )
        atmo["air"] = sk.Species(sk.Rayleigh(), background)
        atmo["ozone"] = sk.Species(sk.O3DBM(), ozone)
        atmo.brdf = float(atmo_data.Albedo[lat_idx])

        return atmo

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from sasktran import Geometry, LineOfSight
from skcomponents.optics import AOTF, CompositeComponent, IdealLCR, LinearPolarizer
from skcomponents.processing import PhotonIntegration
from skretrieval.core.lineshape import (
    Gaussian,
    LineShape,
    Rectangle,
)
from skretrieval.core.sensor import OpticalGeometry
from skretrieval.util import rotation_matrix

from ali_processing.legacy.instrument.aotf import (
    ER2AOTF,
    Brimrose20mmAOTF,
    BrimroseAOTFSingleChannel,
)
from ali_processing.legacy.instrument.lcr import ArcOptixLCR
from ali_processing.legacy.instrument.lenses import (
    GoldMirror,
    Mirror12AOI,
)
from ali_processing.legacy.instrument.lineshape import ALIER2LineShape, ALILineShape
from ali_processing.legacy.instrument.polarizer import MeadowlarkOWLPolarizer
from ali_processing.legacy.instrument.sensors import (
    Cardinal640,
    Cardinal1280,
    RaptorOWL640N,
    Sensor,
    TeledyneCCD97,
)
from ali_processing.legacy.instrument.spectralimager import (
    ALISpectralImage,
    SpectralImagerFast,
)


class ALISensor(SpectralImagerFast):
    """
    Parameters
    ----------
    wavelength_nm: np.ndarray
        Array of wavelengths that will be measuremed. The input radiances should set to accurately samples these values.
    pixel_vert_fov: LineShape
        Vertical lineshape of the spatial point spread function.
    pixel_horiz_fov: LineShape
        Horizontal lineshape of the spatial point spread function.
    image_horiz_fov: float
        Horizontal field of view of the full image in degrees.
    image_vert_fov: float
        Vertical field of view of the full image in degrees.
    num_columns: int
        Number of columns in the sensor.
    num_rows: int
        Number of rows in the sensor
    ideal_optics: bool
        If True, optics are assumed to be lossless and LCR produces a perfect polarization response. Default False
    collapse_images: bool
        If True, measurements are concatenated along the wavelength dimension. If False (default) each wavelength is
        returned as an element of List[RadianceSpectralImage], as would be true in a real instrument.
    central_aotf_wavelength: bool
        Sets the wavelength shift and stretch of diffraction efficiency.
    aperture_effective_area_cm2: float
        Effective aperture area of the instrument. Along with the `image_horiz_fov` and `image_vert_fov` sets the
        etendue.
    single_channel_aotf: bool
        If `True` use the 10mm Brimrose AOTF. If `False` use the 20mm dual channel AOTF.
    """

    def __init__(
        self,
        wavelength_nm: np.ndarray,
        pixel_vert_fov: LineShape = None,
        pixel_horiz_fov: LineShape = None,
        image_horiz_fov: float = 5.0,
        image_vert_fov: float = 1.5,
        num_columns: int = 1,
        num_rows: int = 100,
        ideal_optics: bool = False,
        collapse_images: bool = False,
        central_aotf_wavelength: float = 850.0,
        central_lcr_wavelength: float = 850.0,
        aperture_effective_area_cm2: float = 0.4626,
        single_channel_aotf: bool = True,
        straylight: float = 0.0,
    ):
        if num_columns == 1:
            horiz_mode = "constant"
            vert_mode = "linear"
        else:
            vert_mode = "linear"
            horiz_mode = "linear"

        if pixel_vert_fov is None:
            # pixel_vert_fov = Rectangle(0.0056 * np.pi / 180, mode=vert_mode)
            # pixel_vert_fov = Gaussian(stdev=0.0056 * np.pi / 180, mode=vert_mode, max_stdev=50)
            pixel_vert_fov = Gaussian(
                stdev=0.0092 * np.pi / 180, mode=vert_mode, max_stdev=50
            )

        # if straylight > 0.0:
        #     x = np.linspace(-np.pi / 2, np.pi / 2, 1000)
        #     straylight_fov_vert = UserLineShape(
        #         x_values=x, line_values=np.cos(x), zero_centered=False, mode="integrate"
        #     )
        #     straylight_fov_horiz = UserLineShape(
        #         x_values=x, line_values=np.cos(x), zero_centered=False, mode="simple"
        #     )
        #     # pixel_vert_fov = pixel_vert_fov + straylight_fov_vert
        #     # pixel_horiz_fov = pixel_horiz_fov + straylight_fov_horiz
        # else:
        #     straylight_fov_vert = None
        #     straylight_fov_horiz = None
        #     pixel_vert_fov = pixel_vert_fov + straylight_fov
        # elif straylight < 0.0:
        #     x = np.linspace(-np.pi/2, np.pi/2, 1000)
        #     pixel_vert_fov = UserLineShape(x_values=x, line_values=np.cos(x), zero_centered=True, mode='integrate')

        if pixel_horiz_fov is None:
            pixel_horiz_fov = Rectangle(0.015 * np.pi / 180, mode=horiz_mode)
            # pixel_horiz_fov = Gaussian(0.025 * np.pi / 180, mode=horiz_mode)

        wavelength_nm = np.array(wavelength_nm)
        self.spectral_lineshape = ALILineShape()
        self.spectral_lineshape_area = np.ones(wavelength_nm.shape, dtype=float)
        for idx, wavel in enumerate(wavelength_nm):
            self.spectral_lineshape_area[idx] = self.spectral_lineshape.area(wavel)

        super().__init__(
            wavelength_nm,
            self.spectral_lineshape,
            pixel_vert_fov,
            pixel_horiz_fov,
            image_horiz_fov,
            image_vert_fov,
            num_columns,
            num_rows,
        )

        self._apply_post_processing = True
        self.apply_calibration = True
        self._add_noise = True
        self._add_dark_current = True
        self._add_adc = True
        self._use_ideal_optics = ideal_optics
        self._aperture_area_cm2 = aperture_effective_area_cm2
        # self._aperture_fov_sr = aperture_field_of_view_sr
        self._exposure_time = 1.0
        self._ccd_temperature = 273.15
        self._simulate_pixel_averaging = 0
        self._gain = 0
        self._collapse_images = collapse_images
        self._central_aotf_wavelength = central_aotf_wavelength
        self._central_lcr_wavelength = central_lcr_wavelength
        self._subsampled_detector = True
        self._single_channel_aotf = single_channel_aotf

        self.twist_angle_off = 0
        self.twist_angle_on = 90

        self.sensor = Sensor()
        self._optics = self._create_optics()
        self.auto_exposure = False
        self.max_exposure = 60

        self.save_level0_signal = True
        self.level0 = None
        self.straylight = straylight

        self.save_diagnostics = False
        self.diagnostics = {"settings": {}, "radiance": {}, "signal": {}}

    @property
    def ccd(self):
        return self.sensor

    @ccd.setter
    def ccd(self, value):
        if value.lower().replace("_", "").replace(" ", "") == "cardinal640":
            self.sensor = Cardinal640()
        elif value.lower().replace("_", "").replace(" ", "") == "cardinal1280":
            self.sensor = Cardinal1280()
        elif value.lower().replace("_", "").replace(" ", "") == "raptorowl640n":
            self.sensor = RaptorOWL640N()
        elif value.lower().replace("_", "").replace(" ", "") == "teledyneccd97":
            self.sensor = TeledyneCCD97()
        else:
            msg = "Sensor not recognized"
            raise ValueError(msg)
        self.sensor.add_noise = self._add_noise
        self.sensor.add_adc = self._add_adc
        self.sensor.add_dark_current = self._add_dark_current
        self.sensor.exposure_time = self._exposure_time
        self._optics = self._create_optics()  # update quantum efficiency

    @property
    def apply_post_processing(self):
        return self._apply_post_processing

    @apply_post_processing.setter
    def apply_post_processing(self, value: bool):
        self._apply_post_processing = value
        if not self._apply_post_processing:
            self.apply_calibration = False

    @property
    def gain(self):
        return self.sensor.gain

    @gain.setter
    def gain(self, value):
        self.sensor.gain = value

    @property
    def add_noise(self) -> bool:
        """
        If True, simulate readout and shot noise on the measurements
        """
        return self._add_noise

    @add_noise.setter
    def add_noise(self, value: bool):
        self._add_noise = value
        self.sensor.add_noise = value

    @property
    def add_dark_current(self) -> bool:
        """
        If True, simulate dark current on the measurements
        """
        return self._add_dark_current

    @add_dark_current.setter
    def add_dark_current(self, value: bool):
        self._add_dark_current = value
        self.sensor.add_dark_current = value

    @property
    def add_adc(self) -> bool:
        """
        If True, simulate the analog-to-digital conversion of the signal
        """
        return self._add_adc

    @add_adc.setter
    def add_adc(self, value: bool):
        self._add_adc = value
        self.sensor.add_adc = value

    @property
    def exposure_time(self) -> float | np.ndarray:
        """
        Exposure time of the measurement in seconds. May be an array if exposure has multiple wavelengths.
        """
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float | np.ndarray):
        self._exposure_time = value
        self.sensor.exposure_time = value

    @property
    def ccd_temperature(self) -> float:
        """
        Temperature of the sensor in kelvin.
        """
        return self._ccd_temperature

    @ccd_temperature.setter
    def ccd_temperature(self, value: float):
        self._ccd_temperature = value
        self.sensor.ccd_temperature = value

    @property
    def use_ideal_optics(self):
        return self._use_ideal_optics

    @use_ideal_optics.setter
    def use_ideal_optics(self, value):
        self._use_ideal_optics = value
        self._optics = self._create_optics()

    @property
    def optics(self):
        return CompositeComponent(list(self._optics.values()))

    @property
    def rotator_is_on(self):
        return self._optics["rotator"].twist_angle == self.twist_angle_on

    def _create_optics(self):
        if self.use_ideal_optics:
            return self._create_ideal_optics()
        return self._create_real_optics()

    def _create_real_optics(self):
        # 'mirror-45aoi': Mirror45AOI(),
        # 'mirror-12aoi': Mirror12AOI(),
        # 'lens-codev': CodeVLens(),
        comps = {}
        comps["rotator"] = ArcOptixLCR(
            twist_angle=0,
            thickness=3,
            reference_wavelength=self._central_lcr_wavelength,
        )
        comps["frontend-polarizer"] = MeadowlarkOWLPolarizer(orientation=90)
        if self._single_channel_aotf:
            comps["aotf"] = BrimroseAOTFSingleChannel(
                central_wavelength=self._central_aotf_wavelength
            )
        else:
            comps["aotf"] = Brimrose20mmAOTF()
        comps["backend-polarizer"] = MeadowlarkOWLPolarizer(orientation=0)
        comps["quantum-efficiency"] = self.sensor.quantum_efficiency()
        return comps

        # return {'rotator': ArcOptixLCR(twist_angle=0, thickness=1),
        #         'frontend-polarizer': MeadowlarkOWLPolarizer(orientation=90),
        #         'aotf': BrimroseAOTFSingleChannel(central_wavelength=self._central_aotf_wavelength),
        #         'backend-polarizer': MeadowlarkOWLPolarizer(orientation=0),
        #         'quantum-efficiency': self.sensor.quantum_efficiency()}

    def _create_ideal_optics(self):
        return {
            "rotator": IdealLCR(twist_angle=0),
            "frontend-polarizer": LinearPolarizer(orientation=90),
            "aotf": AOTF(),
            "backend-polarizer": LinearPolarizer(orientation=0),
        }

    @property
    def photon_integrator(self):
        if self._subsampled_detector:
            num_cols = self.sensor.num_columns
            num_rows = self.sensor.num_rows
        else:
            num_rows = self.num_rows
            num_cols = self.num_columns
        return PhotonIntegration(
            aperture_area_cm2=self._aperture_area_cm2,
            num_rows=num_rows,
            num_cols=num_cols,
            horizontal_fov=self.horizontal_fov,
            vertical_fov=self.vertical_fov,
            integration_time=self._exposure_time,
            wavelength_area=self.spectral_lineshape_area,
        )

    def turn_rotator_on(self):
        """
        Turn the liquid crystal rotator on.
        """
        self._optics["rotator"].twist_angle = self.twist_angle_on

    def turn_rotator_off(self):
        """
        Turn the liquid crystal rotator off.
        """
        self._optics["rotator"].twist_angle = self.twist_angle_off

    def calibrate_signal(self, model_value, model_wavel_nm):
        # convert from ADC count back to electrons (L1)
        if self.add_adc:
            model_value.data["radiance"].values = (
                model_value.data["radiance"].to_numpy() * self.sensor.adc.adu
            )
            model_value.data["error"].values = (
                model_value.data["error"].to_numpy() * self.sensor.adc.adu
            )

        # remove dark current
        if self.add_dark_current:
            if type(self.exposure_time) is np.ndarray:
                dc = xr.DataArray(
                    self.exposure_time * self.sensor.dark_current.electrons_per_second,
                    dims=["wavelength"],
                    coords=[model_value.data.wavelength.to_numpy()],
                )
                model_value.data["radiance"].values = (
                    model_value.data["radiance"] - dc
                ).to_numpy()
            else:
                dc = self.exposure_time * self.sensor.dark_current.electrons_per_second
                model_value.data["radiance"].values = (
                    model_value.data["radiance"].to_numpy() - dc
                )

        # convert from electrons to radiance
        model_value.data["radiance"].values = (
            model_value.data["radiance"].to_numpy()
            / self.photon_integrator.scale_factor[:, np.newaxis, np.newaxis]
        )
        model_value.data["error"].values = (
            model_value.data["error"].to_numpy()
            / self.photon_integrator.scale_factor[:, np.newaxis, np.newaxis]
        )

        # undo transmission effects
        g = self.g_parameters(model_wavel_nm)
        # scale_factor = np.abs(g[0]) + np.abs(g[1])
        scale_factor = g[0]
        model_value.data["radiance"].values = (
            model_value.data["radiance"].to_numpy()
            / scale_factor[:, np.newaxis, np.newaxis]
        )
        model_value.data["error"].values = (
            model_value.data["error"].to_numpy()
            / scale_factor[:, np.newaxis, np.newaxis]
        )
        if "wf" in model_value.data:
            model_value.data["wf"].values = (
                model_value.data["wf"].to_numpy()
                / scale_factor[:, np.newaxis, np.newaxis, np.newaxis]
            )

        return model_value

    def g_parameters(self, model_wavel_nm: np.ndarray | None = None):
        if model_wavel_nm is None:
            model_wavel_nm = self._wavelength_nm

        if isinstance(model_wavel_nm, float):
            model_wavel_nm = np.array([model_wavel_nm])

        a = np.ones((4, len(model_wavel_nm)), dtype=float)
        for idx, wavel in enumerate(model_wavel_nm):
            a[:, idx] = self.optics.matrix(wavel)[0]

        g = np.ones((4, len(self._wavelength_nm)), dtype=float)
        for midx, meas_wavel in enumerate(self._wavelength_nm):
            w = self.spectral_lineshape.integration_weights(
                meas_wavel, model_wavel_nm, normalize=True
            )
            g[:, midx] = a @ w

        return g

    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
    ) -> ALISpectralImage | list[ALISpectralImage]:
        if self.save_diagnostics:
            self.diagnostics["settings"]["model_geometry"] = model_geometry
            self.diagnostics["settings"]["optical_geometry"] = optical_geometry
            self.diagnostics["settings"]["model_wavel_nm"] = model_wavel_nm
            self.diagnostics["settings"]["optical_axes"] = self.optical_geometries(
                optical_geometry
            )
            self.diagnostics["radiance"]["frontend_radiance"] = radiance
            self.diagnostics["noise"] = {}

        # pass radiance through optical chain
        if any(["wf" in key for key in radiance.keys()]):
            wf_var = [key for key in radiance.keys() if "wf" in key]
            radiance, wf = self.optics.model_radiance(
                optical_geometry,
                model_wavel_nm,
                model_geometry,
                radiance.drop_vars(wf_var),
                radiance[wf_var],
            )
        elif wf is not None:
            radiance, wf = self.optics.model_radiance(
                optical_geometry, model_wavel_nm, model_geometry, radiance, wf
            )
        else:
            radiance = self.optics.model_radiance(
                optical_geometry, model_wavel_nm, model_geometry, radiance
            )

        # if self.straylight > 0:
        #     radiance['I'] += radiance.I.quantile(0.95) * self.straylight
        #     radiance['Q'] += radiance.Q.quantile(0.95) * self.straylight

        if self.save_diagnostics:
            self.diagnostics["radiance"]["post_optics_radiance"] = radiance

        # convert radiance to detected signal
        model_value = super().model_radiance(
            optical_geometry, model_wavel_nm, model_geometry, radiance, wf=wf
        )

        if self.save_diagnostics:
            self.diagnostics["signal"]["post_sensor_signal"] = model_value.data[
                "radiance"
            ].copy()
            self.diagnostics["settings"]["exposure_time"] = self.exposure_time

        # process the signal through electronic components
        if self.apply_post_processing:
            if np.any(np.isnan(model_value.data["radiance"].to_numpy())):
                logging.warning("Nan value encountered")

            model_value.data["radiance"].values = self.photon_integrator.process_signal(
                model_value.data["radiance"].to_numpy()
            )

            if self.auto_exposure:
                max_val = model_value.data["radiance"].max(dim=["ny", "nx"])
                scale = (self.sensor._max_well_depth / 2) / max_val
                if (
                    self.photon_integrator.integration_time * scale.values
                    > self.max_exposure
                ):
                    # use scale/scale to maintain dimensions
                    scale = (
                        scale
                        * (self.max_exposure / self.photon_integrator.integration_time)
                        / scale
                    )
                model_value.data["radiance"].values = (
                    model_value.data["radiance"] * scale
                ).values
                self.exposure_time = (
                    self.photon_integrator.integration_time * scale.values
                )
                self.photon_integrator.integration_time = self.exposure_time

            if type(self.exposure_time) is np.ndarray:
                exp = self.exposure_time
            else:
                exp = np.ones_like(self.measurement_wavelengths()) * self.exposure_time

            model_value.data["exposure_time"] = xr.DataArray(
                exp,
                dims=["wavelength"],
                coords=[model_value.data.wavelength.to_numpy()],
            )

            model_value.data["error"] = xr.ones_like(model_value.data["radiance"])
            model_value.data["error"].values = self.sensor.noise_estimate(
                model_value.data["radiance"].to_numpy()
            )

            if np.any(np.isnan(model_value.data["radiance"].to_numpy())):
                logging.warning("Nan value encountered")

            if self.save_diagnostics:
                self.diagnostics["signal"]["post_photon-integration_signal"] = (
                    model_value.data["radiance"].copy()
                )
                self.diagnostics["signal"]["post_photon-integration_error"] = (
                    model_value.data["error"].copy()
                )
                self.diagnostics["settings"][
                    "exposure_time"
                ] = self.exposure_time  # update in case of auto exposure

            if self._simulate_pixel_averaging:
                rad = model_value.data["radiance"].to_numpy() * 0
                for i in range(self._simulate_pixel_averaging):
                    rad += self.sensor.process_signal(
                        model_value.data["radiance"].to_numpy()
                    )
                model_value.data["radiance"].values = (
                    rad / self._simulate_pixel_averaging
                )
                model_value.data["error"].values = model_value.data[
                    "error"
                ].to_numpy() / np.sqrt(self._simulate_pixel_averaging)
            else:
                model_value.data["radiance"].values = self.sensor.process_signal(
                    model_value.data["radiance"].to_numpy()
                )

            if self.save_diagnostics:
                self.diagnostics["signal"]["post_electronics_signal"] = (
                    model_value.data["radiance"].copy()
                )
                self.diagnostics["signal"]["post_electronics_error"] = model_value.data[
                    "error"
                ].copy()

        if self.save_level0_signal:
            self.level0 = model_value.data.copy(deep=True)

        if self.apply_calibration:
            # calibrate the signal from L0 -> L1
            model_value = self.calibrate_signal(model_value, model_wavel_nm)

            if self.save_diagnostics:
                self.diagnostics["signal"]["post_calibration_signal"] = (
                    model_value.data["radiance"].copy()
                )
                self.diagnostics["signal"]["post_calibration_error"] = model_value.data[
                    "error"
                ].copy()

        if not self._collapse_images and (len(model_value.data.wavelength.values) > 1):
            model_values = []
            for wavel in model_value.data.wavelength.to_numpy():
                model_values.append(
                    ALISpectralImage(
                        model_value.to_gridded().data.sel(wavelength=float(wavel)),
                        num_columns=self.num_columns,
                    )
                )
            return model_values

        if len(model_value.data.wavelength.values) == 1:
            model_value = ALISpectralImage(
                model_value.to_gridded().data.sel(
                    wavelength=float(model_value.data.wavelength.to_numpy())
                ),
                num_columns=self.num_columns,
            )

        return model_value

    def measurement_geometry(
        self, optical_geometry, num_columns: int = None, num_rows: int = None
    ):
        if self.straylight:
            los = super().optical_geometries(optical_geometry, num_columns, num_rows)
            left = np.cross(optical_geometry.look_vector, optical_geometry.local_up)
            left /= np.linalg.norm(left)
            for angle in np.arange(-90, 90, 0.2) * np.pi / 180:
                if np.abs(angle) < (self.vertical_fov / 2 * (np.pi / 180)):
                    continue
                r = rotation_matrix(left, angle)
                l = r @ optical_geometry.look_vector
                u = np.cross(left, l)
                u /= np.linalg.norm(u)
                o = OpticalGeometry(
                    look_vector=l,
                    observer=optical_geometry.observer,
                    local_up=u,
                    mjd=optical_geometry.mjd,
                )
                los.append(o)

            angles = []
            for l in los:
                angles.append(
                    np.sign(np.dot(l.look_vector, optical_geometry.local_up))
                    * np.arccos(np.dot(l.look_vector, optical_geometry.look_vector))
                    * 180
                    / np.pi
                )
            angles = np.array(angles)
            sidx = np.argsort(angles)
            return [
                LineOfSight(
                    mjd=los[idx].mjd,
                    observer=los[idx].observer,
                    look_vector=los[idx].look_vector,
                )
                for idx in sidx
            ]

        else:
            return super().measurement_geometry(optical_geometry, num_columns, num_rows)


class ALISensorER2(ALISensor):
    def __init__(
        self,
        wavelength_nm: np.ndarray,
        pixel_vert_fov: LineShape = None,
        pixel_horiz_fov: LineShape = None,
        image_horiz_fov: float = 5.0,
        image_vert_fov: float = 1.5,
        num_columns: int = 1,
        num_rows: int = 100,
        ideal_optics: bool = False,
        collapse_images: bool = False,
        central_aotf_wavelength: float = 850.0,
        central_lcr_wavelength: float = 850.0,
        aperture_effective_area_cm2: float = 0.4626,
        single_channel_aotf: bool = True,
        straylight: float = 0.0,
    ):
        super().__init__(
            wavelength_nm=wavelength_nm,
            pixel_vert_fov=pixel_vert_fov,
            pixel_horiz_fov=pixel_horiz_fov,
            image_horiz_fov=image_horiz_fov,
            image_vert_fov=image_vert_fov,
            num_columns=num_columns,
            num_rows=num_rows,
            ideal_optics=ideal_optics,
            collapse_images=collapse_images,
            central_aotf_wavelength=central_aotf_wavelength,
            central_lcr_wavelength=central_lcr_wavelength,
            aperture_effective_area_cm2=aperture_effective_area_cm2,
            single_channel_aotf=single_channel_aotf,
            straylight=straylight,
        )

        wavelength_nm = np.array(wavelength_nm)
        self.spectral_lineshape = ALIER2LineShape()
        self.spectral_lineshape_area = np.ones(wavelength_nm.shape, dtype=float)
        for idx, wavel in enumerate(wavelength_nm):
            self.spectral_lineshape_area[idx] = self.spectral_lineshape.area(wavel)

        self.ccd = "raptorowl640n"

    def _create_real_optics(self):
        comps = {}
        comps["periscope"] = GoldMirror() + GoldMirror() + GoldMirror()
        comps["front-optics"] = (
            Mirror12AOI() + Mirror12AOI() + Mirror12AOI() + Mirror12AOI()
        )
        comps["rotator"] = ArcOptixLCR(
            twist_angle=0,
            thickness=3,
            reference_wavelength=self._central_lcr_wavelength,
        )
        comps["frontend-polarizer"] = MeadowlarkOWLPolarizer(orientation=90)
        comps["aotf"] = ER2AOTF()
        comps["backend-polarizer"] = MeadowlarkOWLPolarizer(orientation=0)
        comps["back-optics"] = (
            Mirror12AOI() + Mirror12AOI() + Mirror12AOI() + Mirror12AOI()
        )
        comps["quantum-efficiency"] = self.sensor.quantum_efficiency()
        return comps

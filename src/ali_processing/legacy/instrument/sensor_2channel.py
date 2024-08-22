from __future__ import annotations

from typing import List, Union

import numpy as np
import xarray as xr
from sasktran import Geometry
from skretrieval.core.lineshape import DeltaFunction, Gaussian, LineShape, Rectangle
from skretrieval.core.radianceformat import RadianceSpectralImage
from skretrieval.core.sensor import OpticalGeometry
from skretrieval.legacy.core.sensor.imager import SpectralImager

from ali_processing.legacy.instrument.sensor import ALISensor


class ALISensorDualChannel:
    """
    ALI instrument with a two-channel design that uses a dichroic beam splitter to separate the incoming beam into a
    short and long-wavelength channel. Parameters passed to the model are applied to both of the channels.

    Parameters
    ----------
    wavelength_nm: np.ndarray
        Array of wavelengths that will be measuremed. The input radiances should set to accurately samples these values.
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
    """

    def __init__(
        self,
        wavelength_nm: np.ndarray,
        image_horiz_fov: float = 5.0,
        image_vert_fov: float = 1.5,
        num_columns: int = 1,
        num_rows: int = 100,
        pixel_vert_fov: LineShape = None,
        pixel_horiz_fov: LineShape = None,
        ideal_optics=False,
        straylight=0.0,
    ):

        wavelength_nm = np.array(wavelength_nm)
        self.split_wavelength_nm = 950.1
        wavelength_nm_1 = wavelength_nm[wavelength_nm <= self.split_wavelength_nm]
        wavelength_nm_2 = wavelength_nm[wavelength_nm > self.split_wavelength_nm]

        aperture_area = 1.0 * (4 * 4) / (image_horiz_fov * image_vert_fov) * 10
        self.channel_1 = ALISensor(
            wavelength_nm_1,
            image_horiz_fov=image_horiz_fov,
            image_vert_fov=image_vert_fov,
            num_columns=num_columns,
            num_rows=num_rows,
            pixel_vert_fov=pixel_vert_fov,
            pixel_horiz_fov=pixel_horiz_fov,
            ideal_optics=ideal_optics,
            central_aotf_wavelength=750.0,
            central_lcr_wavelength=750.0,
            aperture_effective_area_cm2=aperture_area,
            straylight=straylight,
        )
        self.channel_1.ccd = "teledyneccd97"
        self.channel_2 = ALISensor(
            wavelength_nm_2,
            image_horiz_fov=image_horiz_fov,
            image_vert_fov=image_vert_fov,
            num_columns=num_columns,
            num_rows=num_rows,
            pixel_vert_fov=pixel_vert_fov,
            pixel_horiz_fov=pixel_horiz_fov,
            ideal_optics=ideal_optics,
            central_aotf_wavelength=1200.0,
            central_lcr_wavelength=1200.0,
            aperture_effective_area_cm2=aperture_area,
            straylight=straylight,
        )
        self.channel_2.ccd = "raptorowl640n"
        self._wavelength_nm = wavelength_nm
        self._straylight = 0.0

    def set_gain(self, gain, channel=2):
        if channel == 2:
            self.channel_2.gain = gain
        elif channel == 1:
            self.channel_1.gain = gain

    @property
    def straylight(self):
        return self._straylight

    @straylight.setter
    def straylight(self, value):
        self.channel_1.straylight = value
        self.channel_2.straylight = value

    @property
    def auto_exposure(self):
        return self.channel_1.auto_exposure

    @auto_exposure.setter
    def auto_exposure(self, value: bool):
        self.channel_1.auto_exposure = value
        self.channel_2.auto_exposure = value

    @property
    def num_columns(self):
        return self.channel_1.num_columns

    @property
    def num_rows(self):
        return self.channel_1.num_rows

    @property
    def save_diagnostics(self):
        return self.channel_1.save_diagnostics

    @save_diagnostics.setter
    def save_diagnostics(self, value):
        self.channel_1.save_diagnostics = value
        self.channel_2.save_diagnostics = value

    @property
    def simulate_pixel_averaging(self):
        return self.channel_1._simulate_pixel_averaging

    @simulate_pixel_averaging.setter
    def simulate_pixel_averaging(self, value):
        self.channel_1._simulate_pixel_averaging = value
        self.channel_2._simulate_pixel_averaging = value

    @property
    def diagnostics(self):
        return [self.channel_1.diagnostics, self.channel_2.diagnostics]

    @property
    def ccd_temperature(self):
        return self.channel_1.ccd_temperature

    @ccd_temperature.setter
    def ccd_temperature(self, value):
        self.channel_1.ccd_temperature = value
        self.channel_2.ccd_temperature = value

    @property
    def exposure_time(self) -> list[Union[float, np.ndarray]]:
        wavels = self.measurement_wavelengths()
        exp = []
        if any(wavels < self.split_wavelength_nm):
            exp.append(self.channel_1.exposure_time)
        if any(wavels > self.split_wavelength_nm):
            exp.append(self.channel_2.exposure_time)
        if len(exp) == 1:
            exp = exp[0]
        return exp

    @exposure_time.setter
    def exposure_time(self, value):
        if not hasattr(value, "__len__"):
            value = [value, value]
        self.channel_1.exposure_time = value[0]
        self.channel_2.exposure_time = value[1]

    @property
    def add_adc(self):
        return self.channel_1.add_adc

    @add_adc.setter
    def add_adc(self, value):
        self.channel_1.add_adc = value
        self.channel_2.add_adc = value

    @property
    def add_noise(self):
        return self.channel_1.add_noise

    @add_noise.setter
    def add_noise(self, value):
        self.channel_1.add_noise = value
        self.channel_2.add_noise = value

    @property
    def add_dark_current(self):
        return self.channel_1.add_dark_current

    @add_dark_current.setter
    def add_dark_current(self, value):
        self.channel_1.add_dark_current = value
        self.channel_2.add_dark_current = value

    @property
    def rotator_is_on(self):
        return self.channel_1.rotator_is_on

    def turn_rotator_on(self):
        self.channel_1.turn_rotator_on()
        self.channel_2.turn_rotator_on()

    def turn_rotator_off(self):
        self.channel_1.turn_rotator_off()
        self.channel_2.turn_rotator_off()

    def optical_geometries(
        self, optical_geometry, num_columns: int = None, num_rows: int = None
    ):
        return self.channel_1.optical_geometries(
            optical_geometry, num_columns, num_rows
        )

    def measurement_geometry(
        self,
        optical_geometry: OpticalGeometry,
        num_columns: int = None,
        num_rows: int = None,
    ):
        return self.channel_1.measurement_geometry(
            optical_geometry, num_columns, num_rows
        )

    def measurement_wavelengths(self) -> np.ndarray:
        w1 = np.unique(
            np.array([p.measurement_wavelengths() for p in self.channel_1._pixels])
        )
        w2 = np.unique(
            np.array([p.measurement_wavelengths() for p in self.channel_2._pixels])
        )
        return np.unique(np.concatenate([w1, w2]))

    def required_wavelengths(self, res_nm: float) -> np.ndarray:
        w1 = np.unique(
            np.array([p.required_wavelengths(res_nm) for p in self.channel_1._pixels])
        )
        w2 = np.unique(
            np.array([p.required_wavelengths(res_nm) for p in self.channel_2._pixels])
        )
        return np.concatenate([w1, w2])

    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
    ) -> RadianceSpectralImage:

        model_wavel_nm_1 = model_wavel_nm[model_wavel_nm <= self.split_wavelength_nm]
        model_wavel_nm_2 = model_wavel_nm[model_wavel_nm > self.split_wavelength_nm]

        if len(model_wavel_nm_1 > 0):
            good_wavel = model_wavel_nm <= self.split_wavelength_nm
            if wf:
                r1 = self.channel_1.model_radiance(
                    optical_geometry,
                    model_wavel_nm[good_wavel],
                    model_geometry,
                    radiance.sel(wavelength=slice(0, self.split_wavelength_nm)),
                    [w.sel(wavelength=slice(0, self.split_wavelength_nm)) for w in wf],
                )
            else:
                r1 = self.channel_1.model_radiance(
                    optical_geometry,
                    model_wavel_nm[good_wavel],
                    model_geometry,
                    radiance.sel(wavelength=slice(0, self.split_wavelength_nm)),
                    None,
                )

        if len(model_wavel_nm_2 > 0):
            good_wavel = model_wavel_nm >= self.split_wavelength_nm
            if wf:
                r2 = self.channel_2.model_radiance(
                    optical_geometry,
                    model_wavel_nm[good_wavel],
                    model_geometry,
                    radiance.sel(wavelength=slice(self.split_wavelength_nm, 3000.0)),
                    [
                        w.sel(wavelength=slice(self.split_wavelength_nm, 3000))
                        for w in wf
                    ],
                )
            else:
                r2 = self.channel_2.model_radiance(
                    optical_geometry,
                    model_wavel_nm[good_wavel],
                    model_geometry,
                    radiance.sel(wavelength=slice(self.split_wavelength_nm, 3000.0)),
                    None,
                )

        if len(model_wavel_nm_1 > 0) and len(model_wavel_nm_2 > 0):
            if type(r1) is list and type(r2) is list:
                return [*r1, *r2]
            elif type(r1) is list:
                return [*r1, r2]
            elif type(r2) is list:
                return [r1, *r2]
            else:
                return [r1, r2]
        elif len(model_wavel_nm_1 > 0):
            return r1
        elif len(model_wavel_nm_2 > 0):
            return r2

    def pixel_optical_axes(
        self,
        optical_axis: OpticalGeometry,
        hfov: float,
        vfov: float,
        num_columns: int,
        num_rows: int,
    ) -> list[OpticalGeometry]:

        return self.channel_1.pixel_optical_axes(
            optical_axis, hfov, vfov, num_columns, num_rows
        )

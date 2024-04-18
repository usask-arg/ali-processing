from __future__ import annotations

import logging
from copy import copy

import numpy as np
import xarray as xr
from sasktran import Geometry, StokesVector
from skretrieval.core import OpticalGeometry
from skretrieval.core.lineshape import LineShape
from skretrieval.core.radianceformat import (
    RadianceGridded,
    RadianceRaw,
    RadianceSpectralImage,
)
from skretrieval.core.sensor.imager import SpectralImager


class ALISpectralImage(RadianceSpectralImage):
    def to_gridded(self) -> RadianceGridded:
        # dsc = self._ds.copy().stack(los=('nx', 'ny')).reset_index('los', drop=True)
        dsc = self._ds.copy().stack(los=("ny", "nx")).reset_index("los", drop=True)
        try:
            dsc["radiance"] = dsc["radiance"].transpose("wavelength", "los")
        except ValueError:
            dsc["radiance"] = dsc["radiance"].transpose("los")

        if "wf" in dsc:
            try:
                dsc["wf"] = dsc["wf"].transpose("wavelength", "los", ...)
            except ValueError:
                dsc["wf"] = dsc["wf"].transpose("los", ...)
        else:
            for key in dsc:
                if "wf_" in key:
                    try:
                        dsc[key] = dsc[key].transpose("wavelength", "los", ...)
                    except ValueError:
                        dsc[key] = dsc[key].transpose("los", ...)
        return RadianceGridded(dsc)

    def to_raw(self) -> RadianceRaw:
        return self.to_gridded().to_raw()


class SpectralImagerFast(SpectralImager):
    """
    Similar to `SpectralImager`, but cannot have a variable `LineShape` for different pixels.
    """

    def __init__(
        self,
        wavelength_nm: np.ndarray,
        spectral_lineshape: LineShape,
        pixel_vert_fov: LineShape,
        pixel_horiz_fov: LineShape,
        image_horiz_fov: float,
        image_vert_fov: float,
        num_columns: int,
        num_rows: int,
    ):
        self.horizontal_fov = image_horiz_fov
        self.vertical_fov = image_vert_fov
        self._num_columns = num_columns
        self._num_rows = num_rows
        self._wavelength_nm = wavelength_nm
        self._spectral_lineshape = spectral_lineshape
        self._los_interpolator = None
        self._cached_wavel_interp = None
        self._cached_wavel_interp_wavel = None
        self.pixel_vert_fov = pixel_vert_fov
        self.pixel_horiz_fov = pixel_horiz_fov
        self.chunk_size = None

    @property
    def num_columns(self):
        return self._num_columns

    @property
    def num_rows(self):
        return self._num_rows

    def optical_geometries(
        self,
        optical_geometry,
        num_columns: int | None = None,
        num_rows: int | None = None,
    ):
        if num_columns is None:
            num_columns = self._num_columns

        if num_rows is None:
            num_rows = self._num_rows

        return self.pixel_optical_axes(
            optical_geometry,
            self.horizontal_fov,
            self.vertical_fov,
            num_columns,
            num_rows,
        )

    @staticmethod
    def project_to_plane(
        optical_axis: OpticalGeometry,
        model_geometry: list[OpticalGeometry] | Geometry,
    ):
        """
        Return the vertical and horizontal angles in radians of the `model_geometry` with respect to the `optical_axis`.

        Parameters
        ----------
        optical_axis
        model_geometry
        """
        x_axis = np.array(optical_axis.look_vector)
        vert_normal = np.cross(np.array(x_axis), np.array(optical_axis.local_up))
        vert_normal = vert_normal / np.linalg.norm(vert_normal)
        vert_y_axis = np.cross(vert_normal, x_axis)

        horiz_y_axis = vert_normal

        horiz_angle = []
        vert_angle = []

        try:
            lines_of_sight = model_geometry.lines_of_sight
        except AttributeError:
            lines_of_sight = model_geometry

        for los in lines_of_sight:
            vert_angle.append(
                np.arctan2(
                    np.dot(los.look_vector, vert_y_axis),
                    np.dot(los.look_vector, x_axis),
                )
            )
            horiz_angle.append(
                np.arctan2(
                    np.dot(los.look_vector, horiz_y_axis),
                    np.dot(los.look_vector, x_axis),
                )
            )

        return np.array(horiz_angle), np.array(vert_angle)

    def project_to_detector(
        self,
        optical_axis: OpticalGeometry,
        model_geometry: list[OpticalGeometry] | Geometry,
    ):
        """
        Return the vertical and horizontal positions of the `model_geometry`  on the detector plane in units of pixels.
        """

        hfov = self.horizontal_fov * np.pi / 180
        vfov = self.vertical_fov * np.pi / 180

        hangle, vangle = self.project_to_plane(optical_axis, model_geometry)
        hpixel = (hangle + hfov / 2) * self._num_columns
        vpixel = (vangle + vfov / 2) * self._num_rows

        return hpixel, vpixel

    @staticmethod
    def pixel_optical_axes(
        optical_axis: OpticalGeometry,
        hfov: float,
        vfov: float,
        num_columns: int,
        num_rows: int,
    ) -> list[OpticalGeometry]:
        """
        Get the optical geometry at the center of each pixel in the sensor.

        Parameters
        ----------
        optical_axis : OpticalGeometry
            The optical axis of the center of the sensor
        hfov : float
            horizontal field of view in degrees
        vfov : float
            vertical field of view in degrees
        num_columns :
            Number of colums in the sensor
        num_rows :
            Number of rows in the sensor

        Returns
        -------
        List[OpticalGeometry]
            The optical geoemetry of each pixel in the sensor as a row-major list.
        """
        look = optical_axis.look_vector
        up = optical_axis.local_up
        mjd = optical_axis.mjd
        obs = optical_axis.observer

        dist = 1e6  # used for scaling purposes to improve numerical precision
        htan = np.tan(hfov / 2 * np.pi / 180) * dist
        vtan = np.tan(vfov / 2 * np.pi / 180) * dist

        if num_columns == 1:
            x = np.array([0.0])
        else:
            x = np.linspace(-htan, htan, num_columns)

        y = np.array([0.0]) if num_rows == 1 else np.linspace(-vtan, vtan, num_rows)

        nx, ny = np.meshgrid(x, y)
        center = obs + look * dist
        xaxis = np.cross(up, look)
        xaxis /= np.linalg.norm(xaxis)
        yaxis = up

        nyf = ny.flatten()
        nxf = nx.flatten()
        pos = center + xaxis * nxf[:, np.newaxis] + yaxis * nyf[:, np.newaxis]
        los = pos - obs
        los = los / np.linalg.norm(los, axis=1)[:, np.newaxis]
        pixel_yaxis = np.cross(los, up, axisa=1)
        pixel_yaxis = pixel_yaxis / np.linalg.norm(pixel_yaxis, axis=1)[:, np.newaxis]
        pixel_up = np.cross(pixel_yaxis, los, axisa=1, axisb=1)
        pixel_up = pixel_up / np.linalg.norm(pixel_up, axis=1)[:, np.newaxis]

        # TODO: the list allocation is kind of slow, can this be improved?
        return [
            OpticalGeometry(observer=obs, look_vector=look, local_up=u, mjd=mjd)
            for look, u in zip(los, pixel_up, strict=False)
        ]

    def _construct_interpolators(
        self, model_geometry, optical_geometry, model_wavel_nm, los_interp=True
    ):
        if los_interp:
            # los_interp = self._pixels[0]._construct_los_interpolator(model_geometry, optical_geometry)
            los_interp = self._construct_los_interpolator(
                model_geometry, optical_geometry
            )
        else:
            los_interp = None

        if not np.array_equal(model_wavel_nm, self._cached_wavel_interp_wavel):
            # wavel_interp = []
            wavel_interp = self._construct_wavelength_interpolator(model_wavel_nm)
            # for p in self._pixels:
            #     wavel_interp.append(
            #         p._construct_wavelength_interpolator(model_wavel_nm)
            #     )

            # wavel_interp = np.vstack(wavel_interp)
            self._cached_wavel_interp = wavel_interp
            self._cached_wavel_interp_wavel = copy(model_wavel_nm)

        return self._cached_wavel_interp, los_interp

    def model_radiance(
        self,
        optical_geometry: OpticalGeometry,
        model_wavel_nm: np.array,
        model_geometry: Geometry,
        radiance: np.array,
        wf=None,
        boresight: OpticalGeometry = None,
    ) -> ALISpectralImage:
        if boresight is None:
            boresight = optical_geometry

        optical_axes = self.optical_geometries(optical_geometry)
        # wavel_interp, _ = self._pixels[0]._construct_interpolators(model_geometry, optical_axes[0], model_wavel_nm, los_interp=False)
        wavel_interp, _ = self._construct_interpolators(
            model_geometry, optical_axes[0], model_wavel_nm, los_interp=False
        )

        # if self._los_interpolator is None:
        if self.chunk_size:
            optical_axes_chunks = self.divide_chunks(optical_axes, self.chunk_size)
        else:
            optical_axes_chunks = [optical_axes]

        # modelled_radiances = []
        idx = 0
        data = []
        for oa in optical_axes_chunks:
            self._los_interpolator = self._construct_los_interpolator(
                model_geometry, oa
            )

            if type(radiance) is xr.Dataset:
                if "I" in radiance:
                    radiance = radiance.I.to_numpy()
            elif type(radiance[0][0]) is StokesVector:
                radiance = np.array([[r.I for r in rad] for rad in radiance])

            if len(self._los_interpolator.shape) <= 2:
                modelled_radiances = np.einsum(
                    "ij,jk...,kl",
                    wavel_interp,
                    radiance,
                    self._los_interpolator,
                    optimize="optimal",
                )
            else:
                x = np.einsum(
                    "...j,jk...", radiance, self._los_interpolator, optimize="optimal"
                )
                modelled_radiances = np.einsum(
                    "ij,jk", wavel_interp, x, optimize="optimal"
                )
            data.append(
                xr.Dataset(
                    {
                        "radiance": (["wavelength", "los"], modelled_radiances),
                        "mjd": (["los"], np.array([o.mjd for o in oa])),
                        "los_vectors": (
                            ["los", "xyz"],
                            np.array([o.look_vector for o in oa]),
                        ),
                        "observer_position": (
                            ["los", "xyz"],
                            np.array([o.observer for o in oa]),
                        ),
                    },
                    coords={
                        "wavelength": self.measurement_wavelengths(),
                        "xyz": ["x", "y", "z"],
                        "los": np.arange(0, len(oa)) + idx,
                    },
                )
            )
            if wf is not None:
                if type(wf) is xr.Dataset:
                    for key in wf:
                        if len(wf[key].shape) == 4:  # select the "I" component
                            modelled_wf = np.einsum(
                                "ij,jkl,km->iml",
                                wavel_interp,
                                wf[key][:, :, :, 0],
                                self._los_interpolator,
                                optimize="optimal",
                            )
                            data[-1][key] = (wf[key].dims[0:-1], modelled_wf)
                        else:
                            modelled_wf = np.einsum(
                                "ij,jkl,km->iml",
                                wavel_interp,
                                wf[key],
                                self._los_interpolator,
                                optimize="optimal",
                            )
                            data[-1][key] = (wf[key].dims, modelled_wf)
                else:
                    modelled_wf = np.einsum(
                        "ij,jkl,km->iml",
                        wavel_interp,
                        wf,
                        self._los_interpolator,
                        optimize="optimal",
                    )
                    data[-1]["wf"] = (wf.dims, modelled_wf)

            idx += len(oa)

        data = xr.concat(data, dim="los")
        return ALISpectralImage(data, num_columns=self._num_columns)

    def _construct_los_interpolator(
        self, model_geometry, optical_axes: list[OpticalGeometry]
    ):
        """
        Internally constructs the matrix used for the line of sight interpolation
        """
        x_axis = np.array([oa.look_vector for oa in optical_axes])
        local_up = np.array([oa.local_up for oa in optical_axes])
        vert_normal = np.cross(x_axis, local_up, axisa=1, axisb=1)
        vert_normal = vert_normal / np.linalg.norm(vert_normal, axis=1)[:, np.newaxis]
        vert_y_axis = np.cross(vert_normal, x_axis, axisa=1, axisb=1)

        horiz_y_axis = vert_normal
        geom_los = np.array([los.look_vector for los in model_geometry.lines_of_sight])

        los_dot_x = np.einsum("ij,kj->ik", geom_los, x_axis)
        vert_angle = np.arctan2(
            np.einsum("ij,kj->ik", geom_los, vert_y_axis), los_dot_x
        )
        horiz_angle = np.arctan2(
            np.einsum("ij,kj->ik", geom_los, horiz_y_axis), los_dot_x
        )

        horiz_interpolator = np.ones_like(horiz_angle)
        vert_interpolator = np.ones_like(horiz_angle)

        mean_vert = 0.0
        mean_hor = 0.0
        horiz_fov = self.pixel_horiz_fov
        vert_fov = self.pixel_vert_fov
        # horiz_fov = self.horizontal_fov
        # vert_fov = self.vertical_fov

        for idx in range(len(optical_axes)):
            horiz_interpolator[:, idx] = horiz_fov.integration_weights(
                mean_hor, horiz_angle[:, idx]
            )
            vert_interpolator[:, idx] = vert_fov.integration_weights(
                mean_vert, vert_angle[:, idx]
            )

        # TODO: Only valid for exponential distributions?
        los_interpolator = horiz_interpolator * vert_interpolator
        los_interpolator = (
            los_interpolator / np.nansum(los_interpolator, axis=0)[np.newaxis, :]
        )
        los_interpolator[np.isnan(los_interpolator)] = 0.0

        # if self._vert_fov_straylight:
        #     los_interpolator_stray = horiz_interpolator_stray * vert_interpolator_stray
        #     los_interpolator_stray = los_interpolator_stray / np.nansum(
        #         los_interpolator_stray
        #     )

        if np.any(np.isnan(los_interpolator)):
            logging.warning("NaN encountered in SpectralImagerFast los constructor")

        # if self._vert_fov_straylight:
        #     return (
        #         los_interpolator
        #         + (los_interpolator_stray * self._straylight_scalar)[:, np.newaxis]
        #     )
        # else:
        return los_interpolator

    def _construct_wavelength_interpolator(self, hires_wavel_nm):
        """
        Internally constructs the matrix used for the wavelength interpolation
        """
        wavel_interpolator = np.zeros((len(self._wavelength_nm), len(hires_wavel_nm)))
        for idx, wavel in enumerate(self._wavelength_nm):
            wavel_interpolator[idx, :] = self._spectral_lineshape.integration_weights(
                wavel, hires_wavel_nm
            )

        return wavel_interpolator

    @staticmethod
    def divide_chunks(array, n):
        # looping till length l
        for i in range(0, len(array), n):
            yield array[i : i + n]

    def measurement_wavelengths(self) -> np.ndarray:
        return self._wavelength_nm

    def radiance_format(self) -> type[ALISpectralImage]:
        return ALISpectralImage

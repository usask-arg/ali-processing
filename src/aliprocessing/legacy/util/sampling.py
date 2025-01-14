from __future__ import annotations

import os

import numpy as np
import sasktran as sk
import xarray as xr
from skretrieval.core import OpticalGeometry
from skretrieval.legacy.core.sensor.imager import SpectralImager
from skretrieval.util import rotation_matrix


class Sampling:
    """
    Construct a measurement geometry that samples the sensor at a variable horizontal and vertical resolutions
    depending on the tangent altitude.


    Parameters
    ----------
    sensor: SpectralImager
    optical_axis: OpticalGeometry


    Examples
    --------
    >>> sensor = SpectralImager(...)
    >>> opt_geom = OpticalGeometry(...)
    >>> samples = Sampling(sensor, opt_geom)
    >>> samples.horizontal_spacing_trop = 1.0  # increase resolution at the tropopause to 1 km
    >>> samples.tropopause_altitude = 12.0  # set tropopause altitude to 12 km
    >>> model_geometry = samples.meas_geom()
    >>> samples.set_weights(sensor)  # The LineShape weights in the sensor need to be set before performing a radiance calculation

    """

    def __init__(self, sensor: SpectralImager, optical_axis: OpticalGeometry):

        self.sensor = sensor
        self._optical_axis = optical_axis

        self.horizontal_spacing_trop = 5
        self.vertical_spacing_trop = 0.25
        self.horizontal_spacing_ground = 25
        self.vertical_spacing_ground = 0.5
        self.horizontal_spacing_toa = 100
        self.vertical_spacing_toa = 2

        self.tropopause_altitude = 10
        self.toa_altitude = 45

        geo = sk.Geodetic()
        geo.from_tangent_point(optical_axis.observer, optical_axis.look_vector)
        self.boresight_altitude = geo.altitude / 1000
        self.boresight_tp = geo.location
        self.boresight_dist = np.linalg.norm(geo.location - optical_axis.observer)

        geo.from_xyz(self._optical_axis.observer)
        self.observer_altitude = geo.altitude / 1000

        self.truncate_edges = False
        self.edges = []
        self.angles = []
        self.widths = []
        self.heights = []
        self._meas_geom = None

    def set_weights(self, sensor: SpectralImager):
        for spectrograph in sensor._pixels:
            for pixel in spectrograph._pixels:
                pixel._horiz_fov.interpolation_width_left = self.widths
                pixel._horiz_fov.interpolation_width_right = self.widths
                pixel._vert_fov.interpolation_width_left = self.heights
                pixel._vert_fov.interpolation_width_right = self.heights

    @property
    def optical_axis(self):
        return self._optical_axis

    @optical_axis.setter
    def optical_axis(self, value: OpticalGeometry):
        self._optical_axis = value
        geo = sk.Geodetic()
        geo.from_tangent_point(value.observer, value.look_vector)
        self.boresight_altitude = geo.altitude / 1000
        self.edges = []
        self.angles = []
        self.widths = []
        self.heights = []
        self._meas_geom = None

    def meas_geom(self):

        if self._meas_geom is not None:
            return self._meas_geom

        pixels = self.sensor.pixel_optical_axes(
            self._optical_axis,
            self.sensor.horizontal_fov,
            self.sensor.vertical_fov,
            self.sensor.num_columns,
            self.sensor.num_rows,
        )

        geo = sk.Geodetic()
        alts = []
        for pixel in pixels:
            geo.from_tangent_point(
                observer=pixel.observer, look_vector=pixel.look_vector
            )
            alts.append(geo.altitude / 1000)

        # alts = np.array(alts).reshape(self.sensor.num_rows, self.sensor.num_columns)
        alts = np.array(alts)

        v_angles_above = self.angular_samples(
            self.tropopause_altitude, np.max(alts) + self.vertical_spacing_toa
        )
        v_angles_below = self.angular_samples(
            self.tropopause_altitude, np.min(alts) - self.vertical_spacing_ground
        )
        vert_angles = np.concatenate(
            [
                v_angles_below,
                [self.vertical_angle_from_altitude(self.tropopause_altitude)],
                v_angles_above,
            ]
        )
        vert_angles = np.sort(vert_angles)

        hfov = self.sensor.horizontal_fov / 2 * np.pi / 180
        vfov = self.sensor.vertical_fov / 2 * np.pi / 180
        lines_of_sight = []
        for vidx, vert_angle in enumerate(vert_angles):
            # s = np.interp(v,
            #               [0, self.tropopause_altitude, self.toa_altitude],
            #               [self.horizontal_spacing_ground, self.horizontal_spacing_trop, self.horizontal_spacing_toa])

            hor = np.cross(self._optical_axis.look_vector, self._optical_axis.local_up)
            hor /= np.linalg.norm(hor)
            vlos = rotation_matrix(hor, vert_angle) @ self._optical_axis.look_vector

            los_right = rotation_matrix(self._optical_axis.local_up, hfov) @ vlos
            geo.from_tangent_point(self._optical_axis.observer, los_right)
            tp_right = geo.location

            los_left = rotation_matrix(self._optical_axis.local_up, -hfov) @ vlos
            geo.from_tangent_point(self._optical_axis.observer, los_left)
            tp_left = geo.location

            hspan_km = np.linalg.norm(tp_left - tp_right) / 1000
            geo.from_tangent_point(self._optical_axis.observer, vlos)
            n_samples = hspan_km / self.horizontal_spacing(geo.altitude / 1000)

            if (
                n_samples < 3
            ):  # ensure we at least sample the edges and center of the detector
                n_samples = 3

            # make the sampling symmetric about the center of the detector
            h_angles = np.linspace(0, hfov, int(np.ceil(n_samples / 2)))
            h_angles = np.unique(np.concatenate([-h_angles[::-1], h_angles]))
            h_width = 2 * hfov / (len(h_angles) - 1)

            if vidx == 0:
                vert_width_below = (vert_angles[1] - vert_angles[0]) / 2
                vert_width_above = (vert_angles[1] - vert_angles[0]) / 2
            elif vidx == len(vert_angles) - 1:
                vert_width_below = (vert_angles[-1] - vert_angles[-2]) / 2
                vert_width_above = (vert_angles[-1] - vert_angles[-2]) / 2
            else:
                vert_width_below = (vert_angles[vidx] - vert_angles[vidx - 1]) / 2
                vert_width_above = (vert_angles[vidx + 1] - vert_angles[vidx]) / 2

            for h_angle in h_angles:
                los = rotation_matrix(self._optical_axis.local_up, h_angle) @ vlos
                lines_of_sight.append(
                    sk.LineOfSight(
                        mjd=self._optical_axis.mjd,
                        observer=self._optical_axis.observer,
                        look_vector=los,
                    )
                )
                self.angles.append((h_angle, vert_angle))

                left = h_angle - h_width / 2
                right = h_angle + h_width / 2
                upper = vert_angle + vert_width_above
                lower = vert_angle - vert_width_below

                if self.truncate_edges:
                    left = -hfov if left < -hfov else left
                    right = hfov if right > hfov else right
                    upper = vfov if upper > vfov else upper
                    lower = -vfov if lower < -vfov else lower

                self.edges.append(
                    [
                        (left, lower),
                        (left, upper),
                        (right, upper),
                        (right, lower),
                        (left, lower),
                    ]
                )
                self.widths.append(np.abs(right - left))
                self.heights.append(np.abs(upper - lower))

        self.widths = np.array(self.widths)
        self.heights = np.array(self.heights)
        geom = sk.Geometry()
        geom.lines_of_sight = lines_of_sight
        self._meas_geom = geom
        return geom

    def angular_samples(self, start_alt, max_alt):
        altitude = start_alt
        v_angles = []

        if start_alt < max_alt:
            while altitude < max_alt:
                s = self.vertical_spacing(altitude)
                if altitude + s > self.observer_altitude:
                    v_diff = v_angles[-1] - v_angles[-2]
                    v_angles.append(v_angles[-1] + v_diff)
                    altitude += s
                else:
                    s = self.vertical_spacing(altitude)
                    altitude += s
                    v_angles.append(self.vertical_angle_from_altitude(altitude))
        else:
            while altitude > max_alt:
                s = self.vertical_spacing(altitude)
                altitude -= s
                v_angles.append(self.vertical_angle_from_altitude(altitude))

        return v_angles

    def vertical_spacing(self, altitude):
        return np.interp(
            altitude,
            [0, self.tropopause_altitude, self.toa_altitude],
            [
                self.vertical_spacing_ground,
                self.vertical_spacing_trop,
                self.vertical_spacing_toa,
            ],
        )

    def horizontal_spacing(self, altitude):
        return np.interp(
            altitude,
            [0, self.tropopause_altitude, self.toa_altitude],
            [
                self.horizontal_spacing_ground,
                self.horizontal_spacing_trop,
                self.horizontal_spacing_toa,
            ],
        )

    def vertical_angle_from_altitude(self, altitude):
        geo = sk.Geodetic()
        geo.from_tangent_altitude(
            altitude=altitude * 1000,
            observer=self._optical_axis.observer,
            boresight=self._optical_axis.look_vector,
        )
        alt_tp = geo.location
        trop_dist = np.linalg.norm(self._optical_axis.observer - alt_tp)

        o = np.linalg.norm(self.boresight_tp - alt_tp)
        angle = np.arccos(
            (self.boresight_dist**2 + trop_dist**2 - o**2)
            / (2 * self.boresight_dist * trop_dist)
        )
        if altitude < self.boresight_altitude:
            return -angle
        return angle

    @staticmethod
    def plot_samples(
        edges, values=None, ax=None, cmap=None, vmin=None, vmax=None, **kwargs
    ):
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon

        if cmap is None:
            cmap = plt.cm.Spectral_r

        if vmin is None:
            vmin = np.min(values)

        if vmax is None:
            vmax = np.max(values)

        if values is not None:
            values = (values - vmin) / (vmax - vmin)
        else:
            values = np.ones(len(edges)) * 0.3

        polys = []
        colors = []
        for edge, value in zip(edges, values, strict=False):
            colors.append(cmap(value))
            rect = Polygon(np.array(edge), closed=True)
            polys.append(rect)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        p = PatchCollection(polys, alpha=1, facecolors=colors, **kwargs)
        ax.add_collection(p)

        xs = np.concatenate([np.array(s)[:, 0] for s in edges])
        ys = np.concatenate([np.array(s)[:, 1] for s in edges])
        ax.set_xlim(np.min(xs), np.max(xs))
        ax.set_ylim(np.min(ys), np.max(ys))

        return p


class SamplingNadir(Sampling):
    """
    Construct a measurement geometry that samples the sensor at a variable horizontal and vertical resolutions
    depending on the tangent altitude.

    Parameters
    ----------
    sensor: SpectralImager
    optical_axis: OpticalGeometry

    Examples
    --------
    >>> sensor = SpectralImager(...)
    >>> opt_geom = OpticalGeometry(...)
    >>> samples = Sampling(sensor, opt_geom)
    >>> samples.along_track_resolution_deg = 1.0  # increase resolution at the tropopause to 1 km
    >>> samples.cross_track_resolution_deg = 1.0  # set tropopause altitude to 12 km
    >>> model_geometry = samples.meas_geom()
    >>> samples.set_weights(sensor)  # The LineShape weights in the sensor need to be set before performing a radiance calculation
    """

    def __init__(self, sensor: SpectralImager, optical_axis: OpticalGeometry):

        super().__init__(sensor, optical_axis)
        self.along_track_resolution_deg = 1
        self.cross_track_resolution_deg = 1
        self.truncate_edges = True

    def meas_geom(self):

        if self._meas_geom is not None:
            return self._meas_geom

        hfov = self.sensor.horizontal_fov / 2 * np.pi / 180
        vfov = self.sensor.vertical_fov / 2 * np.pi / 180

        na = int(self.sensor.vertical_fov / self.along_track_resolution_deg)
        if na % 2 == 0:
            na += 1
        if na < 3:
            na = 3
        nc = int(self.sensor.horizontal_fov / self.cross_track_resolution_deg)
        if nc % 2 == 0:
            nc += 1
        if nc < 3:
            nc = 3
        vert_angles = np.linspace(-vfov, vfov, na)
        h_angles = np.linspace(-hfov, hfov, nc)

        lines_of_sight = []
        for vidx, vert_angle in enumerate(vert_angles):
            hor = np.cross(self._optical_axis.look_vector, self._optical_axis.local_up)
            hor /= np.linalg.norm(hor)
            vlos = rotation_matrix(hor, vert_angle) @ self._optical_axis.look_vector
            h_width = 2 * hfov / (len(h_angles) - 1)

            if vidx == 0:
                vert_width_below = (vert_angles[1] - vert_angles[0]) / 2
                vert_width_above = (vert_angles[1] - vert_angles[0]) / 2
            elif vidx == len(vert_angles) - 1:
                vert_width_below = (vert_angles[-1] - vert_angles[-2]) / 2
                vert_width_above = (vert_angles[-1] - vert_angles[-2]) / 2
            else:
                vert_width_below = (vert_angles[vidx] - vert_angles[vidx - 1]) / 2
                vert_width_above = (vert_angles[vidx + 1] - vert_angles[vidx]) / 2

            for h_angle in h_angles:
                los = rotation_matrix(self._optical_axis.local_up, h_angle) @ vlos
                lines_of_sight.append(
                    sk.LineOfSight(
                        mjd=self._optical_axis.mjd,
                        observer=self._optical_axis.observer,
                        look_vector=los,
                    )
                )
                self.angles.append((h_angle, vert_angle))

                left = h_angle - h_width / 2
                right = h_angle + h_width / 2
                upper = vert_angle + vert_width_above
                lower = vert_angle - vert_width_below

                if self.truncate_edges:
                    left = -hfov if left < -hfov else left
                    right = hfov if right > hfov else right
                    upper = vfov if upper > vfov else upper
                    lower = -vfov if lower < -vfov else lower

                self.edges.append(
                    [
                        (left, lower),
                        (left, upper),
                        (right, upper),
                        (right, lower),
                        (left, lower),
                    ]
                )
                self.widths.append(np.abs(right - left))
                self.heights.append(np.abs(upper - lower))

        self.widths = np.array(self.widths)
        self.heights = np.array(self.heights)
        geom = sk.Geometry()
        geom.lines_of_sight = lines_of_sight
        self._meas_geom = geom
        return geom

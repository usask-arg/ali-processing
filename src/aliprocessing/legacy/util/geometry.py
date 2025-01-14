from __future__ import annotations

import numpy as np
import sasktran as sk
from skretrieval.core import OpticalGeometry


def optical_axis_from_geometry(geometry: sk.Geometry()):
    """
    Given a geometry object compute the optical geometry needed by an imager. This will be replaced by an instance
    of a `Platform` class in the future
    """
    mjd = geometry.lines_of_sight[0].mjd
    observer = geometry.lines_of_sight[0].observer
    look_vector = geometry.lines_of_sight[0].look_vector

    geodetic = sk.Geodetic()
    geodetic.from_xyz(observer)
    local_up = geodetic.local_up
    yaxis = np.cross(local_up, look_vector)
    yaxis /= np.linalg.norm(yaxis)
    obs_up = np.cross(look_vector, yaxis)
    obs_up /= np.linalg.norm(obs_up)

    return OpticalGeometry(observer, look_vector, obs_up, mjd)


def rotate(v, k, theta):
    """
    Rotate a vector v about the axis k
    """
    theta *= np.pi / 180.0
    return (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1 - np.cos(theta))
    )

"""The L1B Stokes convention matches what L2 gets from sasktran2.

L2 passes the L1B relative_solar_azimuth_angle to sasktran2.TangentAltitudeSolar
and models radiance in StokesBasis.Observer. For single Rayleigh scattering the
polarization is perpendicular to the scattering plane, so its angle in the image,
from the vertical towards the observer's right, is known independently of
sasktran2. If either the azimuth or the U sign convention disagreed, sasktran2
would put that angle on the wrong side of the vertical.
"""

from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk

from aliprocessing.geometry import InstrumentFrame, sun_direction, tangent_geometry


def _single_scatter_qu(
    tangent_altitude, relative_azimuth_deg, observer_altitude, sza_deg
):
    config = sk.Config()
    config.num_stokes = 3
    config.stokes_basis = sk.StokesBasis.Observer
    config.multiple_scatter_source = sk.MultipleScatterSource.NoSource
    config.single_scatter_source = sk.SingleScatterSource.Exact
    cos_sza = np.cos(np.deg2rad(sza_deg))
    geometry = sk.Geometry1D(
        cos_sza=cos_sza,
        solar_azimuth=0.0,
        earth_radius_m=6372000.0,
        altitude_grid_m=np.arange(0.0, 100001.0, 500.0),
        interpolation_method=sk.InterpolationMethod.LinearInterpolation,
        geometry_type=sk.GeometryType.Spherical,
    )
    viewing = sk.ViewingGeometry()
    viewing.add_ray(
        sk.TangentAltitudeSolar(
            tangent_altitude,
            np.deg2rad(relative_azimuth_deg),
            observer_altitude,
            cos_sza,
        )
    )
    atmosphere = sk.Atmosphere(geometry, config, wavelengths_nm=np.array([745.0]))
    sk.climatology.us76.add_us76_standard_atmosphere(atmosphere)
    atmosphere["rayleigh"] = sk.constituent.Rayleigh()
    radiance = (
        sk.Engine(config, geometry, viewing)
        .calculate_radiance(atmosphere)["radiance"]
        .isel(wavelength=0, los=0)
        .to_numpy()
    )
    return radiance[1] / radiance[0], radiance[2] / radiance[0]


@pytest.mark.parametrize(
    ("boresight_azimuth", "time"),
    [
        # Looking east before local noon: the sun is to the right (south)
        (90.0, "2026-06-21T16:00"),
        # Looking west before local noon: the sun is to the left
        (270.0, "2026-06-21T16:00"),
        # Looking north in the afternoon: the sun is behind and to the right
        (0.0, "2026-06-21T21:00"),
    ],
)
def test_polarization_angle_matches_sasktran2(boresight_azimuth, time):
    frame = InstrumentFrame.from_attitude(
        40.0, -100.0, 500e3, boresight_azimuth, -21.0, 0.0
    )
    sun = sun_direction(np.datetime64(time))
    point = tangent_geometry(frame.observer, frame.boresight[np.newaxis], sun)

    # Prediction: E-field perpendicular to the scattering plane, in the image
    # frame of the instrument (up and right axes at zero roll)
    field = np.cross(sun, frame.boresight)
    predicted_angle = np.arctan2(field @ frame.right, field @ frame.up)

    q, u = _single_scatter_qu(
        float(point.altitude[0]),
        float(point.relative_solar_azimuth_angle[0]),
        500e3,
        float(point.solar_zenith_angle[0]),
    )
    sasktran_angle = 0.5 * np.arctan2(u, q)

    difference = np.angle(np.exp(2j * (sasktran_angle - predicted_angle))) / 2
    assert abs(np.degrees(difference)) < 1.0
    # The test is only meaningful when U is clearly non-zero
    assert abs(u) > 0.05

from __future__ import annotations

import numpy as np
import pytest
import sasktran2 as sk

from aliprocessing.geometry import (
    InstrumentFrame,
    LocalSphere,
    boresight_elevation_for_tangent_altitude,
    look_at_tangent_altitude,
    sun_direction,
    tangent_geometry,
)


@pytest.fixture
def frame():
    return InstrumentFrame.from_attitude(52.0, -106.0, 500e3, 90.0, -21.3, 0.0)


def _exact_altitude(observer, look):
    geodetic = sk.WGS84()
    geodetic.from_tangent_point(observer, look)
    return geodetic.altitude


def test_axes_are_orthonormal(frame):
    axes = np.stack([frame.boresight, frame.up, frame.right])
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)


def test_right_axis_is_to_the_observers_right(frame):
    # Looking east, the observer's right is south
    south = sk.WGS84()
    south.from_lat_lon_alt(52.0, -106.0, 500e3)
    assert frame.right @ np.asarray(south.local_south) > 0.99


def test_positive_roll_turns_up_towards_right(frame):
    rolled = InstrumentFrame.from_attitude(52.0, -106.0, 500e3, 90.0, -21.3, 90.0)
    np.testing.assert_allclose(rolled.up, frame.right, atol=1e-12)
    np.testing.assert_allclose(rolled.boresight, frame.boresight, atol=1e-12)


@pytest.mark.parametrize(
    ("azimuth", "elevation"), [(0, 0), (2, 0), (-2, 0), (0, 0.6), (2, 0.6), (-2, -0.6)]
)
def test_local_sphere_matches_wgs84(frame, azimuth, elevation):
    sphere = LocalSphere.at_tangent_point(frame.observer, frame.boresight)
    look = frame.look_vectors(np.array(azimuth, float), np.array(elevation, float))
    approximate = sphere.tangent_altitude(frame.observer, look)
    assert abs(approximate - _exact_altitude(frame.observer, look)) < 5.0


def test_tangent_altitude_rises_along_a_row(frame):
    sphere = LocalSphere.at_tangent_point(frame.observer, frame.boresight)
    azimuth = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    altitude = sphere.tangent_altitude(
        frame.observer, frame.look_vectors(azimuth, np.zeros(5))
    )
    assert altitude[2] == altitude.min()
    # About 0.6 km higher at 2 degrees from a 500 km orbit
    assert 400 < altitude[0] - altitude[2] < 800


def test_look_at_tangent_altitude(frame):
    sphere = LocalSphere.at_tangent_point(frame.observer, frame.boresight)
    targets = np.array([0.0, 25e3, 50e3])
    looks = look_at_tangent_altitude(frame, sphere, 1.5, targets)
    exact = [_exact_altitude(frame.observer, look) for look in looks]
    np.testing.assert_allclose(exact, targets, atol=5.0)


def test_look_at_unreachable_altitude_raises(frame):
    sphere = LocalSphere.at_tangent_point(frame.observer, frame.boresight)
    with pytest.raises(ValueError, match="outside"):
        look_at_tangent_altitude(frame, sphere, 0.0, np.array([400e3]))


def test_boresight_elevation_for_tangent_altitude():
    elevation = boresight_elevation_for_tangent_altitude(40.0, -100.0, 500e3, 90.0, 25e3)
    frame = InstrumentFrame.from_attitude(40.0, -100.0, 500e3, 90.0, elevation, 0.0)
    assert abs(_exact_altitude(frame.observer, frame.boresight) - 25e3) < 1.0


def _subsolar(time):
    sun = sun_direction(np.datetime64(time))
    return np.degrees(np.arcsin(sun[2])), np.degrees(np.arctan2(sun[1], sun[0]))


@pytest.mark.parametrize(
    ("time", "latitude", "longitude"),
    [
        # Solstices: declination ±23.44, equation of time about -1.8 and +1.9 min
        ("2026-06-21T12:00", 23.44, 0.45),
        ("2026-12-21T12:00", -23.44, -0.48),
        # March equinox at 14:46 UTC, equation of time about -7.4 min
        ("2026-03-20T14:46", 0.0, -39.6),
    ],
)
def test_subsolar_point(time, latitude, longitude):
    found_latitude, found_longitude = _subsolar(time)
    assert abs(found_latitude - latitude) < 0.05
    assert abs(found_longitude - longitude) < 0.1


def test_solar_angles_at_the_subsolar_point():
    # Observer above the subsolar point, looking north: the sun is behind it
    latitude, longitude = _subsolar("2026-06-21T12:00")
    frame = InstrumentFrame.from_attitude(latitude - 20, longitude, 500e3, 0.0, -21.0, 0.0)
    sun = sun_direction(np.datetime64("2026-06-21T12:00"))
    points = tangent_geometry(frame.observer, frame.boresight[np.newaxis], sun)
    assert abs(points.los_azimuth_angle[0] % 360) < 1.0 or points.los_azimuth_angle[0] > 359
    # Looking away from the sun is backscatter
    assert abs(points.relative_solar_azimuth_angle[0]) > 170
    assert 0 < points.solar_zenith_angle[0] < 90

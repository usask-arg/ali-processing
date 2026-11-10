"""Viewing and solar geometry shared by the forward model and L1A→L1B.

Conventions follow the calibration database model: a pixel looks along
normalize(b + tan(α)·x + tan(β)·y), where b is the boresight, y and x are the
instrument up and right axes after the roll, and α and β are the pixel's azimuth
and elevation field angles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sasktran2 as sk


@dataclass(frozen=True)
class InstrumentFrame:
    """Observer position and instrument axes for one frame, in ECEF metres."""

    observer: np.ndarray
    boresight: np.ndarray
    up: np.ndarray
    right: np.ndarray
    roll_rad: float

    @classmethod
    def from_attitude(
        cls,
        latitude: float,
        longitude: float,
        altitude: float,
        boresight_azimuth: float,
        boresight_elevation: float,
        boresight_roll: float,
    ) -> InstrumentFrame:
        """Build the frame from the L1A position and attitude (angles in degrees)."""
        geodetic = sk.WGS84()
        geodetic.from_lat_lon_alt(latitude, longitude, altitude)
        local_up = np.asarray(geodetic.local_up)
        north = -np.asarray(geodetic.local_south)
        east = -np.asarray(geodetic.local_west)

        azimuth, elevation, roll = np.deg2rad(
            [boresight_azimuth, boresight_elevation, boresight_roll]
        )
        boresight = np.cos(elevation) * (
            np.cos(azimuth) * north + np.sin(azimuth) * east
        ) + np.sin(elevation) * local_up

        # Unrolled axes: up in the vertical plane through the boresight, right
        # completing a right-handed (right, up, boresight) set as the observer sees it
        up0 = local_up - np.dot(local_up, boresight) * boresight
        up0 /= np.linalg.norm(up0)
        right0 = np.cross(boresight, up0)

        # Positive roll is clockwise looking along the boresight: up turns towards right
        up = np.cos(roll) * up0 + np.sin(roll) * right0
        right = np.cos(roll) * right0 - np.sin(roll) * up0
        return cls(
            observer=np.asarray(geodetic.location),
            boresight=boresight,
            up=up,
            right=right,
            roll_rad=float(roll),
        )

    def look_vectors(
        self, azimuth_angle: np.ndarray, elevation_angle: np.ndarray
    ) -> np.ndarray:
        """Unit look vectors for field angles in degrees; shape (..., 3)."""
        tan_azimuth = np.tan(np.deg2rad(azimuth_angle))[..., np.newaxis]
        tan_elevation = np.tan(np.deg2rad(elevation_angle))[..., np.newaxis]
        look = self.boresight + tan_azimuth * self.right + tan_elevation * self.up
        return look / np.linalg.norm(look, axis=-1, keepdims=True)


@dataclass(frozen=True)
class LocalSphere:
    """Osculating sphere to the WGS84 ellipsoid at a reference tangent point.

    Gives fast, vectorized tangent altitudes for lines of sight near the
    reference, which is all of one frame's field of view.
    """

    center: np.ndarray
    radius: float

    @classmethod
    def at_tangent_point(cls, observer: np.ndarray, look: np.ndarray) -> LocalSphere:
        geodetic = sk.WGS84()
        geodetic.from_tangent_point(observer, look)
        radius, offset = geodetic.osculating_spheroid()
        return cls(center=np.asarray(offset), radius=float(radius))

    def tangent_altitude(self, observer: np.ndarray, look: np.ndarray) -> np.ndarray:
        """Geometric tangent altitude of each look vector in ``look`` (..., 3)."""
        relative = observer - self.center
        distance = -(look @ relative)
        closest = relative + distance[..., np.newaxis] * look
        return np.linalg.norm(closest, axis=-1) - self.radius


@dataclass(frozen=True)
class TangentGeometry:
    """Geolocation and solar geometry at tangent points; angles in degrees."""

    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    solar_zenith_angle: np.ndarray
    relative_solar_azimuth_angle: np.ndarray
    los_azimuth_angle: np.ndarray


def tangent_geometry(
    observer: np.ndarray, looks: np.ndarray, sun: np.ndarray
) -> TangentGeometry:
    """Exact WGS84 tangent points of ``looks`` (n, 3) and the solar angles there."""
    fields = {name: np.empty(len(looks)) for name in TangentGeometry.__annotations__}
    geodetic = sk.WGS84()
    for i, look in enumerate(looks):
        geodetic.from_tangent_point(observer, look)
        up = np.asarray(geodetic.local_up)
        north = -np.asarray(geodetic.local_south)
        east = -np.asarray(geodetic.local_west)

        los_azimuth = np.degrees(np.arctan2(look @ east, look @ north))
        solar_azimuth = np.degrees(np.arctan2(sun @ east, sun @ north))
        fields["latitude"][i] = geodetic.latitude
        fields["longitude"][i] = geodetic.longitude
        fields["altitude"][i] = geodetic.altitude
        fields["solar_zenith_angle"][i] = np.degrees(np.arccos(np.clip(up @ sun, -1, 1)))
        fields["los_azimuth_angle"][i] = los_azimuth % 360.0
        fields["relative_solar_azimuth_angle"][i] = _wrap_180(solar_azimuth - los_azimuth)
    return TangentGeometry(**fields)


def look_at_tangent_altitude(
    frame: InstrumentFrame,
    sphere: LocalSphere,
    azimuth_angle: float,
    altitudes: np.ndarray,
    elevation_search_deg: float = 5.0,
    num_search: int = 4001,
) -> np.ndarray:
    """Look vectors at ``azimuth_angle`` whose tangent altitudes are ``altitudes``.

    Searches the image column at that azimuth field angle, within
    ±``elevation_search_deg`` of the boresight, using the local sphere. Returns an
    array of shape (len(altitudes), 3).
    """
    elevation = np.linspace(-elevation_search_deg, elevation_search_deg, num_search)
    looks = frame.look_vectors(np.full_like(elevation, azimuth_angle), elevation)
    tangent_altitude = sphere.tangent_altitude(frame.observer, looks)
    # Tangent altitude increases with elevation angle across the limb
    found = np.interp(altitudes, tangent_altitude, elevation, left=np.nan, right=np.nan)
    if np.any(np.isnan(found)):
        msg = f"altitudes {altitudes} are outside the searched elevation range"
        raise ValueError(msg)
    return frame.look_vectors(np.full_like(found, azimuth_angle), found)


def boresight_elevation_for_tangent_altitude(
    latitude: float,
    longitude: float,
    altitude: float,
    boresight_azimuth: float,
    tangent_altitude: float,
) -> float:
    """Boresight elevation (degrees) that puts the boresight at ``tangent_altitude``.

    Solved on the WGS84 ellipsoid for an observer at the given position looking
    along ``boresight_azimuth`` (degrees clockwise from north).
    """
    geodetic = sk.WGS84()
    geodetic.from_lat_lon_alt(latitude, longitude, altitude)
    observer = np.asarray(geodetic.location)
    up = np.asarray(geodetic.local_up)
    north = -np.asarray(geodetic.local_south)
    east = -np.asarray(geodetic.local_west)
    azimuth = np.deg2rad(boresight_azimuth)
    horizontal = np.cos(azimuth) * north + np.sin(azimuth) * east

    look = np.asarray(geodetic.from_tangent_altitude(tangent_altitude, observer, horizontal))
    return float(np.degrees(np.arcsin(look @ up)))


def sun_direction(time: np.datetime64) -> np.ndarray:
    """Unit vector from the Earth's centre towards the sun, in ECEF.

    Low-precision solar coordinates from the Astronomical Almanac, accurate to
    about 0.01° between 1950 and 2050. Parallax (under 0.003°) is ignored.
    """
    days = (np.datetime64(time, "ns") - np.datetime64("2000-01-01T12:00:00", "ns")) / (
        np.timedelta64(1, "D")
    )
    mean_longitude = np.deg2rad(280.460 + 0.9856474 * days)
    mean_anomaly = np.deg2rad(357.528 + 0.9856003 * days)
    ecliptic_longitude = (
        mean_longitude
        + np.deg2rad(1.915) * np.sin(mean_anomaly)
        + np.deg2rad(0.020) * np.sin(2 * mean_anomaly)
    )
    obliquity = np.deg2rad(23.439 - 0.0000004 * days)

    right_ascension = np.arctan2(
        np.cos(obliquity) * np.sin(ecliptic_longitude), np.cos(ecliptic_longitude)
    )
    declination = np.arcsin(np.sin(obliquity) * np.sin(ecliptic_longitude))
    sidereal_time = np.deg2rad(15.0 * (18.697374558 + 24.06570982441908 * days))

    hour_angle_of_greenwich = right_ascension - sidereal_time
    return np.array(
        [
            np.cos(declination) * np.cos(hour_angle_of_greenwich),
            np.cos(declination) * np.sin(hour_angle_of_greenwich),
            np.sin(declination),
        ]
    )


def _wrap_180(angle):
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0

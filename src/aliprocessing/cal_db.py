"""Calibration databases for the simple ALI instrument model.

``simulated_cal_db`` builds a calibration database from a ``SimpleInstrument``.
Every number in ``SimpleInstrument`` is a placeholder until the instrument is
characterized; they are chosen to give realistic signal levels and layout, not
to describe the flight hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from aliprocessing.schemas import apply_metadata, check

EARTH_RADIUS = 6371e3


@dataclass(frozen=True)
class SimpleInstrument:
    """Placeholder parameters of the simple instrument model.

    The three sub-images are stacked vertically on the detector, separated by
    ``gap_rows`` unused rows, and each covers ``tangent_altitude_coverage`` from
    ``orbit_altitude``. Pixels are square unless ``horizontal_ifov_deg`` is set.
    """

    # Detector layout
    rows_per_subimage: int = 320
    gap_rows: int = 32
    num_columns: int = 1280
    horizontal_ifov_deg: float | None = None

    # Viewing geometry used to size the field of view
    orbit_altitude: float = 500e3
    tangent_altitude_coverage: tuple[float, float] = (-1e3, 51e3)

    # Polarization
    analyzer_angles_deg: tuple[float, ...] = (-60.0, 0.0, 60.0)
    analyzer_efficiency: float = 0.995
    analyzer_transmission: float = 0.4

    # Spectral channels: Gaussian passbands
    center_wavelengths_nm: tuple[float, ...] = (
        745.0,
        820.0,
        870.0,
        1020.0,
        1230.0,
        1350.0,
        1550.0,
    )
    bandpass_fwhm_nm: float = 10.0
    peak_throughput: float = 0.5
    throughput_step_nm: float = 0.5

    # Radiometry
    aperture_diameter: float = 0.02
    vignetting: float = 0.15
    pixel_response_nonuniformity: float = 0.01

    # Detector (InGaAs-like)
    detector_gain: float = 3.7
    detector_read_noise: float = 40.0
    detector_full_well: float = 60e3
    detector_adc_bits: int = 14
    dark_offset: float = 1950.0
    dark_offset_spread: float = 20.0
    dark_current: float = 500.0
    dark_current_spread: float = 50.0

    # L1B output grid
    altitude_grid: tuple[float, float, float] = (0.0, 50e3, 250.0)
    reference_altitudes: tuple[float, ...] = (0.0, 12.5e3, 25e3, 37.5e3, 50e3)
    across_track_half_width_deg: float = 2.0
    num_across_track_bins: int = 10

    @property
    def num_rows(self) -> int:
        num_analyzers = len(self.analyzer_angles_deg)
        return (
            num_analyzers * self.rows_per_subimage + (num_analyzers - 1) * self.gap_rows
        )

    def depression_angle_deg(self, tangent_altitude: float) -> float:
        """Angle below the horizontal, on a spherical Earth, to a tangent altitude."""
        observer_radius = EARTH_RADIUS + self.orbit_altitude
        return float(
            np.degrees(np.arccos((EARTH_RADIUS + tangent_altitude) / observer_radius))
        )

    @property
    def vertical_ifov_deg(self) -> float:
        low, high = self.tangent_altitude_coverage
        span = self.depression_angle_deg(low) - self.depression_angle_deg(high)
        return span / self.rows_per_subimage

    @property
    def boresight_tangent_altitude(self) -> float:
        """Tangent altitude to point the boresight at to centre the coverage.

        Pass it to ``geometry.boresight_elevation_for_tangent_altitude``.
        """
        low, high = self.tangent_altitude_coverage
        middle_depression = 0.5 * (
            self.depression_angle_deg(low) + self.depression_angle_deg(high)
        )
        observer_radius = EARTH_RADIUS + self.orbit_altitude
        return float(
            observer_radius * np.cos(np.deg2rad(middle_depression)) - EARTH_RADIUS
        )

    @property
    def pixel_altitude_spacing(self) -> float:
        """Tangent-altitude step between rows near the middle of the coverage."""
        middle = 0.5 * sum(self.tangent_altitude_coverage)
        observer_radius = EARTH_RADIUS + self.orbit_altitude
        distance = np.sqrt(observer_radius**2 - (EARTH_RADIUS + middle) ** 2)
        return float(distance * np.deg2rad(self.vertical_ifov_deg))


def simulated_cal_db(
    instrument: SimpleInstrument | None = None, seed: int = 0
) -> xr.Dataset:
    """Calibration database for ``instrument``, with seeded pixel-to-pixel variation."""
    instrument = SimpleInstrument() if instrument is None else instrument
    rng = np.random.default_rng(seed)

    num_analyzers = len(instrument.analyzer_angles_deg)
    num_channels = len(instrument.center_wavelengths_nm)
    shape = (instrument.num_rows, instrument.num_columns)
    pixel_dims = ["detector_row", "detector_column"]
    channel_pixel_dims = ["spectral_channel", *pixel_dims]

    # Sub-image layout and field angles
    pixel_analyzer = np.full(shape, -1)
    row_in_subimage = np.full(instrument.num_rows, -1)
    for k in range(num_analyzers):
        start = k * (instrument.rows_per_subimage + instrument.gap_rows)
        rows = slice(start, start + instrument.rows_per_subimage)
        pixel_analyzer[rows] = k
        row_in_subimage[rows] = np.arange(instrument.rows_per_subimage)
    used = pixel_analyzer >= 0

    vertical_ifov = instrument.vertical_ifov_deg
    horizontal_ifov = instrument.horizontal_ifov_deg or vertical_ifov
    # Row 0 of each sub-image looks highest
    elevation_rows = (
        (instrument.rows_per_subimage - 1) / 2 - row_in_subimage
    ) * vertical_ifov
    azimuth_columns = (
        np.arange(instrument.num_columns) - (instrument.num_columns - 1) / 2
    ) * horizontal_ifov
    elevation = np.where(used, elevation_rows[:, np.newaxis], np.nan)
    azimuth = np.where(used, azimuth_columns, np.nan)

    # Polarization
    two_theta = np.deg2rad(2 * np.array(instrument.analyzer_angles_deg))
    k = np.where(used, pixel_analyzer, 0)
    q_sensitivity = np.where(
        used, instrument.analyzer_efficiency * np.cos(two_theta)[k], np.nan
    )
    u_sensitivity = np.where(
        used, instrument.analyzer_efficiency * np.sin(two_theta)[k], np.nan
    )

    # Radiometry: aperture times pixel solid angle, with vignetting and PRNU
    pixel_solid_angle = np.deg2rad(vertical_ifov) * np.deg2rad(horizontal_ifov)
    aperture_area = np.pi * (instrument.aperture_diameter / 2) ** 2
    radius = np.hypot(
        azimuth / np.nanmax(np.abs(azimuth_columns)),
        elevation / np.nanmax(np.abs(elevation_rows)),
    ) / np.sqrt(2)
    etendue = (
        aperture_area
        * pixel_solid_angle
        * (1 - instrument.vignetting * radius**2)
        * (1 + instrument.pixel_response_nonuniformity * rng.standard_normal(shape))
    )

    centers = np.array(instrument.center_wavelengths_nm)
    fwhm = instrument.bandpass_fwhm_nm
    wavelength = np.arange(
        centers.min() - 5 * fwhm,
        centers.max() + 5 * fwhm,
        instrument.throughput_step_nm,
    )
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    throughput = instrument.peak_throughput * np.exp(
        -0.5 * ((wavelength - centers[:, np.newaxis]) / sigma) ** 2
    )

    dark_offset = (
        instrument.dark_offset
        + instrument.dark_offset_spread * rng.standard_normal(shape)
    )
    dark_current = np.clip(
        instrument.dark_current
        + instrument.dark_current_spread * rng.standard_normal(shape),
        0,
        None,
    )

    low, high, step = instrument.altitude_grid
    altitude = np.arange(low, high + step / 2, step)
    half_width = instrument.across_track_half_width_deg
    kernel_half_width = max(step, instrument.pixel_altitude_spacing)

    def per_channel(values):
        return np.broadcast_to(values, (num_channels, *shape)).copy()

    cal_db = xr.Dataset(
        {
            "pixel_analyzer": (pixel_dims, pixel_analyzer),
            "q_sensitivity": (channel_pixel_dims, per_channel(q_sensitivity)),
            "u_sensitivity": (channel_pixel_dims, per_channel(u_sensitivity)),
            "analyzer_transmission": (
                ["spectral_channel", "analyzer"],
                np.full(
                    (num_channels, num_analyzers), instrument.analyzer_transmission
                ),
            ),
            "pixel_elevation_angle": (pixel_dims, elevation),
            "pixel_azimuth_angle": (pixel_dims, azimuth),
            "pixel_etendue": (pixel_dims, np.where(used, etendue, np.nan)),
            "spectral_throughput": (
                ["spectral_channel", "throughput_wavelength"],
                throughput,
            ),
            "center_wavelength": (["spectral_channel"], centers),
            "bandpass_fwhm": (["spectral_channel"], np.full(num_channels, fwhm)),
            "dark_offset": (pixel_dims, dark_offset),
            "dark_current": (pixel_dims, dark_current),
            "detector_gain": ((), instrument.detector_gain),
            "detector_read_noise": ((), instrument.detector_read_noise),
            "detector_full_well": ((), instrument.detector_full_well),
            "detector_adc_bits": ((), np.int64(instrument.detector_adc_bits)),
            "altitude_kernel_half_width": ((), kernel_half_width),
            "across_track_edge": (
                ["across_track_edge"],
                np.linspace(
                    -half_width, half_width, instrument.num_across_track_bins + 1
                ),
            ),
        },
        coords={
            "throughput_wavelength": wavelength,
            "altitude": altitude,
            "reference_altitude": np.array(instrument.reference_altitudes),
        },
        attrs={"valid_from": "2026-01-01T00:00:00Z"},
    )
    cal_db = apply_metadata(cal_db, "cal_db")
    check(cal_db, "cal_db")
    return cal_db

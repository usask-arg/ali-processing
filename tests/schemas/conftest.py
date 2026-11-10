"""Small datasets that conform to each schema.

The detector has 4 rows and 8 columns: three 2-column sub-images separated by
1-column gaps. Each sub-image has one column in each of the 2 across-track bins.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

NUM_ROWS = 4
NUM_COLUMNS = 8
PIXEL_ANALYZER_COLUMNS = np.array([0, 0, -1, 1, 1, -1, 2, 2])
ANALYZER_ANGLES = np.array([-60.0, 0.0, 60.0])
CENTER_WAVELENGTHS = np.array([745.0, 1020.0])
ALTITUDES = np.arange(10e3, 30001.0, 5e3)
REFERENCE_ALTITUDES = np.array([15e3, 25e3])
ACROSS_TRACK_EDGES = np.array([-1.0, 0.0, 1.0])

# L1A frames: two observation sets with a blank frame between them
FRAME_CHANNELS = np.array([0, 1, -1, 0, 1])
FRAME_SETS = np.array([0, 0, -1, 1, 1])

NUM_FRAMES = len(FRAME_CHANNELS)
NUM_CHANNELS = len(CENTER_WAVELENGTHS)
NUM_SETS = 2
NUM_BINS = len(ACROSS_TRACK_EDGES) - 1


def _var(dims, values, units=None):
    # Copy so that tests editing a fixture cannot change the module constants
    attrs = {} if units is None else {"units": units}
    return xr.Variable(dims, np.array(values, copy=True), attrs=attrs)


def _frame_times():
    start = np.datetime64("2026-11-04T12:00:00", "ns")
    return start + np.arange(NUM_FRAMES) * np.timedelta64(1, "s")


def _pixel_map(column_values, fill=np.nan):
    """Broadcast per-column values over rows, with ``fill`` in the gaps."""
    values = np.where(PIXEL_ANALYZER_COLUMNS >= 0, column_values, fill)
    return np.broadcast_to(values, (NUM_ROWS, NUM_COLUMNS)).copy()


@pytest.fixture
def l1a() -> xr.Dataset:
    image_dims = ["time", "detector_row", "detector_column"]
    shape = (NUM_FRAMES, NUM_ROWS, NUM_COLUMNS)
    per_frame = {
        "exposure_time": (0.05, "s"),
        "detector_temperature": (250.0, "K"),
        "spacecraft_latitude": (50.0, "degrees_north"),
        "spacecraft_longitude": (-106.0, "degrees_east"),
        "spacecraft_altitude": (500e3, "m"),
        "boresight_azimuth_angle": (90.0, "degree"),
        "boresight_elevation_angle": (-21.5, "degree"),
        "boresight_roll_angle": (0.0, "degree"),
    }
    data_vars = {
        "image": _var(image_dims, np.full(shape, 1000.0), "count"),
        "spectral_channel": _var(["time"], FRAME_CHANNELS, "1"),
        "observation_set": _var(["time"], FRAME_SETS, "1"),
        "num_averaged_exposures": _var(["time"], np.full(NUM_FRAMES, 2), "1"),
        "saturated": xr.Variable(image_dims, np.zeros(shape, dtype=bool)),
        **{
            name: _var(["time"], np.full(NUM_FRAMES, value), units)
            for name, (value, units) in per_frame.items()
        },
    }
    return xr.Dataset(
        data_vars, coords={"time": _frame_times()}, attrs={"schema_version": "0.1.0"}
    )


@pytest.fixture
def l1a_simulation() -> xr.Dataset:
    return xr.Dataset(
        {
            "image_noiseless": _var(
                ["time", "detector_row", "detector_column"],
                np.full((NUM_FRAMES, NUM_ROWS, NUM_COLUMNS), 1000.0),
                "count",
            ),
            "noise_enabled": xr.Variable([], True),
            "noise_seed": xr.Variable([], np.int64(7)),
        }
    )


@pytest.fixture
def l1b() -> xr.Dataset:
    cell_dims = ["observation_set", "across_track", "wavelength", "altitude"]
    cell_shape = (NUM_SETS, NUM_BINS, NUM_CHANNELS, len(ALTITUDES))
    geo_dims = ["observation_set", "across_track", "wavelength", "reference_altitude"]
    geo_shape = (NUM_SETS, NUM_BINS, NUM_CHANNELS, len(REFERENCE_ALTITUDES))
    frame_dims = ["observation_set", "wavelength"]
    frame_times = _frame_times()[FRAME_SETS >= 0].reshape(NUM_SETS, NUM_CHANNELS)

    radiance_units = "W m-2 sr-1 nm-1"
    q, u = 0.05, 0.02
    cells = {
        "radiance": (1e-3, radiance_units),
        "radiance_noise": (1e-5, radiance_units),
        "q": (q, "1"),
        "q_noise": (1e-3, "1"),
        "u": (u, "1"),
        "u_noise": (1e-3, "1"),
        "dolp": (np.hypot(q, u), "1"),
        "dolp_noise": (1e-3, "1"),
        "q_u_correlation": (0.0, "1"),
        "saturated_fraction": (0.0, "1"),
    }
    geometry = {
        "tangent_latitude": (52.0, "degrees_north"),
        "tangent_longitude": (-100.0, "degrees_east"),
        "solar_zenith_angle": (60.0, "degree"),
        "relative_solar_azimuth_angle": (30.0, "degree"),
        "los_azimuth_angle": (90.0, "degree"),
    }
    spacecraft = {
        "spacecraft_latitude": (50.0, "degrees_north"),
        "spacecraft_longitude": (-106.0, "degrees_east"),
        "spacecraft_altitude": (500e3, "m"),
    }
    data_vars = {
        "bandpass_fwhm": _var(["wavelength"], np.full(NUM_CHANNELS, 10.0), "nm"),
        "across_track_angle": _var(["across_track"], np.array([-0.5, 0.5]), "degree"),
        "time": xr.Variable(frame_dims, frame_times),
        **{
            name: _var(cell_dims, np.full(cell_shape, value), units)
            for name, (value, units) in cells.items()
        },
        **{
            name: _var(geo_dims, np.full(geo_shape, value), units)
            for name, (value, units) in geometry.items()
        },
        **{
            name: _var(frame_dims, np.full((NUM_SETS, NUM_CHANNELS), value), units)
            for name, (value, units) in spacecraft.items()
        },
    }
    return xr.Dataset(
        data_vars,
        coords={
            "wavelength": _var(["wavelength"], CENTER_WAVELENGTHS, "nm"),
            "altitude": _var(["altitude"], ALTITUDES, "m"),
            "reference_altitude": _var(
                ["reference_altitude"], REFERENCE_ALTITUDES, "m"
            ),
        },
        attrs={"schema_version": "0.1.0", "radiance_noise_model": "detector"},
    )


@pytest.fixture
def cal_db() -> xr.Dataset:
    pixel_dims = ["detector_row", "detector_column"]
    channel_pixel_dims = ["spectral_channel", *pixel_dims]

    analyzer = _pixel_map(PIXEL_ANALYZER_COLUMNS, fill=-1).astype(int)
    two_theta = np.deg2rad(2 * ANALYZER_ANGLES)
    used = analyzer >= 0
    q_sensitivity = np.where(used, np.cos(two_theta)[analyzer], np.nan)
    u_sensitivity = np.where(used, np.sin(two_theta)[analyzer], np.nan)

    # Each sub-image's two columns sit in the two across-track bins
    azimuth_columns = np.tile([-0.5, 0.5, np.nan], 3)[:NUM_COLUMNS]
    elevation_rows = np.array([-0.3, -0.1, 0.1, 0.3])
    elevation = np.where(used, elevation_rows[:, np.newaxis], np.nan)

    throughput_wavelength = np.arange(700.0, 1101.0, 5.0)
    throughput = np.where(
        np.abs(throughput_wavelength - CENTER_WAVELENGTHS[:, np.newaxis]) <= 10.0,
        0.5,
        0.0,
    )

    data_vars = {
        "pixel_analyzer": _var(pixel_dims, analyzer, "1"),
        "q_sensitivity": _var(
            channel_pixel_dims,
            np.broadcast_to(q_sensitivity, (NUM_CHANNELS, *analyzer.shape)).copy(),
            "1",
        ),
        "u_sensitivity": _var(
            channel_pixel_dims,
            np.broadcast_to(u_sensitivity, (NUM_CHANNELS, *analyzer.shape)).copy(),
            "1",
        ),
        "analyzer_transmission": _var(
            ["spectral_channel", "analyzer"], np.full((NUM_CHANNELS, 3), 0.4), "1"
        ),
        "pixel_elevation_angle": _var(pixel_dims, elevation, "degree"),
        "pixel_azimuth_angle": _var(pixel_dims, _pixel_map(azimuth_columns), "degree"),
        "pixel_etendue": _var(pixel_dims, _pixel_map(1e-10), "m2 sr"),
        "spectral_throughput": _var(
            ["spectral_channel", "throughput_wavelength"], throughput, "1"
        ),
        "center_wavelength": _var(["spectral_channel"], CENTER_WAVELENGTHS, "nm"),
        "bandpass_fwhm": _var(["spectral_channel"], np.full(NUM_CHANNELS, 10.0), "nm"),
        "dark_offset": _var(pixel_dims, np.full(analyzer.shape, 100.0), "count"),
        "dark_current": _var(pixel_dims, np.full(analyzer.shape, 1.0), "count s-1"),
        "detector_gain": _var([], 2.0, "1"),
        "detector_read_noise": _var([], 10.0, "1"),
        "detector_full_well": _var([], 1e5, "1"),
        "detector_adc_bits": _var([], np.int64(14), "1"),
        "altitude_kernel_half_width": _var([], 5e3, "m"),
        "across_track_edge": _var(["across_track_edge"], ACROSS_TRACK_EDGES, "degree"),
    }
    return xr.Dataset(
        data_vars,
        coords={
            "throughput_wavelength": _var(
                ["throughput_wavelength"], throughput_wavelength, "nm"
            ),
            "altitude": _var(["altitude"], ALTITUDES, "m"),
            "reference_altitude": _var(
                ["reference_altitude"], REFERENCE_ALTITUDES, "m"
            ),
        },
        attrs={"schema_version": "0.1.0", "valid_from": "2026-11-04T00:00:00Z"},
    )

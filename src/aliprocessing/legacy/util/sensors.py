from __future__ import annotations

import numpy as np
from ali_processing.legacy.instrument.sensor import ALISensor, ALISensorER2
from ali_processing.legacy.instrument.sensor_2channel import ALISensorDualChannel
from skretrieval.core.lineshape import Rectangle


def _simulation_sensor(
    wavelength_nm: np.ndarray | None = None,
    image_horiz_fov=5,
    image_vert_fov=1.1,
    num_columns=1,
    num_rows=512,
    simulated_pixel_averaging=0,
    noisy=True,
    dual_channel=True,
    vertical_resolution: float | None = None,
    straylight=0.0,
    er2: bool = False,
):

    if wavelength_nm is None:
        wavelength_nm = np.array([750.0])

    if vertical_resolution is not None:
        # vertical_resolution = Gaussian(fwhm=vertical_resolution, max_stdev=50, mode='linear')
        vertical_resolution = Rectangle(width=vertical_resolution, mode="linear")

    if dual_channel:
        sensor = ALISensorDualChannel(
            wavelength_nm=wavelength_nm,
            image_horiz_fov=image_horiz_fov,
            image_vert_fov=image_vert_fov,
            pixel_vert_fov=vertical_resolution,
            num_columns=num_columns,
            num_rows=num_rows,
            straylight=straylight,
        )
        sensor.simulate_pixel_averaging = simulated_pixel_averaging

    else:
        if er2:
            sensor = ALISensorER2(
                wavelength_nm=wavelength_nm,
                image_horiz_fov=image_horiz_fov,
                image_vert_fov=image_vert_fov,
                pixel_vert_fov=vertical_resolution,
                num_columns=num_columns,
                num_rows=num_rows,
                straylight=straylight,
            )
        else:
            sensor = ALISensor(
                wavelength_nm=wavelength_nm,
                image_horiz_fov=image_horiz_fov,
                image_vert_fov=image_vert_fov,
                pixel_vert_fov=vertical_resolution,
                num_columns=num_columns,
                num_rows=num_rows,
                straylight=straylight,
            )
        sensor._simulate_pixel_averaging = simulated_pixel_averaging

    sensor.add_adc = noisy
    sensor.add_noise = noisy
    sensor.add_dark_current = noisy
    return sensor


def simulation_sensor(
    wavelength_nm: np.ndarray | None = None,
    image_horiz_fov=5,
    image_vert_fov=1.1,
    num_columns=1,
    num_rows=512,
    simulated_pixel_averaging=0,
    noisy=True,
    dual_channel=True,
    split_wavelengths=False,
    vertical_resolution: float | None = None,
    straylight=0.0,
    er2: bool = False,
):
    if wavelength_nm is None:
        wavelength_nm = np.array([750.0])

    if split_wavelengths and len(wavelength_nm) > 1:
        sensors = []
        for wavel in wavelength_nm:
            sensors.append(
                _simulation_sensor(
                    np.array([wavel]),
                    image_horiz_fov,
                    image_vert_fov,
                    num_columns,
                    num_rows,
                    simulated_pixel_averaging=simulated_pixel_averaging,
                    noisy=noisy,
                    dual_channel=dual_channel,
                    vertical_resolution=vertical_resolution,
                    straylight=straylight,
                    er2=er2,
                )
            )
        return sensors

    return _simulation_sensor(
        wavelength_nm,
        image_horiz_fov,
        image_vert_fov,
        num_columns,
        num_rows,
        simulated_pixel_averaging=simulated_pixel_averaging,
        noisy=noisy,
        dual_channel=dual_channel,
        vertical_resolution=vertical_resolution,
        straylight=straylight,
        er2=er2,
    )


def retrieval_sensor(
    wavelength_nm: np.ndarray | None = None,
    image_horiz_fov=5,
    image_vert_fov=1.1,
    num_columns=1,
    num_rows=512,
    simulated_pixel_averaging=0,
    split_wavelengths=False,
    vertical_resolution: float | None = None,
    er2: bool = False,
):

    if wavelength_nm is None:
        wavelength_nm = np.array([750.0])

    return simulation_sensor(
        wavelength_nm,
        image_horiz_fov,
        image_vert_fov,
        num_columns,
        num_rows,
        simulated_pixel_averaging,
        noisy=False,
        dual_channel=True,
        split_wavelengths=split_wavelengths,
        vertical_resolution=vertical_resolution,
        er2=er2,
    )

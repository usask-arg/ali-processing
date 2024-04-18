from __future__ import annotations

import warnings

import numpy as np
from skretrieval.core.lineshape import LineShape


class ALILineShape(LineShape):

    def __init__(self, normalize=True):
        self.normalize = normalize

    def area(self, mean: float, num_periods=10, sampling=0.01):

        width = num_periods * np.pi / self._sync_freq(mean)
        x_values = np.concatenate(
            [
                np.arange(mean, mean + width, sampling),
                np.arange(mean, mean - width, -sampling),
            ]
        )
        x_values = np.sort(np.unique(x_values))
        peak = self.get_lineshape(mean, mean)
        return np.trapz(self.get_lineshape(mean, x_values) / peak, x_values)

    def integration_weights(
        self, mean: float, available_samples: np.ndarray, normalize=None
    ):
        # interpolate the line shape to the available samples

        if normalize is None:
            normalize = self.normalize

        line_shape_interp = self.get_lineshape(mean, available_samples)

        if not normalize:
            msg = "UserLineShape currently only supports normalized line shapes"
            raise ValueError(msg)

        line_shape_interp /= np.sum(line_shape_interp)

        return line_shape_interp

    def bounds(self):
        return 0.0, 2000.0

    def sample_bounds(self, mean, num_periods):
        width = float(num_periods * np.pi / self._sync_freq(mean))
        return mean - width, mean + width

    def get_lineshape(self, wavelength, samples):
        peak = self._gaussian(wavelength, wavelength) + self._sync(
            wavelength, wavelength
        )
        ls = self._gaussian(wavelength, samples) + self._sync(wavelength, samples)
        return ls / peak

    def _gaussian_std(self, wavelength):
        return wavelength * 0.01050691 - 1.721324491818138

    def _gaussian_scale(self, wavelength):
        return 1.0

    def _gaussian(self, wavelength, samples):
        sigma = self._gaussian_std(wavelength)
        scale = self._gaussian_scale(wavelength)
        return (
            scale
            / (sigma * np.sqrt(2))
            * np.exp((-((wavelength - samples) ** 2)) / (2 * sigma**2))
        )

    def _sync_freq(self, wavelength):
        if wavelength < 890.0:
            p = np.array([-0.00379218, 2.53151829])
        else:
            p = np.array([-0.00199282, 1.3438562])
        return np.exp(np.polyval(p, [wavelength]))

    def _sync(self, wavelength, samples):

        sync_freq = self._sync_freq(wavelength)
        sync_area = np.pi / sync_freq
        scale = (
            10 * self._gaussian_scale(wavelength) / sync_area
        )  # set area of sync to 10x area gaussian

        # we will fix the divide by zero in a second
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vals = (
                scale
                * (
                    np.sin((wavelength - samples) * sync_freq)
                    / (sync_freq * (wavelength - samples))
                )
                ** 2
            )

        vals[wavelength == samples] = scale
        return vals


class ALIER2LineShape(ALILineShape):
    """
    Fitting parameters found from ER2_aotf_calibration.nc data
    """

    def _gaussian(self, wavelength, samples):
        sigma = self._gaussian_std(wavelength)
        scale = self._gaussian_scale(wavelength)
        wavel = self._gaussian_wavelength(wavelength)
        return (
            scale
            / (sigma * np.sqrt(2))
            * np.exp((-((wavel - samples) ** 2)) / (2 * sigma**2))
        )

    def _gaussian_std(self, wavelength):
        return np.array([10.0])

    def _gaussian_scale(self, wavelength):
        return np.array([1.0])

    def _gaussian_wavelength(self, wavelength):
        return (
            np.array(
                [
                    -4.17892120e-05 * wavelength**2
                    + 9.70685681e-02 * wavelength
                    + -4.97850467e01
                ]
            )
            + wavelength
        )

    def _sync_freq(self, wavelength):
        return np.array(
            [1.35668884e-06 * wavelength**2 - 3.96345088e-03 * wavelength + 3.11818513]
        )

    def _sync_scale(self, wavelength):
        return self._gaussian_scale(wavelength) * (
            1.31978493e-05 * wavelength**2
            + -2.96125277e-02 * wavelength
            + 1.70149850e01
        )

    def _sync(self, wavelength, samples):

        sync_freq = self._sync_freq(wavelength)
        scale = self._sync_scale(wavelength)

        # we will fix the divide by zero in a second
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            vals = (
                scale
                * (
                    np.sin((wavelength - samples) * sync_freq)
                    / (sync_freq * (wavelength - samples))
                )
                ** 2
            )

        vals[wavelength == samples] = scale
        return vals

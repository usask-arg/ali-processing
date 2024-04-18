from __future__ import annotations

import numpy as np
from skcomponents.optics import Filter
from skcomponents.processing import (
    ADC,
    GaussianNoise,
    PoissonCounting,
    SignalProcessor,
    SimpleDarkCurrent,
)


class Sensor(SignalProcessor):

    def __init__(
        self,
        exposure_time=1,
        ccd_temperature=285,
        ccd_readout_noise=50,
        max_well_depth=1_000_000,
        adc_bits=12,
        num_columns=512,
        num_rows=512,
    ):
        """
        Base class to handle readout electronics of a typical CCD or CMOS sensor.

        Parameters
        ----------
        exposure_time: float
            Image exposure time in seconds.
        ccd_temperature: float
            Sensor temperature in kelvin.
        ccd_readout_noise: float
            Readout noise in electrons.
        max_well_depth: int
            Well depth in electrons.
        adc_bits: int
            Number of bits in the analog-to-digital converter.
        num_columns: int
            Number of columns in the sensor.
        num_rows: int
            Number of rows in the sensor.
        """
        self._exposure_time = exposure_time
        self._ccd_temperature = ccd_temperature
        self._ccd_readout_noise = ccd_readout_noise
        self._max_well_depth = max_well_depth
        self._adc_bits = adc_bits
        self.add_noise = True
        self.add_dark_current = True
        self.dark_current_temps = np.array([322.0, 298.0, 276.0])
        self.dark_current_values = np.array([10e-15, 1e-15, 0.1e-15])
        self.add_adc = True
        self.gain = 0
        self.num_rows = num_columns
        self.num_columns = num_rows
        self.processors = self._create_post_processors()

    @property
    def adc(self) -> ADC:
        """
        Return the analogue to digital processor

        Returns
        -------
            SignalProcessor
        """
        return self.processors["adc"]

    @property
    def dark_current(self) -> SimpleDarkCurrent:
        """
        Return the dark current processor

        Returns
        -------
            SignalProcessor
        """
        return self.processors["dark-current"]

    @property
    def readout_noise(self) -> GaussianNoise:
        """
        Return the readout noise processor

        Returns
        -------
            SignalProcessor
        """
        return self.processors["readout-noise"]

    @property
    def shot_noise(self) -> PoissonCounting:
        """
        Return the shot noise processor

        Returns
        -------
            SignalProcessor
        """
        return self.processors["shot-noise"]

    @property
    def ccd_temperature(self) -> float:
        """
        Temperature of the CCD in kelvin. Used for dark current calculations.

        Returns
        -------
            float
        """
        return self._ccd_temperature

    @ccd_temperature.setter
    def ccd_temperature(self, value: float):
        self._ccd_temperature = value
        if self.add_dark_current:
            self.dark_current.sensor_temperature = value

    @property
    def exposure_time(self) -> float:
        """
        exposure time of the image. Used for dark current calculations.

        Returns
        -------
            float
        """
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value: float):
        self._exposure_time = value
        if self.add_dark_current:
            self.dark_current.exposure_time = value

    def quantum_efficiency(self) -> Filter:
        return Filter(np.array([0, 1e6]), np.array([1, 1]))

    def process_signal(self, radiance):
        self.processors = self._create_post_processors()  # update values

        for name, processor in self.processors.items():
            radiance = processor.process_signal(radiance)

        return radiance

    def noise_estimate(self, signal: np.ndarray) -> np.ndarray:
        self.processors = self._create_post_processors()  # update values

        sig = np.ones_like(signal) * 0.0
        if self.add_noise:
            sig += self.shot_noise.noise_estimate(signal) ** 2
            sig += self.readout_noise.noise_estimate(1) ** 2
        if self.add_dark_current:
            dc = self.dark_current.noise_estimate(1) ** 2
            if type(dc) is np.ndarray:
                if dc.size > 1:  # catch case of 1d array
                    sig = sig + dc[:, np.newaxis, np.newaxis]
                else:
                    sig += dc
            else:
                sig += dc

        if self.add_adc:
            sig += self.adc.noise_estimate(1) ** 2
            if not self.adc.rescale:
                return np.sqrt(sig) / self.adc.adu
        return np.sqrt(sig)

    def _create_post_processors(self):
        processors = {}

        if self.add_noise:
            processors["shot-noise"] = PoissonCounting()

        if self.add_dark_current:
            processors["dark-current"] = SimpleDarkCurrent(
                dark_current=self.dark_current_values,
                temperature=self.dark_current_temps,
                exposure_time=self._exposure_time,
                sensor_temperature=self._ccd_temperature,
            )

        if self.add_noise:
            processors["readout-noise"] = GaussianNoise(
                noise_level=self._ccd_readout_noise, relative=False
            )

        if self.add_adc:
            processors["adc"] = ADC(
                bits=self._adc_bits,
                min_value=0,
                max_value=self._max_well_depth,
                rescale=False,
            )

        return processors

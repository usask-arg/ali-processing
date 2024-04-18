from __future__ import annotations

from pathlib import Path

import numpy as np
from skcomponents.optics import Filter, Mirror


class Mirror12AOI(Filter):

    def __init__(self):

        wavelength = np.arange(400.0, 1650.0, 50.0)
        transmission = [
            0.9097469,
            0.9876272,
            0.9894966,
            0.9940575,
            0.9863379,
            0.9908437,
            0.9823487,
            0.9821165,
            0.9678582,
            0.9675521,
            0.967152,
            0.9679117,
            0.9695703,
            0.9709087,
            0.9724537,
            0.9740802,
            0.9757646,
            0.9769103,
            0.9783501,
            0.9791632,
            0.980234,
            0.9810411,
            0.9818643,
            0.9817832,
            0.98244,
        ]

        super().__init__(wavelength_nm=wavelength, transmission=transmission)


class Mirror45AOI(Filter):

    def __init__(self):
        wavelength = np.arange(400.0, 1650.0, 50.0)
        transmission = [
            0.8871807,
            0.9573861,
            0.9655209,
            0.9634123,
            0.9575384,
            0.9572349,
            0.9539356,
            0.9526094,
            0.9553123,
            0.9611861,
            0.9660538,
            0.9684723,
            0.9701422,
            0.9724538,
            0.9740065,
            0.9756533,
            0.9768021,
            0.9780677,
            0.9791726,
            0.9798208,
            0.980691,
            0.9813352,
            0.9817338,
            0.9821452,
            0.9823722,
        ]

        super().__init__(wavelength_nm=wavelength, transmission=transmission)


class CodeVLens(Filter):

    def __init__(self):
        wavelength = np.arange(400.0, 1650.0, 50.0)
        transmission = [
            0.7127,
            0.7415,
            0.7704,
            0.79925,
            0.8281,
            0.85695,
            0.8858,
            0.90465,
            0.9235,
            0.93235,
            0.9412,
            0.9501,
            0.959,
            0.9541,
            0.9492,
            0.9443,
            0.9394,
            0.93355,
            0.9277,
            0.92085,
            0.914,
            0.9071,
            0.9002,
            0.89335,
            0.8865,
        ]

        super().__init__(wavelength_nm=wavelength, transmission=transmission)


class AluminumMirror(Filter):

    def __init__(self):

        filename = (
            Path(__file__).resolve().parent / "data" / "aluminum_coating_thorlab.txt"
        )
        data = np.loadtxt(filename, skiprows=1)
        wavelength = data[:, 0] * 1000
        transmission = data[:, 1] / 100
        super().__init__(wavelength_nm=wavelength, transmission=transmission)


class GoldMirror(Mirror):

    def __init__(self):

        filename = (
            Path(__file__).resolve().parent / "data" / "gold_coating_mirror_aoi45.txt"
        )
        data = np.loadtxt(filename, skiprows=1)
        super().__init__(data[:, 0] * 1000, data[:, 2] / 100, data[:, 1] / 100)

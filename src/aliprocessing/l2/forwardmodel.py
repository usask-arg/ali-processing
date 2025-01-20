from __future__ import annotations

import numpy as np
import sasktran2 as sk
from skretrieval.core.sasktranformat import SASKTRANRadiance
from skretrieval.retrieval.forwardmodel import IdealViewingSpectrograph


class ALIForwardModel(IdealViewingSpectrograph):
    def __init__(self, *args, **kwargs):
        self._pol_states = kwargs.get("pol_states", False)

        stokes_sensitivities = {}
        i = 0
        for i, stat in enumerate(self._pol_states):
            sens = np.zeros(4)
            sens[i] = 1
            stokes_sensitivities[stat] = sens

        super().__init__(
            *args, **{**kwargs, "stokes_sensitivities": stokes_sensitivities}
        )

        self._inst_model = {
            "*": self._inst_model[next(iter(self._model_geometry.keys()))]
        }

    def _construct_engine(self):
        engines = {}

        base_key = next(iter(self._model_geometry.keys()))

        engines["*"] = sk.Engine(
            self._engine_config,
            self._model_geometry[base_key],
            self._viewing_geo[base_key],
        )

        return engines

    def _construct_atmosphere(self):
        atmo = {}

        base_key = next(iter(self._model_geometry.keys()))

        atmo["*"] = sk.Atmosphere(
            self._model_geometry[base_key],
            self._engine_config,
            wavelengths_nm=self._model_wavelength[base_key],
            pressure_derivative=False,
            temperature_derivative=False,
            specific_humidity_derivative=False,
        )

        self._state_vector.add_to_atmosphere(atmo["*"])
        self._ancillary.add_to_atmosphere(atmo["*"])

        return atmo

    def calculate_radiance(self):
        l1 = {}
        for key in self._engine:
            sk2_rad = self._engine[key].calculate_radiance(self._atmosphere[key])

            sk2_rad = self._state_vector.post_process_sk2_radiances(sk2_rad)
            sk2_rad = SASKTRANRadiance.from_sasktran2(sk2_rad)

            rad_copy = sk2_rad.data.copy(deep=True)

            for i, state in enumerate(self._pol_states):
                if state == "I":
                    continue

                if state == "dolp":
                    dolp_num = np.sqrt(
                        rad_copy["radiance"].isel(stokes=1) ** 2
                        + rad_copy["radiance"].isel(stokes=2) ** 2
                    )
                    dolp = dolp_num / rad_copy["radiance"].isel(stokes=0)
                    d_dolp_num = (
                        rad_copy["wf"].isel(stokes=1)
                        * rad_copy["radiance"].isel(stokes=1)
                        / dolp_num
                        + rad_copy["wf"].isel(stokes=2)
                        * rad_copy["radiance"].isel(stokes=2)
                        / dolp_num
                    )
                    d_dolp = d_dolp_num / rad_copy["radiance"].isel(
                        stokes=0
                    ) - dolp * rad_copy["wf"].isel(stokes=0) / rad_copy[
                        "radiance"
                    ].isel(
                        stokes=0
                    )

                    sk2_rad.data["radiance"].to_numpy()[:, :, i] = dolp
                    sk2_rad.data["wf"].to_numpy()[:, :, :, i] = d_dolp

                if state == "aolp":
                    aolp = 0.5 * np.arctan(
                        rad_copy["radiance"].isel(stokes=2)
                        / rad_copy["radiance"].isel(stokes=1)
                    )

                    d_aolp = 0.5 * (
                        (
                            1
                            / (
                                1
                                + (
                                    rad_copy["radiance"].isel(stokes=2)
                                    / rad_copy["radiance"].isel(stokes=1)
                                )
                                ** 2
                            )
                        )
                        * (
                            rad_copy["wf"].isel(stokes=2)
                            / rad_copy["radiance"].isel(stokes=1)
                            - rad_copy["radiance"].isel(stokes=2)
                            / rad_copy["radiance"].isel(stokes=1) ** 2
                            * rad_copy["wf"].isel(stokes=1)
                        )
                    )

                    sk2_rad.data["radiance"].to_numpy()[:, :, i] = aolp
                    sk2_rad.data["wf"].to_numpy()[:, :, :, i] = d_aolp.transpose(
                        "x", "spectral_grid", "los"
                    )
                if state == "q":
                    q = rad_copy["radiance"].isel(stokes=1) / rad_copy["radiance"].isel(
                        stokes=0
                    )

                    d_q = rad_copy["wf"].isel(stokes=1) / rad_copy["radiance"].isel(
                        stokes=0
                    ) - q * rad_copy["wf"].isel(stokes=0) / rad_copy["radiance"].isel(
                        stokes=0
                    )

                    sk2_rad.data["radiance"].to_numpy()[:, :, i] = q
                    sk2_rad.data["wf"].to_numpy()[:, :, :, i] = d_q

            l1 = self._inst_model[key].model_radiance(sk2_rad, None)

            self._observation.append_information_to_l1(l1)

        return l1

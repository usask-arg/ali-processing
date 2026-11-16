"""The skretrieval Retrieval used by the ALI L2."""

from __future__ import annotations

from skretrieval.retrieval.processing import Retrieval


class ALIRetrieval(Retrieval):
    """skretrieval Retrieval that also gives the solution errors."""

    def retrieve(
        self,
        enabled_state_elements: list[str] | None = None,
        enabled_measurement_vectors: list[str] | None = None,
    ):
        """Retrieve, and also describe the state with the solution covariance.

        skretrieval's ``state`` gives each element's ``_1sigma_error`` from the
        measurement-noise covariance G S_e G^T. ``solution_state`` is the same
        description with the solution covariance (K^T S_e^-1 K + R)^-1 instead, so
        its ``_1sigma_error`` variables are the solution errors (see l2.yaml).
        """
        sv = self._state_vector.sv
        # The elements enabled during the retrieval; skretrieval re-enables them
        # all afterwards, which would misalign the covariance slices
        enabled = {
            name: (
                element.enabled
                if enabled_state_elements is None
                else name in enabled_state_elements
            )
            for name, element in sv.items()
        }
        results = super().retrieve(enabled_state_elements, enabled_measurement_vectors)

        after = {name: element.enabled for name, element in sv.items()}
        try:
            for name, element in sv.items():
                element.enabled = enabled[name]
            results["solution_state"] = self._state_vector.describe(
                {
                    "error_covariance_from_noise": results["minimizer"][
                        "solution_covariance"
                    ]
                }
            )
        finally:
            for name, element in sv.items():
                element.enabled = after[name]
        return results

from __future__ import annotations

import numpy as np
import sasktran as sk

rg = 0.08
sg = 1.6
alts = np.array([0.0, 100000.0])

particle_size_values = {
    "SKCLIMATOLOGY_LOGNORMAL_MODERADIUS_MICRONS": np.array([rg, rg]),
    "SKCLIMATOLOGY_LOGNORMAL_MODEWIDTH": np.array([sg, sg]),
}
particlesize_climatology = sk.ClimatologyUserDefined(alts, particle_size_values)
optical_property = sk.MieAerosol(particlesize_climatology, "H2SO4")

xsec = optical_property.calculate_cross_sections(
    sk.MSIS90(), 0.0, 0.0, 10000.0, 53000.0, [745.0, 869.0]
)
xsec.total[0] / xsec.total[1]

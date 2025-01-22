from __future__ import annotations

import numpy as np
import sasktran2 as sk


def aerosol_median_radius_db():
    refrac = sk.mie.refractive.H2SO4()
    dist = sk.mie.distribution.LogNormalDistribution().freeze(mode_width=1.6)

    db = sk.database.MieDatabase(
        dist,
        refrac,
        np.array([470, 525, 745, 1020, 1230, 1450, 1500]),
        median_radius=np.arange(10, 600, 10.0),
    )

    db.path()
    ssa = db._database["xs_scattering"] / db._database["xs_total"]
    ssa.to_numpy()[ssa.to_numpy() >= 1] = 0.99999
    db._database["xs_scattering"] = ssa * db._database["xs_total"]

    return db

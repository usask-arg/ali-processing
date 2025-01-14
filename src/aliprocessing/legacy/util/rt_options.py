from __future__ import annotations

import numpy as np


def simulation_rt_opts(
    configure_for_cloud=False,
    cloud_grid=100.0,
    coarse_grid=1000.0,
    cloud_lower_bound=0.0,
    cloud_upper_bound=30000.1,
    two_dim=False,
    single_scatter=False,
):
    options = {}
    if single_scatter:
        options["numordersofscatter"] = 1
    options["polarization"] = True

    if configure_for_cloud:
        toa = 100_000.0
        surface = 0.0

        solar_shells = np.sort(
            np.unique(
                np.concatenate(
                    [
                        np.arange(0, cloud_lower_bound, coarse_grid),
                        np.arange(cloud_lower_bound, cloud_upper_bound, cloud_grid),
                        np.arange(int(cloud_upper_bound), 100001, coarse_grid),
                    ]
                )
            )
        )
        options["manualraytracingshells"] = solar_shells
        options["manualsolarraytracingshells"] = solar_shells

        cloud_heights = np.arange(cloud_lower_bound, cloud_upper_bound, cloud_grid)

        optical_heights = np.concatenate([np.arange(surface, toa, 1000.0), [toa]])
        optical_heights = np.sort(
            np.unique(np.concatenate([optical_heights, cloud_heights]))
        )

        diffuse_heights = np.arange(surface + 1e-6, toa, coarse_grid)
        diffuse_heights = np.sort(
            np.unique(np.concatenate((diffuse_heights, cloud_heights)))
        )

        options["manualopticalheights"] = optical_heights
        options["manualdiffuseheights"] = diffuse_heights

        options["maxopticaldepthofcell"] = 0.01
        options["minextinctionratioofcell"] = 1.0

    if two_dim:
        # options['numordersofscatter'] = 1
        options["opticalanglegrid"] = np.linspace(-10, 10, 50)
        options["opticaltabletype"] = 2

    return options


def retrieval_rt_opts(
    aerosol_ret,
    configure_for_cloud=False,
    cloud_grid=100.0,
    coarse_grid=1000.0,
    cloud_lower_bound=0.0,
    cloud_upper_bound=30000.1,
    single_scatter=False,
):
    options = simulation_rt_opts(
        configure_for_cloud=configure_for_cloud,
        cloud_grid=cloud_grid,
        coarse_grid=coarse_grid,
        cloud_lower_bound=cloud_lower_bound,
        cloud_upper_bound=cloud_upper_bound,
        two_dim=False,
        single_scatter=single_scatter,
    )
    options["calcwf"] = 2
    options["wfheights"] = aerosol_ret.jacobian_altitudes
    options["wfwidths"] = (
        np.ones_like(aerosol_ret.jacobian_altitudes) * aerosol_ret._retrieval_alt_res
    )
    return options

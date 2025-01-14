from __future__ import annotations

import os

folder = os.path.dirname(__file__)
accp_folder = os.path.abspath(os.path.join(folder, "..", "..", ".."))


class Config:
    MATPLOTLIB_STYLE_FILE = os.path.abspath(os.path.join(folder, "grl_paper.mplstyle"))
    DIAGNOSTIC_FIGURE_FOLDER = os.path.join(folder, "..", "diagnostics", "figures")
    CALIPSO_OMPS_FILE = os.path.join(
        accp_folder,
        "accp-simulations",
        "accp",
        "data",
        "curtain_2019_9_1_h21m38_v101.nc",
    )
    AEROSOL_APRIORI_FILE = os.path.join(
        folder, "..", "..", "data", "retrieval", "aerosol_apriori.txt"
    )

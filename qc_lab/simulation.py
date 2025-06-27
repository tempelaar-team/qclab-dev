"""
This file contains the Simulation, State, and Data classes. It also contains additional
functions for initializing and handling these objects.
"""

import numpy as np
from qc_lab.constants import Constants
from qc_lab.variable import Variable


class Simulation:
    """
    The simulation object represents the entire simulation process.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "tmax": 10,
            "dt_update": 0.001,
            "dt_gather": 0.1,
            "num_trajs": 100,
            "batch_size": 25,
        }
        settings = {**self.default_settings, **settings}
        self.settings = Constants()
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.algorithm = None
        self.model = None
        self.state = Variable()

    def initialize_timesteps(self):
        """
        Initialize the timesteps for the simulation based on the parameters.

        First adjusts tmax to be the closest integer multiple of dt_update.
        Then adjusts dt_update_gather to be the closest integer multiple of dt_update as well.
        Then adjusts tmax to be the closest point on the grid defined by dt_gather.
        """
        tmax = self.settings.get("tmax", self.default_settings.get("tmax"))
        dt_update = self.settings.get("dt_update", self.default_settings.get("dt_update"))
        dt_gather = self.settings.get(
            "dt_gather", self.default_settings.get("dt_gather")
        )

        tmax_n = np.round(tmax / dt_update).astype(int)
        dt_gather_n = np.round(dt_gather / dt_update).astype(int)
        tmax_n = np.round(tmax_n / dt_gather_n).astype(int) * dt_gather_n

        self.settings.tmax_n = tmax_n
        self.settings.dt_gather_n = dt_gather_n
        self.settings.tdat = np.arange(0, self.settings.tmax_n + 1, 1) * dt_update
        self.settings.tdat_n = np.arange(0, self.settings.tmax_n + 1, 1)
        self.settings.tdat_output = (
            np.arange(
                0,
                self.settings.tmax_n + self.settings.dt_gather_n,
                self.settings.dt_gather_n,
            )
            * dt_update
        )
        self.settings.tdat_output_n = np.arange(
            0,
            self.settings.tmax_n + self.settings.dt_gather_n,
            self.settings.dt_gather_n,
        )

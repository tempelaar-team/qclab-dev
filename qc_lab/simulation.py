"""
This file contains the Simulation, State, and Data classes. It also contains additional 
functions for initializing and handling these objects.
"""

import numpy as np
from qc_lab.constants import Constants
from qc_lab.vector import Vector


class Simulation:
    """
    The simulation object represents the entire simulation process.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = dict(
            tmax=10, dt=0.01, dt_output=0.1, num_trajs=10, batch_size=1
        )
        settings = {**self.default_settings, **settings}
        self.settings = Constants()
        for key, val in settings.items():
            setattr(self.settings, key, val)
        self.algorithm = None
        self.model = None
        self.state = Vector()

    def initialize_timesteps(self):
        """
        Initialize the timesteps for the simulation based on the parameters.
        """
        self.settings.tmax_n = np.round(
            self.settings.tmax / self.settings.dt, 1
        ).astype(int)
        self.settings.dt_output_n = np.round(
            self.settings.dt_output / self.settings.dt, 1
        ).astype(int)
        self.settings.tdat = (
            np.arange(0, self.settings.tmax_n + 1, 1) * self.settings.dt
        )
        self.settings.tdat_n = np.arange(0, self.settings.tmax_n + 1, 1)
        self.settings.tdat_output = (
            np.arange(0, self.settings.tmax_n + 1, self.settings.dt_output_n)
            * self.settings.dt
        )
        self.settings.tdat_output_n = np.arange(
            0, self.settings.tmax_n + 1, self.settings.dt_output_n
        )

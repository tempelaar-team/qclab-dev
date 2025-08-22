"""
This module defines the Simulation class as well as the default settings for a simulation in QC Lab.
"""

import logging
import numpy as np
from qc_lab.constants import Constants
from qc_lab.variable import Variable

logger = logging.getLogger(__name__)


class Simulation:
    """
    The simulation object represents the entire simulation process.
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = {}
        self.default_settings = {
            "tmax": 10.0,
            "dt_update": 0.001,
            "dt_collect": 0.1,
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

        First ensures that dt_collect >= dt_update, if not it adjusts dt_collect to be equal to dt_update.
        Then adjusts dt_collect to be the closest integer multiple of dt_update.
        Then adjusts tmax to be the closest integer multiple of dt_collect.
        """
        tmax = self.settings.get("tmax")
        dt_update = self.settings.get("dt_update")
        dt_collect = self.settings.get("dt_collect")

        logger.info(
            "Initializing timesteps with tmax=%s, dt_update=%s, dt_collect=%s",
            tmax,
            dt_update,
            dt_collect,
        )

        if dt_update > dt_collect:
            dt_collect = dt_update
            logger.warning(
                "dt_update is greater than dt_collect, setting dt_collect to dt_update."
            )

        # dt_collect_n is the number of update timesteps that defines a collect timestep.
        dt_collect_n = np.round(dt_collect / dt_update).astype(int)
        # tmax_n is the number of update timesteps that defines the total simulation time.
        self.settings.tmax_n = np.round(tmax / dt_collect).astype(int) * dt_collect_n
        self.settings.tmax = self.settings.tmax_n * dt_update
        self.settings.dt_collect_n = dt_collect_n
        self.settings.dt_collect = dt_collect_n * dt_update
        # t_update is the update time array for the simulation.
        self.settings.t_update = np.arange(0, self.settings.tmax_n + 1, 1) * dt_update
        # t_update_n is the update time array in terms of the number of update steps.
        self.settings.t_update_n = np.arange(0, self.settings.tmax_n + 1, 1)
        # t_collect_n is the collect time array for the simulation in terms of the number of update steps.
        self.settings.t_collect_n = np.arange(
            0,
            self.settings.tmax_n + self.settings.dt_collect_n,
            self.settings.dt_collect_n,
        )
        # t_collect is the collect time array for the simulation.
        self.settings.t_collect = self.settings.t_collect_n * dt_update

        logger.info(
            "Initialization finished with tmax=%s, dt_update=%s, dt_collect=%s",
            self.settings.tmax,
            self.settings.get("dt_update"),
            self.settings.dt_collect,
        )

"""
This module contains the serial driver for the dynamics core.
"""

import logging
import numpy as np
from qc_lab.data import Data
from qc_lab.variable import initialize_variable_objects
import qc_lab.dynamics as dynamics
from qc_lab.utils import get_log_output

logger = logging.getLogger(__name__)


def serial_driver(sim, seeds=None, data=None):
    """
    Serial driver for the dynamics core.
    """
    # First initialize the model constants.
    sim.model.initialize_constants()
    if data is None:
        data = Data()
    if seeds is None:
        if len(data.data_dict["seed"]) > 0:
            offset = np.max(data.data_dict["seed"]) + 1
        else:
            offset = 0
        seeds = offset + np.arange(sim.settings.num_trajs, dtype=int)
        num_trajs = sim.settings.num_trajs
    else:
        num_trajs = len(seeds)
        logger.warning(
            "Setting sim.settings.num_trajs to the number of provided seeds: %s",
            num_trajs,
        )
        sim.settings.num_trajs = num_trajs
    # Determine the number of simulations required to execute the total number of trajectories.
    num_sims = int(num_trajs / sim.settings.batch_size) + 1

    logger.info(
        "Running %s simulations with %s seeds in each batch.",
        num_sims,
        sim.settings.batch_size,
    )
    for n in range(num_sims):
        batch_seeds = seeds[
            n * sim.settings.batch_size : (n + 1) * sim.settings.batch_size
        ]
        if len(batch_seeds) == 0:
            break
        logger.info("Running simulation %s with seeds %s.", n + 1, batch_seeds)
        sim.settings.batch_size = len(batch_seeds)
        sim.initialize_timesteps()
        parameters, state = initialize_variable_objects(sim, batch_seeds)
        new_data = Data()
        new_data.data_dict["seed"] = state.seed
        logger.info("Starting dynamics calculation.")
        new_data = dynamics.dynamics(sim, parameters, state, new_data)
        logger.info("Dynamics calculation completed.")
        data.add_data(new_data)
    # Attach the collected log output to the data object before returning.
    data.log = get_log_output()
    return data

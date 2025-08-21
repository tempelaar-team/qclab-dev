"""
This module contains the parallel driver for the dynamics simulation in QC Lab.
"""

import multiprocessing
import logging
import copy
import numpy as np
import qc_lab.dynamics as dynamics
from qc_lab.data import Data
from qc_lab.variable import initialize_variable_objects
from qc_lab.utils import get_log_output

logger = logging.getLogger(__name__)


def parallel_driver_multiprocessing(sim, seeds=None, data=None, num_tasks=None):
    """
    Parallel driver for the dynamics core using the python library multiprocessing.
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
    if num_tasks is None:
        size = multiprocessing.cpu_count()
    else:
        size = num_tasks
    logger.info("Using %s CPU cores for parallel processing.", size)
    if sim.settings.num_trajs % sim.settings.batch_size == 0:
        num_sims = sim.settings.num_trajs // sim.settings.batch_size
    else:
        num_sims = sim.settings.num_trajs // sim.settings.batch_size + 1
    logger.info(
        "Running %s simulations with %s seeds in each batch.",
        num_sims,
        sim.settings.batch_size,
    )
    batch_seeds_list = (
        np.zeros((num_sims * sim.settings.batch_size), dtype=int) + np.nan
    )
    batch_seeds_list[:num_trajs] = seeds
    batch_seeds_list = batch_seeds_list.reshape((num_sims, sim.settings.batch_size))
    sim.initialize_timesteps()
    input_data = [
        (
            copy.deepcopy(sim),
            *initialize_variable_objects(
                sim, batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)
            ),
            Data(batch_seeds_list[n][~np.isnan(batch_seeds_list[n])].astype(int)),
        )
        for n in range(num_sims)
    ]
    for i in range(num_sims):
        input_data[i][0].settings.batch_size = len(input_data[i][2].seed)
        logger.info(
            "Running simulation %s with seeds %s.", i + 1, input_data[i][2].seed
        )
    with multiprocessing.Pool(processes=size) as pool:
        results = pool.starmap(dynamics.dynamics, input_data)
    for result in results:
        data.add_data(result)
    # Attach collected log output
    data.log = get_log_output()
    return data

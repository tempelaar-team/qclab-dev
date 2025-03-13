"""
This module contains the slurm driver for the dynamics core.
"""

import os
import numpy as np
from qc_lab.data import Data
import qc_lab.dynamics.serial_driver as serial_driver


def slurm_driver(sim, seeds=None, num_tasks=1, data=None):
    """
    Slurm driver for the dynamics core.
    """
    if data is None:
        data = Data()  # Create a new Data object if none is provided
    if seeds is None:
        seeds = sim.generate_seeds(data)  # Generate seeds if none are provided
        num_trajs = sim.parameters.num_trajs
    else:
        num_trajs = len(
            seeds
        )  # Use the length of provided seeds as the number of trajectories

    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

    assert (
        np.mod(num_trajs, num_tasks) == 0
    ), "Number of trajectories must be divisible by number of tasks"
    # get number of trajectories per task
    num_trajs_per_task = int(num_trajs / num_tasks)
    seeds = seeds.reshape((num_tasks, num_trajs_per_task))
    task_seeds = seeds[idx]
    data = serial_driver.serial_driver(sim, task_seeds, data=data)
    return data, idx

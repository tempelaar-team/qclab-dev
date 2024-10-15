import os
import numpy as np
import qclab.simulation as simulation
from qclab.drivers.serial_driver import dynamics_serial


def dynamics_parallel_slurm(algorithm, sim, seeds, ntasks, ncpus_per_task, sub_driver=dynamics_serial,
                            data=simulation.Data()):
    # get SLURM id of 
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    # total number of seeds must be an integer multiple of the num_trajs
    assert np.mod(len(seeds), sim.num_trajs) == 0  
    num_sims = int(len(seeds) / sim.num_trajs)
    seeds = seeds.reshape((sim.num_trajs, num_sims))  # get seeds for each simulation
    # number of seeds per simulation must be an integer multiple of ntasks
    assert np.mod(num_sims, ntasks) == 0  
    # determine seeds to be used in this
    num_sims_per_task = int(num_sims / ntasks)
    task_seeds = seeds[:, idx * num_sims_per_task:(idx + 1) * num_sims_per_task]
    # execute trajectories using sub_driver for this task
    data = sub_driver(algorithm, sim, task_seeds.flatten(), ncpus=ncpus_per_task, data=data)
    return data, idx

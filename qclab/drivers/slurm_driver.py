import os 
import numpy as np
import qclab.simulation as simulation
import qclab.dynamics as dynamics
from qclab.drivers.serial_driver import dynamics_serial


def dynamics_parallel_slurm(algorithm, sim, seeds, ntasks, ncpus_per_task, sub_driver = dynamics_serial, data = simulation.Data()):
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    assert np.mod(len(seeds), sim.num_trajs) == 0
    num_sims = int(len(seeds)/sim.num_trajs)
    seeds = seeds.reshape((sim.num_trajs,num_sims)) # get seeds for each simulation
    task_seeds = seeds[:,idx*ntasks:(idx + 1)*ntasks]
    data = sub_driver(algorithm, sim, task_seeds.flatten(), ncpus = ncpus_per_task, data = data) 
    return data


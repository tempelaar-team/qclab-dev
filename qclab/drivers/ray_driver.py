import ray
import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm
import logging
import json
import numpy as np

def dynamics_parallel_ray(sim, seeds, nprocs, data = simulation.Data()):
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    @ray.remote
    def dynamics_ray(sim, seed):
        return dynamics.dynamics(sim, simulation.Trajectory(seed))
    num_sims = int(len(seeds)/sim.num_trajs)
    seeds = seeds.reshape((sim.num_trajs,num_sims))
    for run in tqdm(range(0, int(num_sims/nprocs)+1)):
        seed_list = seeds[:,run*nprocs:(run+1)*nprocs]
        results = [dynamics_ray.remote(sim, seed_list[:,n].flatten()) for n in range(np.shape(seed_list)[-1])]
        for r in results:
            traj = ray.get(r)
            data.add_data(traj)
    ray.shutdown()
    return data

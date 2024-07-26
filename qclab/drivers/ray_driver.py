import ray
import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm
import logging
import json
import numpy as np

def dynamics_parallel_ray(algorithm, sim, seeds, ncpus, data = simulation.Data()):
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    @ray.remote
    def dynamics_ray(algorithm, sim, seeds):
        sim.seeds = seeds
        traj = simulation.Trajectory()
        traj.seeds = seeds
        return dynamics.dynamics(algorithm, sim, traj)
    assert np.mod(len(seeds), sim.num_trajs) == 0
    num_sims = int(len(seeds)/sim.num_trajs)
    seeds = seeds.reshape((sim.num_trajs,num_sims))
    for run in tqdm(range(0, int(num_sims/ncpus)+1)):
        seed_list = seeds[:,run*ncpus:(run+1)*ncpus]
        results = [dynamics_ray.remote(algorithm, sim, seed_list[:,n].flatten()) for n in range(np.shape(seed_list)[-1])]
        for r in results:
            traj = ray.get(r)
            data.add_data(traj)
    ray.shutdown()
    return data

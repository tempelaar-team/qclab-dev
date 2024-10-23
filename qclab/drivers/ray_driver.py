import ray
import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm
import numpy as np


def dynamics_parallel_ray(algorithm, model, seeds, ncpus, data=simulation.Data()):
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    @ray.remote
    def dynamics_ray(algorithm, model, seeds):
        model.seeds = seeds
        traj = simulation.Trajectory()
        traj.seeds = seeds
        return dynamics.dynamics_(algorithm, model, traj)

    assert np.mod(len(seeds), model.batch_size) == 0
    num_sims = int(len(seeds) / model.batch_size)
    seeds = seeds.reshape((model.batch_size, num_sims))
    for run in tqdm(range(0, int(num_sims / ncpus) + 1)):
        seed_list = seeds[:, run * ncpus:(run + 1) * ncpus]
        results = [dynamics_ray.remote(algorithm, model, seed_list[:, n].flatten()) for n in
                   range(np.shape(seed_list)[-1])]
        for r in results:
            traj = ray.get(r)
            data.add_data(traj)
    ray.shutdown()
    return data

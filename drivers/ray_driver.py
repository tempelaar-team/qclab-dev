import ray
import simulation
import dynamics
from tqdm import tqdm

def dynamics_parallel_ray(sim, seeds, nprocs, data = simulation.Data()):
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    @ray.remote
    def dynamics_ray(sim, seed):
        return dynamics.dynamics(sim, simulation.Trajectory(seed))
    num_traj = len(seeds)
    for run in tqdm(range(0, int(num_traj/nprocs)+1)):
        seed_list = seeds[run*nprocs:(run+1)*nprocs]
        results = [dynamics_ray.remote(sim, seed) for seed in seed_list]
        for r in results:
            traj = ray.get(r)
            data.add_data(traj)
    return data

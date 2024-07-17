import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm
import numpy as np

def dynamics_serial(algorithm, sim, seeds, data = simulation.Data()):

    assert np.mod(len(seeds), sim.num_trajs) == 0
    num_sims = int(len(seeds)/sim.num_trajs)
    seeds = seeds.reshape((sim.num_trajs,num_sims))
    for n in tqdm(range(num_sims)):
        sim.seeds = seeds[:,n].flatten()
        traj = simulation.Trajectory()
        traj.seeds = seeds[:,n].flatten()
        traj = dynamics.dynamics(algorithm, sim, traj)
        data.add_data(traj)
    return data
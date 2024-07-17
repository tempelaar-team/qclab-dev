import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm

def dynamics_serial(algorithm, sim, seeds, data = simulation.Data()):
    for seed in tqdm(seeds):
        traj = dynamics.dynamics(algorithm, sim, seed, simulation.Trajectory(seed))
        data.add_data(traj)
    return data
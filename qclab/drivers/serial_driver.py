import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm

def dynamics_serial(sim, seeds, data = simulation.Data()):
    for seed in tqdm(seeds):
        traj = dynamics.dynamics(sim, simulation.Trajectory(seed))
        data.add_data(traj)
    return data
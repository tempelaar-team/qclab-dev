from qclab.auxiliary import Trajectory, Data
import qclab.dynamics as dynamics
from tqdm import tqdm
import numpy as np


def dynamics_serial(recipe, model, seeds, ncpus=1, data=None):
    if data is None:
        data = Data()
    assert np.mod(len(seeds), model.batch_size) == 0
    # partition the sees across each group of model.batch_size trajectories
    num_sims = int(len(seeds) / model.batch_size)
    seeds = seeds.reshape((model.batch_size, num_sims))
    for n in tqdm(range(num_sims)):
        # send seeds and trajectory object to dynamics core
        model.seeds = seeds[:, n].flatten()
        traj = Trajectory()
        traj.seeds = seeds[:, n].flatten()
        traj = dynamics.dynamics(model, recipe, traj)
        # accumulate data in data object
        data.add_data(traj)
    return data

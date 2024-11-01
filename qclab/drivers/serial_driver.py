import qclab.auxiliary as auxiliary
import qclab.dynamics as dynamics
from tqdm import tqdm
import numpy as np


def dynamics_serial(recipe, model, ncpus=None, data=None):
    if data is None:
        data = auxiliary.Data()
    seeds = auxiliary.generate_seeds(recipe.params, data)
    # partition the sees across each group of model.batch_size trajectories
    num_sims = int(recipe.params.num_trajs / recipe.params.batch_size) + 1
    for n in tqdm(range(num_sims)):
        # send seeds and trajectory object to dynamics core
        traj = auxiliary.Trajectory()
        traj.seeds = seeds[n*recipe.params.batch_size:(n+1)*recipe.params.batch_size]
        recipe.params.batch_size = len(traj.seeds)
        recipe.params.seeds = np.copy(traj.seeds)
        traj = dynamics.dynamics(model, recipe, traj)
        # accumulate data in data object
        data.add_data(traj)
    return data

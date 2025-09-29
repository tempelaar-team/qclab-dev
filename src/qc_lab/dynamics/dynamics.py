"""
This module contains the dynamics core.
"""

import numpy as np
from tqdm import tqdm


def run_dynamics(sim, parameter, state, data):
    """
    Dynamics core for QC Lab.
    """
    # Define an update iterator using tqdm if progress_bar is True.
    t_update_iterator = sim.settings.t_update_n
    if getattr(sim.settings, "progress_bar", True):
        t_update_iterator = tqdm(t_update_iterator)

    # Iterate over each time step.
    for sim.t_ind in t_update_iterator:
        if sim.t_ind == 0:
            # Execute initialization recipe.
            parameter, state = sim.algorithm.execute_recipe(
                sim, parameter, state, sim.algorithm.initialization_recipe
            )
        # Detect collect timesteps.
        if np.mod(sim.t_ind, sim.settings.dt_collect_n) == 0:
            # Calculate output variables.
            parameter, state = sim.algorithm.execute_recipe(
                sim, parameter, state, sim.algorithm.collect_recipe
            )
            # Collect totals in output dictionary.
            data.add_output_to_data_dict(sim, state, sim.t_ind)
        # Execute update recipe.
        parameter, state = sim.algorithm.execute_recipe(
            sim, parameter, state, sim.algorithm.update_recipe
        )
    return data

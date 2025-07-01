"""
This file contains the dynamics core.
"""

import numpy as np
from tqdm import tqdm


def dynamics(sim, parameter, state, data):
    """
    Dynamics core for QC Lab.
    """
    # Iterate over each time step.
    for sim.t_ind in tqdm(sim.settings.tdat_n):
        if sim.t_ind == 0:
            # Execute initialization recipe.
            parameter, state = sim.algorithm.execute_recipe(
                sim, parameter, state, sim.algorithm.initialization_recipe
            )
        # Detect output timesteps.
        if np.mod(sim.t_ind, sim.settings.dt_gather_n) == 0:
            # Calculate output variables.
            parameter, state = sim.algorithm.execute_recipe(
                sim, parameter, state, sim.algorithm.gather_recipe
            )
            # Gather output variables into a dictionary.
            state.gather_outputs(sim.algorithm.gather_variables)
            # Collect totals in output dictionary.
            data.add_output_to_data_dict(sim, state, sim.t_ind)
        # Execute update recipe.
        parameter, state = sim.algorithm.execute_recipe(
            sim, parameter, state, sim.algorithm.update_recipe
        )
    return data

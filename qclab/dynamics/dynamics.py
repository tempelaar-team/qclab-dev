"""
This module contains the dynamics core.
"""
import numpy as np
from tqdm import tqdm


def dynamics(sim, full_state, data):
    """
    Run the dynamics of the simulation.

    Args:
        sim (Simulation): The simulation object containing the simulation parameters and state.
        state_list (list): List of state objects for each trajectory.
        full_state (State): The full state object containing all trajectories.
        data (Data): The Data object to store the simulation results.

    Returns:
        Data: The Data object containing the results of the simulation.
    """
    # Execute initialization recipe
    full_state = sim.algorithm.execute_initialization_recipe(sim, full_state)
    # Iterate over each time step
    for sim.t_ind in tqdm(sim.parameters.tdat_n):
        # Detect output timesteps
        if np.mod(sim.t_ind, sim.parameters.dt_output_n) == 0:
            # Calculate output variables
            full_state = sim.algorithm.execute_output_recipe(sim, full_state)
            # Collect output variables into a dictionary
            full_state.collect_output_variables(sim.algorithm.output_variables)
            # Collect totals in output dictionary
            data.add_to_output_total_arrays(sim, full_state, sim.t_ind)
        # Execute update recipe
        full_state = sim.algorithm.execute_update_recipe(sim, full_state)
    return data

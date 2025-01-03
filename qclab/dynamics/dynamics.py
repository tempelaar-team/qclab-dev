import numpy as np
from tqdm import tqdm


def dynamics(sim, state_list, full_state, data):
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
    state_list, full_state = sim.algorithm.execute_initialization_recipe(
        sim, state_list, full_state)

    # Iterate over each time step
    for t_ind in tqdm(sim.parameters.tdat_n):
        # Detect output timesteps
        if np.mod(t_ind, sim.parameters.dt_output_n) == 0:
            # Calculate output variables
            state_list, full_state = sim.algorithm.execute_output_recipe(
                sim, state_list, full_state)
            full_state.collect_output_variables(sim.algorithm.output_variables)
            # Collect totals in output dictionary
            data.add_to_output_total_arrays(sim, full_state, t_ind)
        # Execute update recipe
        state_list, full_state = sim.algorithm.execute_update_recipe(sim, state_list, full_state)
    return data

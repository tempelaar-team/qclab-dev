import qclab.simulation as simulation
import qclab.dynamics as dynamics
from tqdm import tqdm
import numpy as np


def run_simulation(sim, seeds=None, ncpus=1, data=None):
    """
    Run the simulation in a serial manner.

    Args:
        sim (Simulation): The simulation object containing the simulation parameters and state.
        seeds (list, optional): List of seeds for initializing the simulation. If None, seeds will be generated.
        ncpus (int, optional): Number of CPUs to use. This parameter is not used in the serial implementation.
        data (Data, optional): An existing Data object to store the simulation results. If None, a new Data object will be created.

    Returns:
        Data: The Data object containing the results of the simulation.
    """
    if data is None:
        data = simulation.Data()  # Create a new Data object if none is provided
    if seeds is None:
        seeds = sim.generate_seeds(data)  # Generate seeds if none are provided
        num_trajs = sim.parameters.num_trajs
    else:
        num_trajs = len(seeds)  # Use the length of provided seeds as the number of trajectories

    # Partition the seeds across each group of sim.parameters.batch_size trajectories
    num_sims = int(num_trajs / sim.parameters.batch_size) + 1
    for n in range(num_sims):
        batch_seeds = seeds[n * sim.parameters.batch_size:(n + 1) * sim.parameters.batch_size]
        if len(batch_seeds) == 0:
            break  # Exit the loop if there are no more seeds to process

        sim.initialize_timesteps()  # Initialize the timesteps for the simulation
        state_list, full_state = simulation.initialize_state_objects(sim, batch_seeds)  # Initialize state objects
        new_data = simulation.Data()  # Create a new Data object for this batch
        new_data.initialize_output_total_arrays(sim, full_state)  # Initialize output arrays in the Data object
        new_data = dynamics.dynamics(sim, state_list, full_state, new_data)  # Run the dynamics and collect data
        data.add_data(new_data)  # Add the collected data to the main Data object

    return data  # Return the Data object containing all simulation results
    
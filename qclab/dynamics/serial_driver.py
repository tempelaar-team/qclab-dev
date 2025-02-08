"""
This module contains the serial driver for the dynamics core.
"""
import qclab.simulation as simulation
import qclab.dynamics.dynamics as dynamics


def serial_driver(sim, seeds=None, data=None):
    if data is None:
        data = simulation.Data()  # Create a new Data object if none is provided
    if seeds is None:
        seeds = sim.generate_seeds(data)  # Generate seeds if none are provided
        num_trajs = sim.settings.num_trajs
    else:
        num_trajs = len(seeds)  # Use the length of provided seeds as the number of trajectories
    # Partition the seeds across each group of sim.settings.batch_size trajectories
    num_sims = int(num_trajs / sim.settings.batch_size) + 1
    for n in range(num_sims):
        batch_seeds = seeds[n * sim.settings.batch_size:(n + 1) * sim.settings.batch_size]
        if len(batch_seeds) == 0:
            break  # Exit the loop if there are no more seeds to process
        sim.initialize_timesteps()  # Initialize the timesteps for the simulation
        parameters, state = simulation.initialize_vector_objects(sim, batch_seeds)
        new_data = simulation.Data()  # Create a new Data object for this batch
        new_data.data_dic['seed'] = batch_seeds # add seeds from the batch
        new_data = dynamics.dynamics(sim, parameters, state, new_data)
        data.add_data(new_data)  # Add the collected data to the main Data object

    return data  # Return the Data object containing all simulation results

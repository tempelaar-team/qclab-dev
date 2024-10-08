import numpy as np
import qclab.simulation as simulation
import qclab.auxilliary as auxilliary

def dynamics(sim, recipe, traj = simulation.Trajectory()):
    # load defaults of recipe 
    sim = recipe.defaults(sim)
    # first initialize state
    state = recipe.state
    state.t_ind = 0
    state.traj = traj
    for func in recipe.initialize:
        state = func(sim, state)
    for t_ind in sim.tdat_n:
        if np.mod(t_ind, sim.dt_output_n) == 0:
            for func in recipe.output:
                state = func(sim, state)
            # calculate observables at output timestep
            observables_t = auxilliary.evaluate_observables_t(recipe)
            if t_ind == 0:
                for key in observables_t.keys():
                    traj.new_observable(key, (len(sim.tdat_output), *np.shape(observables_t[key])), observables_t[key].dtype)
            
            traj.add_observable_dict(int(t_ind / sim.dt_output_n), observables_t) # add observables to the trajectory object
            state.traj = traj
        for func in recipe.update:
            state = func(sim, state)
        state.t_ind = t_ind + 1
    traj.add_to_dic('t', sim.tdat_output * sim.num_trajs) # add time axis to trajectory object 
    return traj 
import numpy as np
import qclab.auxiliary as auxiliary


def dynamics(model, recipe, traj):
    # initialize the timestep axes
    model = auxiliary.initialize_timesteps(model)
    # load defaults of recipe class
    model = recipe.defaults(model)
    # first initialize state variable
    state = recipe.state
    # get simulation parameters 
    params = recipe.params
    # set initial time index to zero
    state.t_ind = 0
    # attach trajectory object to the state variable
    state.traj = traj
    # execute functions in the initialize procedure
    for func in recipe.initialize:
        state = func(state)
    # begine loop over timesteps
    for t_ind in params.tdat_n:
        # detect output timesteps
        if np.mod(t_ind, params.dt_output_n) == 0:
            # execute functions in the output procedure
            for func in recipe.output:
                state, model, params = func(state, model, params)
            # store the requested variables in a dictionary 
            observables_t = auxiliary.evaluate_observables_t(recipe)
            # if this is the first output timestep initialize the output variables in the trajectory object
            if t_ind == 0:
                for key in observables_t.keys():
                    traj.new_observable(key, (len(params.tdat_output), *np.shape(observables_t[key])),
                                        observables_t[key].dtype)
            # add the variables from the dictionary observables_t to the trajectory object
            traj.add_observable_dict(int(t_ind / params.dt_output_n),
                                     observables_t)  # add observables to the trajectory object
            # update the trajectory object in the state variable
            state.traj = traj
        # evaluate the functions in the update procedure
        for func in recipe.update:
            state = func(state)
        # increment the timestep index
        state.t_ind = t_ind + 1
    # attach a time axis to the trajectory object
    traj.add_to_dic('t', params.tdat_output * params.batch_size)
    return traj

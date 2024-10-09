Dynamics core
===================================

The dynamics core is a central function of the qc-lab package that enables it to execute arbitrary algorithms defined in terms of a recipe class. 
Before discussing the construction of the recipe class, it is illustrative to understand how qc-lab executes the different components of the 
recipe class::
    def dynamics(sim, recipe, traj=simulation.Trajectory()):
        # load defaults of recipe class
        sim = recipe.defaults(sim)
        # first initialize state variable
        state = recipe.state
        # set initial time index to zero
        state.t_ind = 0
        # attach trajectory object to the state variable
        state.traj = traj
        # execute functions in the initialize procedure
        for func in recipe.initialize:
            state = func(sim, state)
        # begine loop over timesteps
        for t_ind in sim.tdat_n:
            # detect output timesteps
            if np.mod(t_ind, sim.dt_output_n) == 0:
                # execute functions in the output procedure
                for func in recipe.output:
                    state = func(sim, state)
                # store the requested variables in a dictionary 
                observables_t = auxiliary.evaluate_observables_t(recipe)
                # if this is the first output timestep initialize the output variables in the trajectory object
                if t_ind == 0:
                    for key in observables_t.keys():
                        traj.new_observable(key, (len(sim.tdat_output), *np.shape(observables_t[key])),
                                            observables_t[key].dtype)
                # add the variables from the dictionary observables_t to the trajectory object
                traj.add_observable_dict(int(t_ind / sim.dt_output_n),
                                        observables_t)  # add observables to the trajectory object
                # update the trajectory object in the state variable
                state.traj = traj
            # evaluate the functions in the update procedure
            for func in recipe.update:
                state = func(sim, state)
            # increment the timestep index
            state.t_ind = t_ind + 1
        # attach a time axis to the trajectory object
        traj.add_to_dic('t', sim.tdat_output * sim.num_trajs)
        return traj
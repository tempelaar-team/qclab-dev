import numpy as np
import qclab.simulation as simulation

def dynamics(dyn, sim, traj=simulation.Trajectory()):
    # load defaults into the simulation object
    dyn = dyn(sim)
    # initialize dynamics
    dyn.initialize_dynamics(sim)
    # generate t=0 observables
    dyn.calculate_observables(sim)
    # store observables in trajectory object
    for key in dyn.observables_t.keys():
        traj.new_observable(key, (len(dyn.tdat_output), *np.shape(dyn.observables_t[key])), dyn.observables_t[key].dtype)
    # Begin dynamics loops
    t_output_ind = 0
    for dyn.t_ind in np.arange(0, len(dyn.tdat)):
        if t_output_ind == len(dyn.tdat_output):
            break
        if dyn.tdat_output[t_output_ind] <= dyn.tdat[dyn.t_ind] + 0.5 * sim.dt:
            ############################################################
            #                            OUTPUT TIMESTEP               #
            ############################################################
            dyn.calculate_observables(sim) # calculate observables
            traj.add_observable_dict(t_output_ind, dyn.observables_t) # add observables to the trajectory object
            t_output_ind += 1
        ############################################################
        #                         DYNAMICS TIMESTEP                #
        ############################################################
        dyn.propagate_classical_subsystem(sim) # evolve classical coordinates
        dyn.propagate_quantum_subsystem(sim) # evolve quantum subsystem
        dyn.update_state(sim) # update the state of the system
    traj.add_to_dic('t', dyn.tdat_output * sim.num_trajs) # add time axis to trajectory object 
    return traj
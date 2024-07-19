Algorithm Class Structure
===================================


**Warning: Modifying existing algorithms has the potential to break them, proceed with caution.**

Before discussing particular attributes of the Algorithm Class it will be 
instructive to examine structure of the dynamics core. An approximate representation
of the dynamics core containing the relevant pieces is::

        def dynamics(dyn: Algorithm Class, sim: Simulation Class, traj: Trajectory Class):
            dyn = dyn(sim) # Load defaults into the simulation object 
            dyn.initialize_dynamics(sim) # initialize dynamics 
            dyn.calculate_observables(sim) # calculate initial set of observables 
            # store observables 
            # store observables in trajectory object
            for key in dyn.observables_t.keys():
                traj.new_observable(key, (len(dyn.tdat_output), *np.shape(dyn.observables_t[key])), dyn.observables_t[key].dtype)
            # Begin dynamics loops:
            for t in timesteps:
                for t_output in output_timesteps:
                    dyn.calculate_observables(sim) # calculate observables 
                    traj.add_observable_dict(t_output_ind, dyn.observables_t) # add observables to the trajectory object
                dyn.propagate_classical_subsystem(sim) # evolve classical coordinates
                dyn.propagate_quantum_subsystem(sim) # evovle quantum state 
                dyn.update_state(sim) # update total state of the system (forces, Hamiltonians, etc.)
            return # observables 

The dynamics core begins by initializing a set of default attributes into the simulation object, accomplished by the ``__init__`` function
of the Algorithm Class. After then initializing the physical system with ``initialize_dynamics`` the dynamics core calculates
the first set of observables (at t=0) with ``calculate_observables`` and stores them in the dictionary ``dyn.observables_t`` which represents the instantaneous value of
the observables. The dynamics then proceeds as two loops,
the outer loop evovles the physical system and the inner loop calculates observables at the output timesteps. 
The evolution of the physical system proceeds in three steps, first the classical coordinates are evolved with 
``propagate_classical_subsystem``, then the quantum state is evolved with ``propagate_quantum_subsystem`` and finally 
the total state of the system is updated with ``update_state``. This final step is responsible for updating the quantum forces as well
as the system Hamiltonian and in surface hopping methods also updates the active surfaces and rescales classicla coordinates. 

This structure is intentionally general and as such relies on the Algorithm Class to implemented in an efficient manner. 
In practice, the Algorithm Classes that come with qc_lab are designed to be totally generic and are therefore missing some
optimizations that could be made for specific systems. 

Algorithm Class Attributes
--------------------------
* .. function:: __init__(self: Algorithm Class, sim: Simulation Class)

    Load default attributes into the Simulation Class and enforce any mandatory configurations of the Simulation Class attributes
    that could cause conflicts or errors. 

    :Parameters:
        self: Algorithm Class 

        sim: Simulation Class 

* .. function:: initialize_dynamics(self: Algorithm Class, sim: Simulation Class)

    Initialize the state of the physical system at the start of the dynamics. 

    :Parameters:
        self: Algorithm Class 

        sim: Simulation Class 

* .. function:: propagate_classical_subsystem(self: Algorithm Class, sim: Simulation Class)

    Evolve the classical coordinates over a single timestep 

    :Parameters:
        self: Algorithm Class 

        sim: Simulation Class 

* .. function:: propagate_quantum_subsystem(self: Algorithm Class, sim: Simulation Class)

    Evolve the quantum state over a single timestep 

    :Parameters:
        self: Algorithm Class 

        sim: Simulation Class 

* .. function:: update_state(self: Algorithm Class, sim: Simulation Class)

    Update the state of the physical system at the end of the timestep.  

    :Parameters:
        self: Algorithm Class 

        sim: Simulation Class 

* .. function:: calculate_observables(self: Algorithm Class, sim: Simulation Class)

    Calculate the instantaneous value of the observables and store them in a dictionary called self.observables_t

    :Parameters:
        self: Algorithm Class 

        sim: Simulation Class 


Example Algorithm Class
-----------------------

Here we demonstrate the implementation of the mean-field (Ehrenfest) method. See the home page for relevant citations. 

::

    class MeanFieldDynamics:
        def __init__(self, sim):
            var_names = list(sim.__dict__.keys())
            defaults = {
                'init_classical': auxilliary.harmonic_oscillator_bolztmann_init_classical,
                'h_c_branch': auxilliary.harmonic_oscillator_h_c_branch,
                'dh_c_dz_branch': auxilliary.harmonic_oscillator_dh_c_dz_branch,
                'dh_c_dzc_branch': auxilliary.harmonic_oscillator_dh_c_dzc_branch,
                'h_c_params' : (sim.h),
                'h_qc_params' : None,
                'h_q_params' : None,
                'tmax': 10,
                'dt_output': 0.1,
                'dt': 0.01,
                'temp':1,
                'num_states':2,
                'num_branches':1,
                'gauge_fix':0,
                'dmat_const':0,
                'observables':auxilliary.no_observables,
                'num_classical_coordinates':None
                }
            for name in defaults.keys():
                if not(name in list(var_names)):
                    sim.__dict__[name] = defaults[name]
            assert sim.num_branches == 1
            return
        
        def initialize_dynamics(self, sim):
            # initialize time axes 
            self.tdat_output = np.arange(0, sim.tmax + sim.dt_output, sim.dt_output)
            self.tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
            # initialize variables describing the state of the system
            self.z_coord = np.zeros((sim.num_trajs, sim.num_branches, sim.num_classical_coordinates), dtype=complex)
            # load initial values of the z coordinate 
            for traj_n in range(sim.num_trajs):
                self.z_coord[traj_n, :, :] = sim.init_classical(sim, sim.seeds[traj_n]) # init_classical could arguablty be in init_state
            self.z_coord = self.z_coord.reshape(sim.num_trajs*sim.num_branches, sim.num_classical_coordinates)
            # load initial values of the wavefunction
            self.wf_db = np.zeros((sim.num_trajs*sim.num_branches, sim.num_states), dtype=complex) + sim.wf_db[np.newaxis, :]
            # initialize gradients (Hamiltonian and quantum forces)
            self.h_tot = sim.h_q(sim.h_q_params)[np.newaxis, :, :] + sim.h_qc_branch(sim.h_qc_params, self.z_coord)
            self.qfzc = auxilliary.quantum_force_branch(self.wf_db, None, self.z_coord, sim)
            return
        
        def propagate_classical_subsystem(self, sim):
            self.z_coord = auxilliary.rk4_c(self.z_coord, self.qfzc, sim.dt, sim)
            return
        
        def propagate_quantum_subsystem(self, sim):
            self.wf_db = auxilliary.rk4_q_branch(self.h_tot, self.wf_db, sim.dt)
            return
        
        def update_state(self, sim):
            self.qfzc = auxilliary.quantum_force_branch(self.wf_db, None, self.z_coord, sim)
            self.h_tot = sim.h_q(sim.h_q_params)[np.newaxis, :, :] + sim.h_qc_branch(sim.h_qc_params, self.z_coord)
            return
        
        def calculate_observables(self, sim):
            self.dm_db = np.einsum('ni,nk->ik', self.wf_db, np.conj(self.wf_db))/(sim.num_branches)
            self.observables_t = sim.observables(sim, self)
            self.observables_t['e_q'] = np.real(np.einsum('ni,nij,nj', np.conjugate(self.wf_db), self.h_tot, self.wf_db))/(sim.num_branches)
            self.observables_t['e_c'] = np.sum(sim.h_c_branch(sim.h_c_params, self.z_coord))/(sim.num_branches)
            self.observables_t['dm_db'] = self.dm_db
            return 
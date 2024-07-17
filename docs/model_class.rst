Model Class Structure
===================================

A model class contains the following required attributes

Simulation Parameters
---------------------
* .. function:: num_trajs (int) -> 1

        An integer value giving the numer of trajectories that the dynamics core should execute at a single time

* .. function:: num_branches (int) -> num_states

        An integer value giving the total number of branches (realizations of a single set of initial classical coordinates) to be executed. 
        if ``sh_deterministic=True`` then ``num_branches = num_states`` otherwise num_branches takes on an algorithm-dependent quantity. 

* .. function:: dt (float) -> 0.01

        A float giving the timestep for the propagation of the mixed quantum-classical dynamics

* .. function:: dt_output (float) -> 0.1

        A float giving the timestep for the calculation of observables, this will be the output resolution. 

* .. function:: tmax (float) -> 10

        A float giving the total simulation time

Physical System
---------------
* .. function:: num_states (int)

        An integer giving the number of quantum states (i.e. the dimension of the quantum Hamiltonian)
* .. function:: num_classical_coordinates (int)

        An integer value giving the total number of classical coordinates in a single realization of the physical system.

* .. function:: h (ndarray, shape=(num_classical_coordinates), dtype=float)

        A numpy array with length ``num_classical_coordinates`` that serves as an arbitrary parameter to define the complex-valued classical coordinates. See reference [1] (Miyazaki 2024) for details on this. When under a harmonic potential, it is convenient to set ``h`` for each coordinate to be its frequency. 

* .. function:: h_q_params ( tuple)

        Any parameters needed to generate the Hamiltonian matrix of the quantum subsystem

* .. function:: h_qc_params (tuple)

        Any parameters needed to generate the Hamiltonian matrix describing the quantum-classical interaction. 

* .. function:: h_c_params (tuple)

        Any parameters needed to generate the Hamiltonian function of the classical subsystem. 

* .. function:: h_q(h_q_params: tuple) -> ndarray

    Calculate the Hamiltonian matrix of the quantum subsystem

    :Parameters:
        h_q_params : tuple

    :Returns:
        ndarray, shape=(num_states, num_states), dtype=complex
            The quantum Hamiltonian


    :See Also:
        h_q_params

* .. function:: h_qc_branch(h_q_params: tuple, z_branch: ndarray) -> ndarray

    Calculate the Hamiltonian matrix of the quantum-classical interaction

    :Parameters:
        h_qc_params : tuple

        z_branch : ndarray, shape=(num_branches * num_trajs, num_classical_coordinates)

    :Returns:
        ndarray, shape=(num_trajs*num_branches, num_states, num_states), dtype=complex
            The quantum-classical interaction Hamiltonian

    :See Also:
        h_qc_params


A model class can be instantiated as follows::

    class ModelClass:
        def __init__(self):
            # here we instantiate the requried attributes of the ModelClass that define the physical system
            self.num_states
            self.num_classical_coordinates
            self.h
            self.h_q_params
            self.h_qc_params
            self.h_c_params
            self.h_q
            self.h_qc_branch
            self.h_c

        


            
Model Class
===================================

The model class is responsible for the definition of the underlying physical system being modeled and has a small set of functions 
and attributes that are necessary for the execution of a mixed quantum-classical dynamics algorithm. Some of these algorithms are 
specific to particular recipes and will be indicated as such. 


Simulation Attributes
---------------------

.. py:data:: model.batch_size
    :type: int
    :value: 1

        The numer of trajectories that the dynamics core should execute at a single time

.. py:data:: model.dt
    :type: float 
    :value: 0.01

        The timestep for the propagation of the dynamics. 

.. py:data:: model.dt_output 
    :type: float 
    :value: 0.1 

        The timestep for the calculation of outputs from the dynamics. Must be an integer multiple of `model.dt`.

.. py:data:: model.tmax 
    :type: float 
    :value: 10

        The total simulation time. Must be an integer multiple of `model.dt_output`

.. py:data:: model.num_branches 
    :type: int 
    :value: model.num_states

        The number of branches each trajectory is associated with. This quantity depends on the algorithm used, in mean-field it is `1` and in surface hopping approaches it can vary.

Surface Hopping Specific Attributes 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:data:: model.sh_deterministic
    :type: bool 
    :value: True

        ``model.sh_deterministic = True``: ``model.num_branches = model.num_states`` and each branch corresponds to an iniailization in a different active surface. 

        ``model.sh_deterministic = False``: the number of branches can be set to anything and each branch corresponds to an active surface that is stochastically initialized accordint to their
        adiabatic populations.

.. py:data:: model.gauge_fix 
    :type: int 
    :value: 0

        Depending on the details of the system being simulated different levels of gauge-fixing may be necessary. 

        ``model.gauge_fix = 0``: At each timestep the sign of each eigenvector is adjusted so that its overlap with the previous timestep's eigenvectors is positive. 
        Appropriate for a real-valued problem.

        ``model.gauge_fix = 1``: At each timestep the phase of each eigenvector is adjusted to maximize its overlap with the previous timestep's eigenvectors. Appropriate for a problem 
        where the Hamiltonian is complex-valued but varies slowly enough that the derivative couplings remain real-valued once set in a real-valued gauge at `t=0`.

        ``model.gauge_fix = 2``: At each timestep the phase of each eigenvector is adjusted so that the derivative couplings are real-valued. Appropriate when the above condition
        does not hold and the derivative couplings become complex-valued. 

        Note that none of these options are sufficient for a problem where the derivative couplings are complex-valued for topological reasons rather than gauge reasons. 
        See Ref. 7 (Krotz, 2024) on the main page for details of the gauge problem in surface hopping methods. 




Physical System Attributes 
--------------------------

.. py:data:: model.num_states 
    :type: int 

        The dimension of the quantum Hilbert space. 

.. py:data:: model.wf_db 
    :type: numpy.ndarray(shape=model.num_states, dtype=complex)

        The initial diabatic wavefunction coefficients

.. py:data:: model.num_classical_coordinates 
    :type: int 

        The total number of classical coordinates in a single instantiation of the model

.. py:data:: model.pq_weight 
    :type: numpy.ndarray(shape=model.num_classical_coordinates, dtype=float)

        The auxiliary weighting parameter used to construct the complex classical coordinate. See Ref. 1 (Myazaki, 2024) from the home page for details. 

.. py:function:: model.h_q[namespace](state)
        
        Calculates the Hamiltonian of the quantum subsystem excluding the quantum-classical interaction term.

        :param state: a namespace containing the dynamic and static quantites characterizing the simulation.

        :returns: ``numpy.ndarray(shape=(state.model.num_branches, state.model.num_branches, state.model.num_states, state.model.num_states), dtype=complex)``

.. py:function:: model.h_qc[namespace, numpy.ndarray](state, z_coord)

        Calculates the quantum-classical interaction Hamiltonian.

        :param state: a namespace containing the dynamic and static quantites characterizing the simulation.
        :param z_coord: complex classical coordinate 
        :returns: ``numpy.ndarray(shape = (*np.shape(z_coord)[:-1], state.model.num_states, state.model.num_states), dtype=complex)``

.. py:function:: model.h_c[namespace, numpy.ndarray](state, z_coord)

        Calculates the classical Hamiltonian of each branch and trajectory. 

        :param state: a namespace containing the dynamic and static quantites characterizing the simulation.
        :param z_coord: complex classical coordinate 
        :returns: ``numpy.ndarray(shape = (*np.shape(z_coord)[:-1]), dtype=complex)``

.. py:function:: model.dh_qc_dz[namespace, ndarray, ndarray, ndarray](state, z_coord, wf_a, wf_b)

        Calculates the matrix element of the gradient of the quantum-classical interaction with respect to z_coord between wf_a and wf_b:  

        .. math::

                \langle a\vert \partial_{z_{i}}\hat{H}_{qc}(\mathbf{z}) \vert b\rangle


        By convention, the complex-conjugation of wf_a is performed internally.

        :param state:
        :param z_coord:
        :param wf_a:
        :param wf_b:
        :returns: ``numpy.ndarray(shape = (*np.shape(z_coord)[:-1], state.model.num_classical_coordinates), dtype=complex)``


.. py:function:: model.dh_qc_dzc[namespace, ndarray, ndarray, ndarray](state, z_coord, wf_a, wf_b)

        Calculates the matrix element of the gradient of the quantum-classical interaction with respect to the conjugate z_coord between wf_a and wf_b:

        .. math::

                \langle a\vert \partial_{z^{*}_{i}}\hat{H}_{qc}(\mathbf{z}) \vert b\rangle


        By convention, the complex-conjugation of wf_a is performed internally.

        :param state:
        :param z_coord:
        :param wf_a:
        :param wf_b:
        :returns: ``numpy.ndarray(shape = (*np.shape(z_coord)[:-1], state.model.num_classical_coordinates), dtype=complex)``

.. py:function:: model.dh_c_dz[namespace, ndarray, ndarray, ndarray](state, z_coord)

        Calculates the gradient of the classical interaction with respect to z_coord:  `dH_{c}/dz`

        :param state:
        :param z_coord:
        :returns: ``numpy.ndarray(shape = np.shape(z_coord), dtype=complex)``


.. py:function:: model.dh_c_dzc[namespace, ndarray, ndarray, ndarray](state, z_coord)

        Calculates the gradient of the classical interaction with respect to z_coord:  `dH_{c}/dz*`
        In general this is the conjugate of ``model.dh_c_dz``

        :param state:
        :param z_coord:
        :returns: ``numpy.ndarray(shape = np.shape(z_coord), dtype=complex)``

.. py:function:: model.init_classical[class, int](model, seed)

        Initializes the classical coordinates associated with trajectory labeled by seed. 

        :param model: model class 
        :param seed: seed labeling the trajectory
        :returns: ``numpy.ndarray(shape = model.num_classical_coordinates, dtype=complex)``


Models included in QC-lab 
-------------------------

.. toctree::
   :maxdepth: 3

   spinboson_class
   dba_class
   holstein_class
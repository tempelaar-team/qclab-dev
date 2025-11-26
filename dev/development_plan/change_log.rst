.. _change_log:

Change Log
=================


Version 1.0.1
-----------------

- Added AdiabaticMeanField algorithm. 
- Added QChemASE Model class that uses the Q-Chem interface of the Atomic Simulation Environment (ASE) to perform ab initio quantum chemistry calculations.
- Added custom ASE Q-Chem calculator. 
- Added numerical constants. 
- AdiabaticMeanField algorithm will by default use RK4 integration for the classical degrees of freedom. We will include
  a tutorial on replacing it with Velocity Verlet for ab initio calculations as well.
- Added AbInitioMeanField algorithm that uses Velocity Verlet integration for the classical degrees of freedom,
  suitable for ab initio calculations or any other problem where the quantum-classical Hamiltonian only depends on the position (real part of z) coordinate.
- In initialization_tasks.copy_in_state and initialization_tasks.copy_to_parameters, numpy.copy was replaced with copy.copy.

Bug Fixes
---------
- Fixed typo in the ``_init_h_qc`` ingredient of ``HolsteinLattice`` model affecting the calculation of the ``diagonal_linear_coupling`` matrix.
  Previously, changing the ``classical_coordinate_weight`` parameter did not correctly influence the coupling terms. Now, the coupling terms properly scale with the ``classical_coordinate_weight`` as intended. This typo had no impact on the correctness of the ``HolsteinLattice`` model where ``classical_coordinate_weight`` was set to ``harmonic_frequency``. If this was changed, however, the results would have been incorrect.
- Fixed typo in the ``h_qc`` ingredient of ``HolsteinLatticeReciprocalSpace`` model affecting the calculation of the quantum-classical coupling Hamiltonian.
  Similar to the previous fix, the coupling terms now correctly scale with the ``classical_coordinate_weight`` parameter. This typo had no impact on the correctness of the ``HolsteinLatticeReciprocalSpace`` model where ``classical_coordinate_weight`` was set to ``harmonic_frequency``. If this was changed, however, the results would have been incorrect.
- Fixed critical typo in tasks.initialize_z_mcmc.
- Fixed typo in numerical_constants.INVCM_TO_300K (1/208.521 --> 1/208.512) (ALEX MAKE THIS CHANGE..)


New Capabilities
--------------------

- Adiabatic Model Objects
  - An Adiabatic Model object is one where the system is defined in an unspecified adiabatic basis in which the Hamiltonian of the quantum subsystem is 
    diagonal.
  - In an Adiabatic Model Object ``h_q`` is the identity operator and ``h_qc`` contains the adiabatic energies on its diagonal and ``h_c`` is the classical Hamiltonian.
  


Version 2.0.0
-----------------

- Change the diagonal_linear_coupling term so that the constant is not changing wrt h
- Move Model flags like ``update_dh_qc_dzc`` and ``update_h_q`` to Model class settings.
- Change force convention so the force is minus the gradient.


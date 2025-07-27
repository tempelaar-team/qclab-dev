.. _ingredients:

===========
Ingredients
===========

Ingredients are methods associated with model classes. A generic ingredient has the form:

.. code-block:: python

    def ingredient_name(model, parameters, **kwargs):
        # Calculate var.
        var = None
        return var


For consistency we include all of the arguments even if the ingredient does not use them.
An ingredient that generates the quantum Hamiltonian for a two-level system might look like this:

.. code-block:: python

    def two_level_system_h_q(model, parameters, **kwargs):
        """
        Quantum Hamiltonian for a two-level system.

        Required Constants:
            - two_level_system_00: Energy of the first level.
            - two_level_system_11: Energy of the second level.
            - two_level_system_01_re: Real part of the coupling between levels.
            - two_level_system_01_im: Imaginary part of the coupling between levels.

        Keyword Arguments:
            - batch_size: (Optional) Number of batches for vectorized computation.
        """
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(parameters.seed)
        h_q = np.zeros((batch_size, 2, 2), dtype=complex)
        h_q[:, 0, 0] = model.constants.two_level_system_00
        h_q[:, 1, 1] = model.constants.two_level_system_11
        h_q[:, 0, 1] = model.constants.two_level_sysmtem_c + 1j * model.constants.two_level_system_01_im
        h_q[:, 1, 0] = model.constants.two_level_system_01_re - 1j * model.constants.two_level_system_01_im
        return h_q


When incorporated directly into the model class (for instance when writing a model class from scratch) one should replace `model` with `self`. See the Model Development section of the User Guide
for a detailed example.


Ingredients supported by QC Lab
===============================

The following table lists the ingredients that are currently incorporated into QC Lab. Note that not all ingredients are utilized by every algorithm.
Indeed, the only required ingredients are those that generate the Hamiltonian.


.. list-table:: QC Lab Ingredients
   :widths: 10 20 20 20
   :header-rows: 1

   * - Ingredient Name
     - kwargs
     - Output
     - Description
   * - `h_q`
     - `batch_size` (optional)
     -  numpy.ndarray((batch_size, model.constants.num_quantum_states, model.constants.num_quantum_states), dtype=complex)
     - Generates the quantum Hamiltonian.
   * - `h_qc`
     - `batch_size` (optional), `z` (numpy.ndarray((batch_size, model.constants.num_classical_coordinates), dtype=complex))
     -  numpy.ndarray((batch_size, model.constants.num_quantum_states, model.constants.num_quantum_states), dtype=complex)
     - Generates the quantum-classical Hamiltonian.
   * - `h_c`
     - `batch_size` (optional), `z` (numpy.ndarray((batch_size, model.constants.num_classical_coordinates), dtype=complex))
     -  numpy.ndarray(batch_size, dtype=complex)
     - Generates the classical Hamiltonian.
   * - `init_classical`
     - `seed` (numpy.ndarray(batch_size, dtype=int))
     - numpy.ndarray((batch_size, model.constants.num_classical_coordinates), dtype=complex)
     - Generates the initial classical coordinate.
   * - `dh_qc_dzc`
     - `batch_size` (optional), `z` (numpy.ndarray((batch_size, model.constants.num_classical_coordinates), dtype=complex))
     -  numpy.ndarray((batch_size, model.constants.num_classical_coordinates, model.constants.num_quantum_states, model.constants.num_quantum_states), dtype=complex)
     - Generates the gradient (with respect to the conjugate classical coordinate) of the quantum-classical Hamiltonian.
   * - `dh_c_dzc`
     - `batch_size` (optional), `z` (numpy.ndarray((batch_size, model.constants.num_classical_coordinates), dtype=complex))
     -  numpy.ndarray((batch_size, model.constants.num_classical_coordinates), dtype=complex)
     - Generates the gradient (with respect to the conjugate classical coordinate) of the classical Hamiltonian.
   * - `hop_function`
     - `z` (numpy.ndarray(model.constants.num_classical_coordinates, dtype=complex)), `delta_z` (numpy.ndarray(model.constants.num_classical_coordinates, dtype=complex)), `ev_diff` (float)
     - numpy.ndarray(model.constants.num_classical_coordinates, dtype=complex)
     - Computes the shift required to rescale a coordinate for a given classical Hamiltonian in FSSH.
   * - `rescaling_direction_fssh`
     - `z` (numpy.ndarray(model.constants.num_classical_coordinates, dtype=complex)), `init_state_ind` (int), `final_state_ind` (int)
     - numpy.ndarray(model.constants.num_classical_coordinates, dtype=complex)
     - Computes the rescaling direction for the classical coordinates in FSSH.


Implementations available in QC Lab
===================================

The following ingredients are implemented in QC Lab and can be accessed through the `qc_lab.ingredients` module.



Quantum Hamiltonian (`h_q`)
---------------------------------

.. automodule:: qc_lab.ingredients
    :members: two_level_system_h_q, nearest_neighbor_lattice_h_q


Quantum-Classical Hamiltonian (`h_qc`)
--------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: diagonal_linear_h_qc

Classical Hamiltonian (`h_c`)
-----------------------------------

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_h_c

Classical Initialization (`init_classical`)
-------------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_boltzmann_init_classical, harmonic_oscillator_wigner_init_classical, harmonic_oscillator_coherent_state_wigner_init_classical


Quantum-Classical Gradients (`dh_qc_dzc`)
-------------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: diagonal_linear_dh_qc_dzc


Classical Gradients (`dh_c_dzc`)
---------------------------------

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_dh_c_dzc

FSSH Hop Function (`hop_function`)
--------------------------------------

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_hop_function

FSSH Rescaling Direction (`rescaling_direction_fssh`)
-------------------------------------------------

.. automodule:: qc_lab.ingredients
    :members: harmonic_oscillator_rescaling_direction_fssh
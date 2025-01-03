.. _ingredients:

Ingredients
-----------

Ingredients are are methods associated with model classes. A generic ingredient has the form:

.. code-block:: python

    def label_var_options(model, **kwargs):
        return var

Here, label is a descriptor for the ingredient, var is the name of a variable used by QC Lab, and options is used to specify additional 
classifications of the ingredient. Examples for var include 
the quantum Hamiltonian `h_q`, the classical Hamiltonian `h_c`, and the quantum-classical Hamiltonian `h_qc`. An ingredient that 
generates the quantum-classical Hamiltonian for the spin-boson model might look like this:

.. code-block:: python

    def spin_boson_h_qc(model, **kwargs):
        z_coord = kwargs['z_coord']
        g = model.parameters.g
        m = model.parameters.mass
        h = model.parameters.pq_weight
        h_qc = np.zeros((2, 2), dtype=complex)
        h_qc[0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h))) * (z_coord + np.conj(z_coord)))
        h_qc[1, 1] = -h_qc[0, 0]
        return h_qc

The "options" part of the ingredient name is a string that is used to specify additional options for the ingredient. For example,
QC Lab treates vectorized and non-vectorized ingredients differently. The options string for a vectorized ingredient is "vectorized".

.. code-block:: python

    def spin_boson_h_qc_vectorized(model, **kwargs):
        z_coord = kwargs['z_coord']
        g = model.parameters.g
        m = model.parameters.mass
        h = model.parameters.pq_weight
        h_qc = np.zeros((*np.shape(z_coord)[:-1], 2, 2), dtype=complex)
        h_qc[..., 0, 0] = np.sum((g * np.sqrt(1 / (2 * m * h)))[..., :] * (z_coord + np.conj(z_coord)), axis=-1)
        h_qc[..., 1, 1] = -h_qc[..., 0, 0]
        return h_qc


When incorporated directly into the model class one should replace `model` with `self` and the name of the method should be `model.var`. See the :ref:`model_class` section 
for a detailed example.


Below we list all of the ingredients available in the current version of QC Lab and group ingredients by the attribute of the model that they pertain to. 



Current Ingredients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantum Hamiltonian
^^^^^^^^^^^^^^^^^^^


.. automodule:: qclab.ingredients
    :members: two_level_system_h_q_vectorized, nearest_neighbor_lattice_h_q_vectorized, spin_boson_h_q_vectorized

Quantum-Classical Interaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ingredients that generate quantum-classical interaction terms.

.. automodule:: qclab.ingredients
    :members: holstein_lattice_h_qc_vectorized, holstein_lattice_dh_qc_dzc_vectorized

Classical Hamiltonian
^^^^^^^^^^^^^^^^^^^^^

Ingredients that generate classical Hamiltonians. 

.. automodule:: qclab.ingredients
    :members: harmonic_oscillator_h_c_vectorized, harmonic_oscillator_dh_c_dzc_vectorized, harmonic_oscillator_hop 

Classical Initialization 
^^^^^^^^^^^^^^^^^^^^^^^^

Ingredients that initialize the complex-valued classical coordinates.

.. automodule:: qclab.ingredients
    :members: harmonic_oscillator_wigner_init_classical, harmonic_oscillator_boltzmann_init_classical



Deprecated and Non-Vectorized Ingredients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Please see the source code. 

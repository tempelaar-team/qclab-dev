.. _change-model:


Changing a model in QC Lab
==========================

Swapping out a model in QC Lab is straightforward. Supose you've run a spin-boson model with the following code:


.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from qc_lab import Simulation
    from qc_lab.models import SpinBoson
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver


    # Initialize the simulation object.
    sim = Simulation()
    # Equip it with a SpinBoson model object.
    sim.model = SpinBoson()
    # Attach the MeanField algorithm.
    sim.algorithm = MeanField()
    # Initialize the diabatic wavefunction.
    sim.state.wf_db = np.array([1, 0], dtype=complex)
    # Run the simulation.
    data_spinboson = serial_driver(sim)

You can change the model to a different one by first importing the new model class, attaching it to the simulation object, 
changing the initial state if necessary, and then running the simulation again.

.. code-block:: python
    from qc_lab.models import HolsteinLattice
    sim.model = HolsteinLattice()
    # use the input constant 'N' to set the number of sites (this is specific to the HolsteinLattice model)
    sim.state.wf_db = np.zeros(sim.model.constants.N, dtype=complex)
    sim.state.wf_db[0] = 1  # Set the first site to be occupied
    data_holstein = serial_driver(sim)
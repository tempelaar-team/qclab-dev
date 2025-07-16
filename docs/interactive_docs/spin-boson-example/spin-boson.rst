.. _spin-boson:

===========================
Running a Spin-Boson Model
===========================

Here's a simple example of how to run a Spin-Boson model with Mean-Field dynamics in QC Lab.


First, we will need to import the necessary modules:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from qc_lab import Simulation
    from qc_lab.models import SpinBoson
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver


Next, we will set up the simulation object and equip it with the model and algorithm objects:

.. code-block:: python

    # Initialize the simulation object.
    sim = Simulation()
    # Equip it with a SpinBoson model object.
    sim.model = SpinBoson()
    # Attach the MeanField algorithm.
    sim.algorithm = MeanField()
    # Initialize the diabatic wavefunction. 
    # Here, the first state is the upper state and the second is the lower state.
    sim.state.wf_db = np.array([1, 0], dtype=complex)
    

Finally, we can run the simulation and visualize the results:

.. code-block:: python

    # Run the simulation.
    data = serial_driver(sim)
   
    # Pull out the time.
    t = data.data_dict["t"]
    # Get populations from the diagonal of the density matrix.
    populations = np.real(np.einsum("tii->ti", data.data_dict["dm_db"]))
    plt.plot(t, populations[:, 0], color="blue",label='MF')
    plt.xlabel('Time')
    plt.ylabel('Excited state population')
    plt.ylim([0.4,1.01])
    plt.legend(frameon=False)
    plt.show()

    
The output of this code is:

.. image:: mf.png
    :alt: Population dynamics.
    :align: center
    :width: 50%
    

I want to increase the reorganization energy.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: model-constants.rst


I want to use FSSH instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: change-algorithm.rst


I want to invert velocities after frustrated hops.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: modify-fssh.rst


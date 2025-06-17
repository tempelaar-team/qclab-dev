.. _change-model:


Changing an algorithm in QC Lab
===============================

Swapping out an algorithm in QC Lab is straightforward. Supose you've run a spin-boson model using the MeanField algorithm 

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
    data_mf = serial_driver(sim)

You can change change the algorithm to a different one by first importing the new algorithm class, attaching it to the simulation object,
and rerunnign the simulation. In the following example, we change the algorithm to `FewestSwitchesSurfaceHopping`.

.. code-block:: python
    from qc_lab.algorithms import FewestSwitchesSurfaceHopping
    sim.algorithm = FewestSwitchesSurfaceHopping()
    sim.state.wf_db = np.array([1, 0], dtype=complex)
    data_fssh = serial_driver(sim)


We can then plot the results from both algorithms to compare their performance.

.. code-block:: python

    plt.plot(data_mf.data_dict["t"], np.real(data_mf.data_dict["dm_db"][:,0,0]), label='MF')
    plt.plot(data_fssh.data_dict["t"], np.real(data_fssh.data_dict["dm_db"][:,0,0]), label='FSSH')
    plt.xlabel('Time')
    plt.ylabel('Excited state population')
    plt.legend()
    plt.show()


.. image:: algorithm_comparison.png
    :alt: Population dynamics.
    :align: center
    :width: 80%


.. note::

    The populations above are not in agreement at the outset of the simulation because the FSSH algorithm 
    stochastically samples the initial state while the MF algorithm does not. If the number of trajectories 
    were increased, the populations would converge to the same value as the `MeanField` algorithm at the outset of the simulation.
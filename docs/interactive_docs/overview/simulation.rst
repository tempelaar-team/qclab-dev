.. _simulation:

The Simulation Object
========================

The simulation object in QC Lab is the vehicle for sending a selected model and algorithm (as well as 
any settings) to the dynamics driver for execution. 

It is instantiated as follows:


.. code-block:: python

    from qc_lab import Simulation

    sim = Simulation(settings={})

The simulation object has a number of settings that can be adjusted by passing a dictionary to the argument `settings`:

.. code-block:: python

    sim = Simulation(settings={
        'num_trajs': 1000,  # Number of trajectories to run
        'batch_size': 100,  # Size of each batch to run at once
        'tmax': 10.0,       # Maximum time for the simulation
        'dt_update': 0.01,  # Time step for updates
        'dt_gather': 0.1,   # Time step for gathering results
    })  

You can see a list of the available settings and an example of changing them for a spin-boson model in `Simulation Settings <../spin-boson-example/simulation-settings.html>`_.
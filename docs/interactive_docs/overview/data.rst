.. _data:

The Data Object
========================


The data object in QC Lab is used to store the results of a simulation, it can be imported from `qc_lab.Data`. Typically, it is created by the dynamics driver
and contains the output variables in the dictionary `data.data_dict`. 

The `data.data_dict` dictionary also contains the time array `t` giving the time points at which the output variables were 
calculated, as well as the `norm_factor` which is used to normalize the output variables, and the `seed` array which contains
the random seeds used in the simulation.

The data object also contains the method `data.add_data` which can be used to combine two data objects. 

Additionally, it contains methods that allwo you to write the data to an `h5` archive and also to read a written archive back into a data object.

These use cases are demonstrated in the following example:


.. code-block:: python

    import numpy as np
    from qc_lab import Simulation, Data
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
    data = serial_driver(sim)
    # Run it again on the last data object
    data = serial_driver(sim, data=data)

    # Print the keys of the data dictionary.
    print(data.data_dict.keys())

    # Write the data to an h5 archive.
    data.save("spin_boson_data.h5")

    # Read the data back into a data object.
    data_read = Data().load("spin_boson_data.h5")

More detailed documentation of the data object can be found in the `Data Object <../../user_guide/data_object.html>`_ documentation page.
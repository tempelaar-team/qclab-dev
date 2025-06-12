.. _model-constants:

Model Constants
=================
Each model class has a constant object, `model.constants` that stores the constants defining a model.
In the spin-boson model, these constants are derived from the definition of the Hamiltonian which can be found in the
 `Spin-Boson Model Documentation <../user_guide/models/spin_boson_model.html>`_.

The table below lists the available constants and their default values.


.. list-table:: Spin-Boson Model Constants
   :widths: 20 20 20 40
   :header-rows: 1

   * - Symbol
     - Name
     - Default value
     - Description
   * - :math:`\Omega`
     - `W`
     - 0.1
     - Characteristic frequency.
   * - :math:`\lambda`
     - `l_reorg`
     - 0.005
     - Reorganization energy.
   * - :math:`E`
     - `E`
     - 0.5
     - Diagonal energy.
   * - :math:`V`
     - `V`
     - 0.5
     - Off-diagonal coupling.
   * - :math:`A`
     - `A`
     - 100
     - Number of bosons.
   * - :math:`m`
     - `boson_mass`
     - 1
     - boson mass.
   * - :math:`k_B T`
     - `kBT`
     - 1
     - Thermal quantum.


The constants object can be initialized with custom values by passing a dictionary to the `SpinBoson` model class at instantiation.

:: 

    from qc_lab.models import SpinBoson
    from qc_lab import Simulation
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver
    import numpy as np
    # Instantiate a simulation
    sim = Simulation()
    # Instantiate a model with custom constants
    sim.model = SpinBoson(constants={
        'W': 0.2,          # Characteristic frequency
        'l_reorg': 0.01,   # Reorganization energy
        'E': 0.6,          # Diagonal energy
        'V': 0.4,          # Off-diagonal coupling
        'A': 150,          # Number of bosons
        'boson_mass': 1.5, # Mass of the bosons
        'kBT': 1.2         # Thermal quantum
    })
    # Instantiate an algorithm
    sim.algorithm = MeanField()
    # Set the initial diabatic wavefunction
    sim.state.wf_db = np.array([1, 0], dtype=complex)
    # Run the simulation using the serial driver
    data = serial_driver(sim)


Alternatively, you can set the constants directly in the model after instantiation.

:: 

    from qc_lab.models import SpinBoson
    from qc_lab import Simulation
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver
    import numpy as np
    # Instantiate a simulation
    sim = Simulation()
    # Instantiate a model with default constants
    sim.model = SpinBoson()
    # Change the constants directly
    sim.model.constants.W = 0.2
    # Instantiate an algorithm
    sim.algorithm = MeanField()
    # Set the initial diabatic wavefunction
    sim.state.wf_db = np.array([1, 0], dtype=complex)
    # Run the simulation using the serial driver
    data = serial_driver(sim)
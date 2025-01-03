.. _holstein_model:

Holstein Lattice Model
~~~~~~~~~~~~~~~~~~~~~~

The Holstein Lattice Model is a nearest-neighbor tight-binding model combined with an idealized optical phonon that interacts via a 
Holstein coupling. The current implementation accommodates a single electronic particle. The quantum-classical Hamiltonian of the Holstein model is:

.. math::
    
    \hat{H}_{\mathrm{q}} = -t\sum_{\langle i,j\rangle}^{N}\hat{c}^{\dagger}_{i}\hat{c}_{j}

where :math:`\langle i,j\rangle` denotes nearest-neighbor sites with or without periodic boundaries determined by the parameter `periodic_boundary=True`.

.. math::

    \hat{H}_{\mathrm{q-c}} = g\omega\sum_{i}^{N} \hat{c}^{\dagger}_{i}\hat{c}_{i} \frac{1}{\sqrt{2mh_{i}}} \left(z^{*}_{i} + z_{i}\right)

.. math::

    H_{\mathrm{c}} = \omega \sum_{i}^{N} z^{*}_{i} z_{i}

Here, :math:`g` is the dimensionless electron-phonon coupling, :math:`\omega` is the phonon frequency, :math:`m` is the phonon mass, and :math:`h_{i}` is 
the complex-valued coordinate parameter which we take to be :math:`h_{i} = \omega`. 

The classical coordinates are sampled from a Boltzmann distribution:

.. math::

    P(z) \propto \exp\left(-\frac{H_{\mathrm{c}}(\boldsymbol{z})}{T}\right)

and by convention we assume that :math:`\hbar = 1`, :math:`k_{B} = 1`.

Parameters
----------

The following table lists all of the parameters required by the `HolsteinLatticeModel` class:

.. list-table:: HolsteinLatticeModel Parameters
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `temp` :math:`(T)`
     - Temperature
     - 1
   * - `g` :math:`(g)`
     - Dimensionless electron-phonon coupling
     - 0.5
   * - `w` :math:`(\omega)`
     - Phonon frequency
     - 0.5
   * - `N` :math:`(N)`
     - Number of sites
     - 10
   * - `t` :math:`(t)`
     - Hopping energy
     - 1
   * - `phonon_mass` :math:`(m)`
     - Phonon mass
     - 1
   * - `periodic_boundary`
     - Periodic boundary condition
     - True

     
Example
-------

::

    from qclab.models import HolsteinLatticeModel
    from qclab import Simulation
    from qclab.algorithms import MeanField
    from qclab.dynamics import serial_driver
    import numpy as np

    # instantiate a simulation
    sim = Simulation()

    # instantiate a model 
    sim.model = HolsteinLatticeModel()

    # instantiate an algorithm 
    sim.algorithm = MeanField()

    # define an initial diabatic wavefunction 
    wf_db_0 = np.zeros((sim.model.parameters.N), dtype=np.complex128)
    wf_db_0[0] = 1.0 + 0.0j
    sim.state.modify('wf_db',wf_db_0)

    # run the simulation
    data = serial_driver(sim)
.. _holstein_model:

Holstein Lattice Model
~~~~~~~~~~~~~~~~~~~~~~

The Holstein Lattice Model is a nearest-neighbor tight-binding model combined with an idealized optical phonon that interacts via a 
Holstein coupling. The current implementation accommodates a single electronic particle. The quantum-classical Hamiltonian of the Holstein model is:

.. math::
    
    \hat{H}_{\mathrm{q}} = -j\sum_{\langle i,j\rangle}^{N}\hat{c}^{\dagger}_{i}\hat{c}_{j}

where :math:`\langle i,j\rangle` denotes nearest-neighbor sites with or without periodic boundaries determined by the parameter `periodic_boundary=True`.

.. math::

    \hat{H}_{\mathrm{q-c}} = g\omega\sum_{i}^{N} \sqrt{\frac{\omega}{h_{i}}}\hat{c}^{\dagger}_{i}\hat{c}_{i} \left(z^{*}_{i} + z_{i}\right)

.. math::

    H_{\mathrm{c}} = \sum_{i}^{N} a_{i}z_{i}^{2} + 2 b_{i}z^{*}_{i}z_{i} + a_{i}z_{i}^{*2}


where :math:`a_{i}=\frac{1}{4}\left(\frac{\omega^{2}}{h_{i}}-h_{i}\right)` and :math:`b_{i}=\frac{1}{4}\left(\frac{\omega^{2}}{h_{i}}+h_{i}\right)`.

Here, :math:`g` is the dimensionless electron-phonon coupling, :math:`\omega` is the phonon frequency, :math:`m` is the phonon mass, and :math:`h_{i}` is 
the classical coordinate weight which may take on arbitrary (nonzero) values. 

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
   * - `j` :math:`(j)`
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

    from qc_lab.models import HolsteinLattice
    from qc_lab import Simulation
    from qc_lab.algorithms import MeanField
    from qc_lab.dynamics import serial_driver
    import numpy as np

    # instantiate a simulation
    sim = Simulation()

    # instantiate a model 
    sim.model = HolsteinLattice()

    # instantiate an algorithm 
    sim.algorithm = MeanField()

    # define an initial diabatic wavefunction 
    wf_db_0 = np.zeros((sim.model.parameters.N), dtype=np.complex128)
    wf_db_0[0] = 1.0 + 0.0j
    sim.state.wf_db = wf_db_0

    # run the simulation
    data = serial_driver(sim)
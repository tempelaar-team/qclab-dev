.. _spin_boson_model:

Spin-Boson Model
~~~~~~~~~~~~~~~~

The quantum-classical Hamiltonian of the spin-boson model is:

.. math::
    
    \hat{H}_{\mathrm{q}} = \left(\begin{array}{cc} -E & V \\ V & E \end{array}\right)

.. math::

    \hat{H}_{\mathrm{q-c}} = \sigma_{z} \sum_{\alpha}^{A}  \frac{g_{\alpha}}{\sqrt{2mh_{\alpha}}} \left(z^{*}_{\alpha} + z_{\alpha}\right)

.. math::

    H_{\mathrm{c}} = \sum_{\alpha}^{A} \omega_{\alpha} z^{*}_{\alpha} z_{\alpha}

The couplings and frequencies are sampled from a Debye spectral density:

.. math::

    \omega_{\alpha} = \Omega\tan\left(\frac{\alpha - 1/2}{2A}\pi\right)

.. math::

    g_{\alpha} = \omega_{\alpha}\sqrt{\frac{2\lambda}{A}}

Where :math:`\Omega` is the characteristic frequency and :math:`\lambda` is the reorganization energy. 

The classical coordinates are sampled from a Boltzmann distribution:

.. math::

    P(z) \propto \exp\left(-\frac{H_{\mathrm{c}}(\boldsymbol{z})}{T}\right)

and by convention we assume that :math:`\hbar = 1`, :math:`k_{B} = 1`, and :math:`\omega_{\alpha} = h_{\alpha}`.

Parameters
----------

The following table lists all of the parameters required by the `SpinBosonModel` class:

.. list-table:: SpinBosonModel Parameters
   :header-rows: 1

   * - Parameter (symbol)
     - Description
     - Default Value
   * - `temp` :math:`(T)`
     - Temperature
     - 1
   * - `V` :math:`(V)`
     - Off-diagonal coupling
     - 0.5
   * - `E` :math:`(E)`
     - Diagonal energy
     - 0.5
   * - `A` :math:`(A)`
     - Number of bosons
     - 100
   * - `W` :math:`(\Omega)`
     - Characteristic frequency
     - 0.1
   * - `l_reorg` :math:`(\lambda)`
     - Reorganization energy
     - 0.005
   * - `boson_mass` :math:`(m)`
     - Mass of the bosons
     - 1


Example
-------

::

    from qclab.models.spin_boson import SpinBosonModel
    from qclab.simulation import Simulation
    from qclab.algorithms.mean_field import MeanField
    from qclab.drivers.serial_driver import run_simulation
    import numpy as np

    # instantiate a simulation
    sim = Simulation()

    # instantiate a model 
    sim.model = SpinBosonModel()

    # instantiate an algorithm 
    sim.algorithm = MeanField()

    # define an initial diabatic wavefunction 
    sim.state.modify('wf_db',np.array([1, 0], dtype=np.complex128))

    # run the simulation
    data = run_simulation(sim)
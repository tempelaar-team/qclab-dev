.. _data:

The Data Object
========================


The data object in QC Lab is used to store the results of a simulation. Typically, it is created by the dynamics driver
and contains the following attributes:

.. list-table:: Data object attributes
   :widths: 20 20 60
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``'t'``
     - `np.ndarray`
     - Time points at which the data was gathered.
   * - ``'wf_db'``
     - `np.ndarray`
     - Diabatic wavefunction at each time point.
   * - ``'wf_ad'``
     - `np.ndarray`
     - Adiabatic wavefunction at each time point.
   * - ``'z'``
     - `np.ndarray`
     - Classical coordinates at each time point.
   * - ``'p'``
     - `np.ndarray`
     - Classical momenta at each time point.
   * - ``'populations'``
     - `np.ndarray`
     - Populations of the diabatic states at each time point.
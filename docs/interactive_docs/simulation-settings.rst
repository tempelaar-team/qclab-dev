.. _simulation-settings:


Simulation settings
===================

The simulation object stores its settings in a dictionary at `Simulation.settings`.
These settings can be modified before running the simulation to customize the behavior of the simulation.
The table below lists the available settings and their default values.


.. list-table:: Simulation settings
   :widths: 20 20 60
   :header-rows: 1

   * - Setting
     - Default value
     - Description
   * - ``'dt'``
     - 0.001
     - Time step for the simulation.
   * - ``'dt_output'``
     - 0.1
     - Time step for the simulation.
   * - ``'tmax'``
     - 10
     - Maximum simulation time.
   * - ``'num_trajs'``
     - 100
     - Total number of trajectories to run.
   * - ``'batch_size'``
     - 25
     - Number of trajectories to run in each batch.

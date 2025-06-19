.. _driver:

====================
Dynamics Driver
====================


.. figure:: _static/dynamics_driver_diagram.svg
    :alt: QC Lab Dynamics Driver Diagram
    :width: 100%
    :align: center
    :name: dynamics-driver-diagram

The **Dynamics Driver** is a central component in QC Lab that orchestrates the execution of quantum dynamics simulations by running (or driving) the 
**Dynamics Core**. It manages the flow of data between the simulation model, algorithm, and data storage, ensuring that all components work together seamlessly.

QC Lab comes with several built-in dynamics drivers:

- **Serial Driver**: Executes the dynamics in a single process, suitable for small to medium-sized simulations.
- **Parallel Driver (multiprocessing)**: Executes the dynamics in parallel using the python `multiprocessing` module,
allowing for efficient use of multiple CPU cores on a single machine.
- **Parallel Driver (MPI)**: Executes the dynamics in parallel across multiple processes using MPI (Message Passing Interface), 
suitable for high-performance computing environments.
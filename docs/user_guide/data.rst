.. _data:

==========================
Data
==========================

Data Objects
---------------------------

Data objects are instances of the `qclab.data.Data` class and are used to store and manage the results of a simulation. They provide methods for collecting, processing, and saving data, as well as logging errors or warnings during a simulation.

In general, a data object has the following attributes:

- ``data_dict``: a dictionary that stores the results of the simulation. Each key in the dictionary corresponds to a specific quantity that was collected during the simulation, and the value is an array containing the values of that quantity averaged over the trajectories.
- ``log``: a string that stores the log of errors or warnings that occurred during the simulation.

Data objects provide several methods for managing and processing the data they contain. Some of the most important methods include:

- ``add_data``: adds data from an existing data object to the current one.
- ``save``: saves the data object to a file.
- ``load``: loads a data object from a file (this adds to any existing data).
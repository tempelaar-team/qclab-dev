.. _algorithms:

Algorithms
----------

Algorithms in QC Lab are classes that can be paired with models to carry out a quantum-classical dynamics simulation.
Like models, each algorithm in QC Lab has settings that allow users to control particular attributes of the 
algorithm. The contents and structure of an algorithm class are described in the Developer Guide.

The algorithms currently available in QC Lab can be imported from the algorithms module:

.. code-block:: python

    from qclab.algorithms import MeanField

After which they can be instantiated with a dictionary of settings:

.. code-block:: python
    
    algorithm = MeanField(settings=None)

Here there are no settings passed to the `MeanField` algorithm.

.. toctree::
    :maxdepth: 1
    :caption: Algorithms

    mf_algorithm
    fssh_algorithm
.. _spin-boson:

===========================
Running a Spin-Boson Model
===========================

Here's a simple example of how to run a Spin-Boson model with Mean-Field dynamics in QC Lab.



.. literalinclude:: basic_spin_boson.py
   :language: python
   :caption: Simple Example.


The output of this code is:

.. image:: mf.png
    :alt: Population dynamics.
    :align: center
    :width: 50%
    

I want to increase the reorganization energy.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: model-constants.rst


I want to use FSSH instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: change-algorithm.rst


I want to invert the momenta of frustrate hops.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: modify-fssh.rst


.. comment::

    I want to run more trajectories.
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. container:: toggle

        .. include:: simulation-settings.rst

    I want it to run faster!
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. container:: toggle

        .. include:: parallel-driver.rst




    Wrapping it all together.
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    .. container:: toggle

        .. literalinclude:: full_example.py
            :language: python
            :caption: This Full example.



    
    .. dropdown:: I want to use the FSSH algorithm instead.

        .. include:: change-algorithm.rst

    .. dropdown:: I want to run more trajectories.

        .. include:: simulation-settings.rst

    .. dropdown:: I want it to run faster!

        .. include:: parallel-driver.rst

    .. dropdown:: I want to change the reorganization energy.

        .. include:: model-constants.rst

    .. dropdown:: I want to modify the FSSH algorithm.

        .. include:: modify-fssh.rst


    .. dropdown:: Wrapping it all together.

        .. literalinclude:: full_example.py
            :language: python
            :caption: This Full example.


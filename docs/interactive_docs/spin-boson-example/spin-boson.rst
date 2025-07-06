.. _spin-boson:

===========================
Running a Spin-Boson Model
===========================

Here's a simple example of how to run a Spin-Boson model with Mean-Field dynamics in QC Lab.



.. literalinclude:: basic_spin_boson.py
   :language: python
   :caption: Simple Example.


The output of this code is:

.. image:: populations.png
    :alt: Population dynamics.
    :align: center
    :width: 50%

.. image:: energies.png
    :alt: Change in energy.
    :align: center
    :width: 50%
    
.. comment::

    .. button-ref:: change-algorithm
        :color: primary
        :shadow:
        :align: center

        I want to use the FSSH algorithm instead.

.. raw:: html

   <br/>

I want to use the FSSH algorithm instead.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: change-algorithm.rst

I want to run more trajectories.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: simulation-settings.rst

I want it to run faster!
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: parallel-driver.rst

I want to change the reorganization energy.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: model-constants.rst


I want to modify the FSSH algorithm.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. include:: modify-fssh.rst

Wrapping it all together.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: toggle

    .. literalinclude:: full_example.py
        :language: python
        :caption: This Full example.


.. comment::

    
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


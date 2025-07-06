.. _parallel-driver:

I want it to run faster!
================================


Okay! We have so far only used the `serial_driver` to run our simulations, which runs one batch of trajectories at a time.
However, we can use the `parallel_driver` to run the simulation in parallel across multiple processes, note that this will only 
work if you have multiple CPU cores available.

.. code-block:: python

    from qc_lab.dynamics import serial_driver, parallel_driver_multiprocessing
    import time

    sim.settings.num_trajs = 1000
    sim.settings.batch_size = 125 # split them up into batches of 125

    st = time.time()
    data_parallel = parallel_driver_multiprocessing(sim)
    et = time.time()
    print(f"Parallel driver took {et-st:.2f} seconds to run.")
    st = time.time()
    data_serial = serial_driver(sim)
    et = time.time()
    print(f"Serial driver took {et-st:.2f} seconds to run.")



You should find that the results are exactly the same as before, but that the simulation runs faster. 
The speedup will in general be less than ideal because of the overhead and also how your particular machine 
is configured. But on my machine with 8 cores, I find that the speedup is about 3x faster than the serial driver.

For an overview of the different drivers available in QC Lab, please refer to the `Drivers documentation <../../user_guide/drivers/drivers.html>`_.


.. comment::

    .. button-ref:: model-constants
        :color: primary
        :shadow:
        :align: center

        I want to change the reorganization energy.
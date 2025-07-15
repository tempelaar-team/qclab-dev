.. _variable_object:


Variable Object
-----------

The variable object in QC Lab is used to store varibles in a simulation. It is generally created internally by the dynamics drivers calling:

.. code-block:: python

    from qc_lab.variable import initialize_variable_objects
    
    parameters, state = initialize_variable_objects(sim, seeds)


which results in two variable objects: `parameters` and `state`. The `sim` argument is the simulation object, and the `seeds` argument
 is a list of seeds for each trajectory. Any quantities initially in `sim.state` are now stored in the `state` variable object with a new 
 index that defines the trajectory it is associated with. , and any quantities in `sim.parameters` are now stored in the `parameters` variable object.


Within an algorithm, two variable objects are used: the `state` variable object, which contains dynamic quantities like the wavefunction and 
classical coordiantes which define th state of the system, and the `parameters` variable object, which contains potentially time-dependent 
parameters needed in the simulation. We conceptually think of parameters as quantities which are not differentiated against but may still change in time.


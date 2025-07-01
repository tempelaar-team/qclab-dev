.. _tasks:


Tasks in QC Lab
=================

In QC Lab, algorithms consist of lists of functions, called **tasks**, that are executed in an order corresponding
 to their ordering in the list (recipe).
These tasks are the building blocks of the algorithm and define how different quantities are manipulated to 
evolve the quantum-classical system.
Tasks have a very well-defined structure, which allows them to be easily defined and executed, 
as well as to be reused across different algorithms.

QC Lab comes with a number of built-in tasks that enable the existing algorithms to work. These can be 
accessed through the `qc_lab.tasks` module.

You can also define your own tasks to implement new algorithms or modify existing ones.

A generic task has the form:


.. code-block:: python

    def generic_task(algorithm, sim, parameters, state):
        """
        Docstring for the task
        """
        # Perform the task's operations here

        # Update the state or parameters as needed

        return parameters, state


This generic task could be inserted directly into an algorithm's recipe, as done in `Modifying the FSSH algorithm <../spin-boson-example/modify-fssh.html>`_.


Most of the tasks in `qc_lab.tasks` have additional keyword arguments that can be used to customize their behavior. As a result, 
before being used they have to be wrapped in an outer task that conforms to the above structure.

In the following example, we define a task that updates the classical energy of the system but requires the user to specify the 
set of classical coordinates to use, the wrapper function selects the coordinates from the state object and passes them to the task:

.. code-block:: python


    def update_classical_energy(algorithm, sim, parameters, state, **kwargs):
        """
        Update the classical energy.

        Required constants:
            - None.
        """
        z = kwargs["z"]
        h_c, _ = sim.model.get("h_c")
        state.classical_energy = np.real(h_c(sim.model, parameters, z=z, batch_size=len(z)))
        return parameters, state

    def update_classical_energy_z(algorithm, sim, parameters, state):
        """
        Wrapper to specify which z to use.
        """
        # Define the task-specific parameters
        return update_classical_energy(
            algorithm, sim, parameters, state, z=state.z
        )

We can then use update_classical_energy_z in an algorithm's recipe or modify its behavior again by modifying the wrapper function
while leaving the underlying task unchanged. This allows for a high degree of flexibility in defining and modifying tasks.
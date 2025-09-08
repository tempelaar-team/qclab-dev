.. tasks:

Tasks
-----

Tasks are methods of an Algorithm object that carry out the elementary steps of an algorithm by modifying 
quantities in the `state` and `parameters` Variable objects. They have a standardized 
set of inputs and outputs as well as an unspecied set of keyword arguments. As an example,
consider the task that evolves a wavefunction one timestep forward:


.. code-block:: python

    def update_wf_rk4(algorithm, sim, parameters, state, **kwargs):
        """
        Update the wavefunction using the 4th-order Runge-Kutta method.

        Required constants:
            None.
        """
        wf = getattr(state, kwargs["wf"])
        dt_update = sim.settings.dt_update
        h_quantum = state.h_quantum
        setattr(state, kwargs["wf"], wf_rk4(h_quantum, wf, dt_update))
        return parameters, state


.. code-block:: python

   def example_task(algorithm, sim, parameters, state, **kwargs):
       # Task implementation goes here
       # my_state_var = getattr(state, kwargs["state_var"])
       return parameters, state


Because tasks are methods of an Algorithm object, they have access to the algorithm instance and can modify its state.
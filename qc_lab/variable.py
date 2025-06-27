"""
This file defines the Variable class.
"""

import numpy as np


class Variable:
    """
    The Variable object defines a vehicle in which quantities can be placed, retrieved, and collected.
    """

    def __init__(self):
        self.output_dict = {}
        self.seed = None

    def __getattr__(self, name):
        """
        If an attribute is not in the Variable object it returns None rather than throwing an error.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def collect_outputs(self, output_names):
        """
        Collect varibles with names output_names into the output dictionary.

        Args:
            output_names: List of output names.
        """
        for var in output_names:
            self.output_dict[var] = getattr(self, var)

    def __getstate__(self):
        """Called when pickling the object."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)



def initialize_variable_objects(sim, batch_seeds):
    """
    This takes the numpy arrays inside the state and parameter objects
    and creates a first index corresponding to the number of trajectories.
    """
    state_variable = Variable()
    state_variable.seed = batch_seeds
    for name in sim.state.__dict__.keys():
        obj = getattr(sim.state, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(batch_seeds), *init_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            setattr(state_variable, name, new_obj)
    parameter_variable = Variable()
    for name in sim.model.parameters.__dict__.keys():
        obj = getattr(sim.model.parameters, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(batch_seeds), *init_shape), dtype=obj.dtype)
                + obj[np.newaxis]
            )
            setattr(parameter_variable, name, new_obj)
    return parameter_variable, state_variable

"""
This module defines the Variable class, which is used to store time-dependent variables in QC Lab.
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

    def collect_outputs(self, names):
        """
        Collect attributes into the output dictionary.

        Args:
            names: List of attribute names.
        """
        for var in names:
            self.output_dict[var] = getattr(self, var)

    def __getstate__(self):
        """Called when pickling the object."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def initialize_variable_objects(sim, seeds):
    """
    This generates the state and parameter objects for the simulation.
    Each object in sim.state is initialized as a numpy array in the 
    state object with a new first index corresponding to the number of trajectories
    as specified by the length of the seeds array.
    """
    state_variable = Variable()
    state_variable.seed = seeds
    for name in sim.state.__dict__.keys():
        obj = getattr(sim.state, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(seeds), *init_shape), dtype=obj.dtype) + obj[np.newaxis]
            )
            setattr(state_variable, name, new_obj)
    parameter_variable = Variable()
    return parameter_variable, state_variable

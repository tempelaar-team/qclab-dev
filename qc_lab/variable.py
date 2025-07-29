"""
This module defines the Variable class, which is used to store time-dependent variables in QC Lab.
"""

import numpy as np


class Variable:
    """
    The Variable class provides an object in which time-dependent quantities can be stored.
    """

    def __init__(self):
        self.output_dict = {}
        self.seed = None

    def __getattr__(self, name):
        """
        Return ``None`` for missing attributes.

        Args:
            name (str): Attribute name.

        Returns:
            Any | None: The attribute value if present, otherwise ``None``.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def collect_outputs(self, names):
        """
        Collect attributes into the output dictionary.

        Args:
            names (Iterable[str]): List of attribute names.
        """
        for var in names:
            self.output_dict[var] = getattr(self, var)

    def __getstate__(self):
        """
        Support pickling of the object.

        Returns:
            dict: The instance dictionary used for pickling.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Restore state during unpickling.

        Args:
            state (dict): Object state.
        """
        self.__dict__.update(state)


def initialize_variable_objects(sim, seeds):
    """
    Generate the ``parameter`` and ``state`` variables for a simulation.

    Args:
        sim (Simulation): The simulation instance.
        seeds (Iterable[int]): Array of trajectory seeds.

    Returns:
        tuple[Variable, Variable]: The ``parameter`` and ``state`` objects.
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

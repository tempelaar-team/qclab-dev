"""
This module defines the Variable class, which is used to store time-dependent variables
in QC Lab.

It also defines a function to initialize parameter and state Variable objects for
simulations.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class Variable:
    """
    Variable class for storing time-dependent variables.
    """

    def __init__(self):
        self.output_dict = {}

    def __getattr__(self, name):
        """
        Return `None` for missing attributes.

        Args:
            name (str): Attribute name.

        Returns:
            Any | None: The attribute value if present, otherwise `None`.
        """
        if name in self.__dict__:
            return self.__dict__[name]
        return None

    def __getstate__(self):
        """
        Support pickling of the object.

        Returns:
            dict: The instance dictionary used for pickling.
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Restore state during unpickling.

        Args:
            state (dict): Object state.
        """
        self.__dict__.update(state)


def initialize_variable_objects(sim, seed):
    """
    Generate the `parameter` and `state` variables for a simulation.

    Args:
        sim (Simulation): The simulation instance.
        seed (Iterable[int]): Array of trajectory seeds.

    Returns:
        tuple[Variable, Variable]: The `parameter` and `state` objects.
    """
    state_variable = Variable()
    state_variable.seed = seed
    logger.info("Initializing state variable with seed %s.", state_variable.seed)
    for name in sim.state.__dict__.keys():
        obj = getattr(sim.state, name)
        if isinstance(obj, np.ndarray) and name[0] != "_":
            init_shape = np.shape(obj)
            new_obj = (
                np.zeros((len(seed), *init_shape), dtype=obj.dtype) + obj[np.newaxis]
            )
            logger.info(
                "Initializing state variable %s with shape %s.", name, new_obj.shape
            )
            setattr(state_variable, name, new_obj)
        elif name[0] != "_":
            logger.warning(
                "Variable %s in sim.state is not an array, "
                "skipping initialization in state Variable object.",
                name,
            )
    parameter_variable = Variable()
    return parameter_variable, state_variable

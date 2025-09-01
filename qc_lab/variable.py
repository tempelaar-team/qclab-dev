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


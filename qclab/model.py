"""
This file contains the Model class, which is the base class for Model objects in QC Lab.
"""

from qclab.parameter import Constants
from qclab.simulation import VectorObject
import qclab.ingredients as ingredients

class Model:
    """
    Base class for models in the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the model.
    """

    def __init__(self, default_constants = None, constants=None):
        if constants is None:
            constants = {}
        if default_constants is None:
            default_constants = {}
        # Add default constants to the provided constants if not already present
        constants = {**default_constants, **constants}
        self.constants = Constants(self.update_model_constants)
        for key, val in constants.items():
            setattr(self.constants, key, val)
        self.constants._init_complete = True
        self.update_model_constants()
        self.parameters = VectorObject()

    def update_model_constants(self):
        """
        Update model parameters. This method should be overridden by subclasses.
        """
        pass

    def h_q(self):
        """
        Quantum Hamiltonian function. This method should be overridden by subclasses.

        Returns:
            None
        """
        return

    def h_qc(self):
        """
        Quantum-classical Hamiltonian function. This method should be overridden by subclasses.

        Returns:
            None
        """
        return

    def h_c(self):
        """
        Classical Hamiltonian function. This method should be overridden by subclasses.

        Returns:
            None
        """
        return

    dh_c_dzc = ingredients.dh_c_dzc_finite_differences
    dh_qc_dzc = ingredients.dh_qc_dzc_finite_differences
    init_classical = ingredients.numerical_boltzmann_init_classical
    #hop_function = ingredients.numerical_fssh_hop
    
from qclab.parameter import Parameter


class Model:
    """
    Base class for models in the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the model.
    """

    def __init__(self, parameters=None):
        """
        Initializes the ModelClass with given parameters.

        Args:
            parameters (dict): A dictionary of parameters to initialize the model.
        """
        if parameters is None:
            parameters = {}
        default_parameters = {}
        # Add default parameters to the provided parameters if not already present
        parameters = {**default_parameters, **parameters}
        self.parameters = Parameter(self.update_model_parameters)
        for key, val in parameters.items():
            setattr(self.parameters, key, val)
        self.parameters._init_complete = True
        self.update_model_parameters()

    def update_model_parameters(self):
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

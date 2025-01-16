""" 
This file contains the spin-boson model class.
"""


import numpy as np
from qclab.model import Model
import qclab.ingredients as ingredients


class SpinBosonModel(Model):
    """
    Spin-Boson model class for the simulation framework.

    Attributes:
        parameters (ParameterClass): The parameters of the model.
    """

    def __init__(self, parameters=None):
        """
        Initializes the SpinBosonModel with given parameters.

        Args:
            parameters (dict): A dictionary of parameters to initialize the model.
        """
        if parameters is None:
            parameters = {}
        self.default_parameters = {
            'temp': 1, 'V': 0.5, 'E': 0.5, 'A': 100, 'W': 0.1,
            'l_reorg': 0.02 / 4, 'boson_mass': 1
        }
        super().__init__(self.default_parameters, parameters)

    def update_model_parameters(self):
        """
        Update model parameters based on the current parameter values.
        """
        self.parameters.w = self.parameters.W * np.tan(
            ((np.arange(self.parameters.A) + 1) - 0.5) *
            np.pi / (2 * self.parameters.A)
        )  # Classical oscillator frequency
        self.parameters.g = self.parameters.w * np.sqrt(
            2 * self.parameters.l_reorg / self.parameters.A
        )  # Electron-phonon coupling
        # Diagonal energy of state 0
        self.parameters.two_level_system_a = self.parameters.E
        # Diagonal energy of state 1
        self.parameters.two_level_system_b = -self.parameters.E
        # Real part of the off-diagonal coupling
        self.parameters.two_level_system_c = self.parameters.V
        # Imaginary part of the off-diagonal coupling
        self.parameters.two_level_system_d = 0
        self.parameters.pq_weight = self.parameters.w
        self.parameters.num_classical_coordinates = self.parameters.A
        self.parameters.mass = np.ones(
            self.parameters.A) * self.parameters.boson_mass

    def h_qc(self, **kwargs):
        """
        Quantum-classical Hamiltonian function.

        Args:
            z_coord (np.ndarray): The z-coordinates.

        Returns:
            np.ndarray: The quantum-classical Hamiltonian matrix.
        """
        z_coord = kwargs['z_coord']
        h_qc = np.zeros((2, 2), dtype=complex)
        h_qc[0, 0] = np.sum(
            (self.parameters.g * np.sqrt(1 / (2 * self.parameters.mass
                                              * self.parameters.pq_weight))) *
            (z_coord + np.conj(z_coord))
        )
        h_qc[1, 1] = -h_qc[0, 0]
        return h_qc

    def h_qc_vectorized(self, **kwargs):
        """
        Vectorized quantum-classical Hamiltonian function.

        Args:
            z_coord (np.ndarray): The z-coordinates.

        Returns:
            np.ndarray: The vectorized quantum-classical Hamiltonian matrix.
        """
        z_coord = kwargs['z_coord']
        h_qc = np.zeros((*np.shape(z_coord)[:-1], 2, 2), dtype=complex)
        h_qc[..., 0, 0] = np.sum(
            (self.parameters.g * np.sqrt(1 / (2 * self.parameters.mass\
                                               * self.parameters.pq_weight)))[..., :] *
            (z_coord + np.conj(z_coord)), axis=-1
        )
        h_qc[..., 1, 1] = -h_qc[..., 0, 0]
        return h_qc

    def dh_qc_dzc(self, **kwargs):
        """
        Gradient of the quantum-classical Hamiltonian with respect to the z-coordinates.

        Args:
            z_coord (np.ndarray): The z-coordinates.

        Returns:
            np.ndarray: The gradient of the quantum-classical Hamiltonian.
        """
        del kwargs
        dh_qc_dzc = np.zeros((self.parameters.A, 2, 2), dtype=complex)
        dh_qc_dzc[:, 0, 0] = self.parameters.g * \
            np.sqrt(1 / (2 * self.parameters.mass * self.parameters.pq_weight))
        dh_qc_dzc[:, 1, 1] = -dh_qc_dzc[:, 0, 0]
        return dh_qc_dzc

    def dh_qc_dzc_vectorized(self, **kwargs):
        """
        Vectorized gradient of the quantum-classical Hamiltonian with respect to the z-coordinates.

        Args:
            z_coord (np.ndarray): The z-coordinates.

        Returns:
            np.ndarray: The vectorized gradient of the quantum-classical Hamiltonian.
        """
        dh_qc_dzc = np.zeros(
            (*np.shape(kwargs['z_coord'])[:-1], self.parameters.A, 2, 2), dtype=complex)
        dh_qc_dzc[..., :, 0, 0] = (
            self.parameters.g *
            np.sqrt(1 / (2 * self.parameters.mass * self.parameters.pq_weight))
        )[..., :]
        dh_qc_dzc[..., :, 1, 1] = -dh_qc_dzc[..., :, 0, 0]
        return dh_qc_dzc

    # Assigning functions from ingredients module
    h_q = ingredients.two_level_system_h_q
    h_c = ingredients.harmonic_oscillator_h_c
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop

    h_c_vectorized = ingredients.harmonic_oscillator_h_c_vectorized
    h_q_vectorized = ingredients.two_level_system_h_q_vectorized
    dh_c_dzc_vectorized = ingredients.harmonic_oscillator_dh_c_dzc_vectorized

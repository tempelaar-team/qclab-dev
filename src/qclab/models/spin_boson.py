"""
This module contains the spin-boson Model class.
"""

import numpy as np
from qclab.model import Model
from qclab import ingredients


class SpinBoson(Model):
    """
    Spin-Boson Model class.

    Reference publication:
    Tempelaar & Reichman. J. Chem. Phys. 2018, 148 (10), 102309. https://doi.org/10.1063/1.5000843.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT": 1.0,
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False
        self.diagonal_h_q = False

    def _init_h_q(self, parameters, **kwargs):
        self.constants.two_level_00 = self.constants.get("E")
        self.constants.two_level_11 = -self.constants.get("E")
        self.constants.two_level_01_re = self.constants.get("V")
        self.constants.two_level_01_im = 0
        return

    def _init_h_qc(self, parameters, **kwargs):
        A = self.constants.get("A")
        l_reorg = self.constants.get("l_reorg")
        boson_mass = self.constants.get("boson_mass")
        h = self.constants.classical_coordinate_weight
        w = self.constants.harmonic_frequency
        self.constants.diagonal_linear_coupling = np.zeros((2, A))
        self.constants.diagonal_linear_coupling[0] = (
            w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        self.constants.diagonal_linear_coupling[1] = (
            -w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        return

    def _init_h_c(self, parameters, **kwargs):
        A = self.constants.get("A")
        W = self.constants.get("W")
        self.constants.harmonic_frequency = W * np.tan(
            np.arange(0.5, A + 0.5, 1.0) * np.pi * 0.5 / A
        )
        return

    def _init_model(self, parameters, **kwargs):
        A = self.constants.get("A")
        boson_mass = self.constants.get("boson_mass")
        self.constants.num_classical_coordinates = A
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.classical_coordinate_mass = boson_mass * np.ones(A)
        return

    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_q", _init_h_q),
        ("_init_h_qc", _init_h_qc),
        ("_init_model", _init_model),
        ("_init_h_c", _init_h_c),
    ]



class DiagonalSpinBoson(Model):
    def __init__(self, constants={}):
        if constants is None:
            constants = {}
        self.default_constants = {
            "kBT": 1.0,
            "V": 0.5,
            "E": 0.5,
            "A": 100,
            "W": 0.1,
            "l_reorg": 0.005,
            "boson_mass": 1.0,
        }
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False
        self.diagonal_h_q = True

    def initialize_constants_model(self, parameters, **kwargs):
        """
        Initialize the model-specific constants.
        """
        A = self.constants.get("A", self.default_constants.get("A"))
        W = self.constants.get("W", self.default_constants.get("W"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        self.constants.harmonic_frequency = W * np.tan(((np.arange(A) + 1) - 0.5) * np.pi / (2 * A))
        self.constants.num_classical_coordinates = A
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.classical_coordinate_mass = boson_mass * np.ones(A)

    def initialize_constants_h_c(self, parameters, **kwargs):
        """
        Initialize the constants for the classical Hamiltonian.
        """

    def initialize_constants_h_qc(self, parameters, **kwargs):
        """
        Initialize the constants for the quantum-classical coupling Hamiltonian.
        """
        A = self.constants.get("A", self.default_constants.get("A"))
        l_reorg = self.constants.get("l_reorg", self.default_constants.get("l_reorg"))
        boson_mass = self.constants.get(
            "boson_mass", self.default_constants.get("boson_mass")
        )
        h = self.constants.classical_coordinate_weight
        w = self.constants.harmonic_frequency
        self.constants.g = w * np.sqrt(2 * l_reorg / A) * (1 / np.sqrt(2 * boson_mass * h))

    def initialize_constants_h_q(self, parameters, **kwargs):
        """
        Initialize the constants for the quantum Hamiltonian.
        """
        e = self.constants.get("E", self.default_constants.get("E"))
        v = self.constants.get("V", self.default_constants.get("V"))
        self.constants.two_level_00 = np.sqrt(e**2 + v**2)
        self.constants.two_level_11 = -np.sqrt(e**2 + v**2)
        self.constants.two_level_01_re = 0
        self.constants.two_level_01_im = 0
        
        
    def h_qc(self, parameters, **kwargs):
        z = kwargs["z"]
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(z)
        out = np.zeros((batch_size, 2, 2), dtype=complex)
        g = self.constants.g
        e = self.constants.E
        v = self.constants.V
        b = np.sqrt(e**2 + v**2)
        prod = np.sum(g[np.newaxis,:]*(2*z.real)/b,axis=1)
        out[:,0,0] = e * prod
        out[:,0,1] = -v * prod#np.sum(-v*g[np.newaxis,:]*(z + np.conj(z))/b,axis=1)
        out[:,1,0] = -v * prod#np.sum(-v*g[np.newaxis,:]*(z + np.conj(z))/b,axis=1)
        out[:,1,1] = -e * prod#np.sum(-e*g[np.newaxis,:]*(z + np.conj(z))/b,axis=1)
        return out
    
    def dh_qc_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        if kwargs.get("batch_size") is not None:
            batch_size = kwargs.get("batch_size")
        else:
            batch_size = len(z)
        num_states = self.constants.num_quantum_states
        num_classical_coordinates = self.constants.num_classical_coordinates
        dh_qc_dzc = np.zeros(
            (num_classical_coordinates, num_states, num_states), dtype=complex
        )
        g = self.constants.g
        e = self.constants.E
        v = self.constants.V
        b = np.sqrt(e**2 + v**2)
        prod = g[np.newaxis]*(np.ones(num_classical_coordinates))/b
        dh_qc_dzc[:, 0, 0] = e*prod
        dh_qc_dzc[:, 0, 1] = -v*prod
        dh_qc_dzc[:, 1, 0] = -v*prod
        dh_qc_dzc[:, 1, 1] = -e*prod
        dh_qc_dzc = dh_qc_dzc[np.newaxis, :, :, :] + np.zeros(
        (batch_size, num_classical_coordinates, num_states, num_states),
        dtype=complex)
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        self.dh_qc_dzc_inds = inds
        self.dh_qc_dzc_mels = dh_qc_dzc[inds]
        self.dh_qc_dzc_shape = shape
        return inds, mels, shape
    
    ingredients = [
        ("h_q", ingredients.h_q_two_level),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop_function", ingredients.hop_harmonic),
        ("_init_h_q", initialize_constants_h_q),
        ("_init_h_qc", initialize_constants_h_qc),
        ("_init_h_c", initialize_constants_h_c),
        ("_init_model", initialize_constants_model),
    ]

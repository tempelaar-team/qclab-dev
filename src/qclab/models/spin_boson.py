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


class AdiabaticSpinBoson(Model):
    """
    Adiabatic Spin-Boson Model class.

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
        self.update_dh_qc_dzc = True
        self.update_h_q = False

    def _init_h_qc(self, parameters, **kwargs):
        A = self.constants.get("A")
        l_reorg = self.constants.get("l_reorg")
        boson_mass = self.constants.get("boson_mass")
        h = self.constants.classical_coordinate_weight
        w = self.constants.harmonic_frequency
        self.constants.diagonal_linear_coupling = (
            w * np.sqrt(2.0 * l_reorg / A) * (1.0 / np.sqrt(2.0 * boson_mass * h))
        )
        return

    def _init_h_c(self, parameters, **kwargs):
        A = self.constants.get("A")
        W = self.constants.get("W")
        self.constants.harmonic_frequency = W * np.tan(
            np.arange(0.5, A + 0.5, 1.0) * np.pi * 0.5 / A
        )
        return
    
    def h_q(self, parameters, **kwargs):
        batch_size = kwargs["batch_size"]
        num_quantum_states = self.constants.num_quantum_states
        out = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )
        return out

    def h_qc(self, parameters, **kwargs):
        z = kwargs["z"]
        num_quantum_states = self.constants.num_quantum_states
        batch_size, num_classical_coordinates = z.shape
        out = np.zeros(
            (batch_size, num_quantum_states, num_quantum_states), dtype=complex
        )
        a = -(self.constants.E + np.sum(self.constants.diagonal_linear_coupling[np.newaxis, :] * 2 * z.real,axis=1))
        b = self.constants.V
        r = np.sqrt(a**2 + b**2)
        out[:, 0, 0] = -r
        out[:, 1, 1] = r
        return out
    
    def dh_qc_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        batch_size = len(z)
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        a = -(self.constants.E + np.sum(self.constants.diagonal_linear_coupling[np.newaxis, :] * 2 * z.real,axis=1))[:,np.newaxis]
        b = self.constants.V
        r = np.sqrt(a**2 + b**2)
        dh_qc_dzc = np.zeros(
            (batch_size, num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        da_dzc = -self.constants.diagonal_linear_coupling[np.newaxis, :]
        db_dzc = 0 * da_dzc
        dh_qc_dzc[:, :, 0, 0] = -(1/(r)) * (a * da_dzc + b * db_dzc)
        dh_qc_dzc[:, :, 1, 1] = -dh_qc_dzc[:, :, 0, 0]
        inds = np.where(dh_qc_dzc != 0)
        mels = dh_qc_dzc[inds]
        shape = np.shape(dh_qc_dzc)
        return inds, mels, shape
    
    def derivative_coupling_dzc(self, parameters, **kwargs):
        z = kwargs["z"]
        batch_size = len(z)
        num_classical_coordinates = self.constants.num_classical_coordinates
        num_quantum_states = self.constants.num_quantum_states
        a = -(self.constants.E + np.sum(self.constants.diagonal_linear_coupling[np.newaxis, :] * 2 * z.real,axis=1))[:,np.newaxis]
        b = self.constants.V
        r = np.sqrt(a**2 + b**2)
        out = np.zeros(
            (batch_size, num_classical_coordinates, num_quantum_states, num_quantum_states),
            dtype=complex,
        )
        da_dzc = -self.constants.diagonal_linear_coupling[np.newaxis, :]
        db_dzc = 0 * da_dzc
        out[:, :, 0, 1] = (1/(2*r**2)) * (b * da_dzc - a * db_dzc)
        out[:, :, 1, 0] = -out[:, :, 0, 1].conj()
        return out

    def _init_model(self, parameters, **kwargs):
        A = self.constants.get("A")
        boson_mass = self.constants.get("boson_mass")
        self.constants.num_classical_coordinates = A
        self.constants.num_quantum_states = 2
        self.constants.classical_coordinate_weight = self.constants.harmonic_frequency
        self.constants.classical_coordinate_mass = boson_mass * np.ones(A)
        return

    ingredients = [
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", dh_qc_dzc),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("derivative_coupling_dzc", derivative_coupling_dzc),
        ("_init_h_qc", _init_h_qc),
        ("_init_model", _init_model),
        ("_init_h_c", _init_h_c),
    ]

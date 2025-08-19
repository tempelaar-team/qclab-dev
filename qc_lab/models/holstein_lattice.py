"""
This module contains the Holstein lattice Model class.
"""

import numpy as np
from qc_lab.model import Model
from qc_lab import ingredients


class HolsteinLattice(Model):
    """
    A model representing a nearest-neighbor tight-binding model
    with Holstein-type electron-phonon coupling with a
    single optical mode.
    """

    def __init__(self, constants=None):
        if constants is None:
            constants = {}

        self.default_constants = {
            "kBT": 1.0,
            "g": 0.5,
            "w": 0.5,
            "N": 10,
            "J": 1.0,
            "phonon_mass": 1.0,
            "periodic": True,
        }
        super().__init__(self.default_constants, constants)

        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        N = self.constants.get("N")
        w = self.constants.get("w")
        phonon_mass = self.constants.get("phonon_mass")
        self.constants.num_quantum_states = N
        self.constants.num_classical_coordinates = N
        self.constants.classical_coordinate_weight = w * np.ones(N)
        self.constants.classical_coordinate_mass = phonon_mass * np.ones(N)
        return

    def _init_h_q(self, parameters, **kwargs):
        J = self.constants.get("J")
        periodic = self.constants.get("periodic")
        self.constants.nearest_neighbor_hopping_energy = J
        self.constants.nearest_neighbor_periodic = periodic
        return

    def _init_h_qc(self, parameters, **kwargs):
        N = self.constants.get("N")
        w = self.constants.get("w")
        g = self.constants.get("g")
        h = self.constants.classical_coordinate_weight
        self.constants.diagonal_linear_coupling = np.diag(
            g * w * np.sqrt(h / w) * np.ones(N)
        )
        return

    def _init_h_c(self, parameters, **kwargs):
        N = self.constants.get("N")
        w = self.constants.get("w")
        self.constants.harmonic_frequency = w * np.ones(N)
        return

    ingredients = [
        ("h_q", ingredients.h_q_nearest_neighbor),
        ("h_qc", ingredients.h_qc_diagonal_linear),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_qc_dzc", ingredients.dh_qc_dzc_diagonal_linear),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("init_classical", ingredients.init_classical_boltzmann_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_h_q", _init_h_q),
        ("_init_h_qc", _init_h_qc),
        ("_init_h_c", _init_h_c),
        ("_init_model", _init_model),
    ]

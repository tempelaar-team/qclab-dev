import numpy as np
from qclab.model import Model
import qclab.ingredients as ingredients


class HolsteinLatticeModel(Model):
    """
    A model representing a nearest-neighbor tight-binding model 
    with Holstein-type electron-phonon coupling with a 
    single optical mode.
    """
    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}
        self.default_parameters = {
            'temp': 1, 'g': 0.5, 'w': 0.5, 'N': 10, 't': 1, \
                'phonon_mass': 1, 'periodic_boundary': True
        }
        super().__init__(self.default_parameters, parameters)

    def update_model_parameters(self):
        self.parameters.w = self.parameters.w * np.ones(self.parameters.N)
        self.parameters.g = self.parameters.g * np.ones(self.parameters.N)
        self.parameters.mass = self.parameters.phonon_mass * np.ones(self.parameters.N)
        self.parameters.num_classical_coordinates = self.parameters.N
        self.parameters.pq_weight = self.parameters.w
        self.parameters.nearest_neighbor_lattice_h_q_num_sites = self.parameters.N
        self.parameters.nearest_neighbor_lattice_h_q_hopping_energy = self.parameters.j
        self.parameters.nearest_neighbor_lattice_h_q_periodic_boundary = \
            self.parameters.periodic_boundary
        self.parameters.holstein_lattice_h_qc_num_sites = self.parameters.N
        self.parameters.holstein_lattice_h_qc_oscillator_frequency = self.parameters.w
        self.parameters.holstein_lattice_h_qc_dimensionless_coupling = self.parameters.g


    h_q_vectorized = ingredients.nearest_neighbor_lattice_h_q_vectorized
    h_q = ingredients.nearest_neighbor_lattice_h_q
    h_c = ingredients.harmonic_oscillator_h_c
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    dh_c_dzc_vectorized = ingredients.harmonic_oscillator_dh_c_dzc_vectorized
    h_c_vectorized = ingredients.harmonic_oscillator_h_c_vectorized
    h_qc = ingredients.holstein_lattice_h_qc
    h_qc_vectorized = ingredients.holstein_lattice_h_qc_vectorized
    dh_qc_dzc = ingredients.holstein_lattice_dh_qc_dzc
    dh_qc_dzc_vectorized = ingredients.holstein_lattice_dh_qc_dzc_vectorized
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical

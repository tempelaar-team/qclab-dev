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

    def update_model_constants(self):
        self.constants.w = self.constants.w * np.ones(self.constants.N)
        self.constants.g = self.constants.g * np.ones(self.constants.N)
        self.constants.mass = self.constants.phonon_mass * np.ones(self.constants.N)
        self.constants.num_classical_coordinates = self.constants.N
        self.constants.pq_weight = self.constants.w
        self.constants.nearest_neighbor_lattice_h_q_num_sites = self.constants.N
        self.constants.nearest_neighbor_lattice_h_q_hopping_energy = self.constants.j
        self.constants.nearest_neighbor_lattice_h_q_periodic_boundary = \
            self.constants.periodic_boundary
        self.constants.holstein_lattice_h_qc_num_sites = self.constants.N
        self.constants.holstein_lattice_h_qc_oscillator_frequency = self.constants.w
        self.constants.holstein_lattice_h_qc_dimensionless_coupling = self.constants.g


    h_q = ingredients.nearest_neighbor_lattice_h_q_vectorized
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc_vectorized
    h_c = ingredients.harmonic_oscillator_h_c_vectorized
    h_qc = ingredients.holstein_lattice_h_qc_vectorized
    dh_qc_dzc = ingredients.holstein_lattice_dh_qc_dzc_vectorized
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical

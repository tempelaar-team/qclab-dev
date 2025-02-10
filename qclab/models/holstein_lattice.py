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
        self.constants.nearest_neighbor_lattice_num_sites = self.constants.N
        self.constants.nearest_neighbor_lattice_hopping_energy = self.constants.j
        self.constants.nearest_neighbor_lattice_periodic_boundary = \
            self.constants.periodic_boundary
        self.constants.holstein_coupling_num_sites = self.constants.N
        self.constants.holstein_coupling_oscillator_frequency = self.constants.w
        self.constants.holstein_coupling_dimensionless_coupling = self.constants.g
        self.constants.harmonic_oscillator_frequency = self.constants.w
        self.constants.harmonic_oscillator_mass = self.constants.mass



    h_q = ingredients.nearest_neighbor_lattice_h_q
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    h_c = ingredients.harmonic_oscillator_h_c
    h_qc = ingredients.holstein_coupling_h_qc
    dh_qc_dzc = ingredients.holstein_coupling_dh_qc_dzc
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical

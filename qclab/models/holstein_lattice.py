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

    def initialize_constants_model(self):
        self.constants.num_quantum_states = self.constants.N
        self.constants.num_classical_coordinates = self.constants.N
        self.constants.classical_coordinate_weight = self.constants.w * np.ones(self.constants.N)
        self.constants.classical_coordinate_mass = self.constants.phonon_mass * np.ones(self.constants.N)

    def initialize_constants_h_q(self):
        self.constants.nearest_neighbor_lattice_hopping_energy = self.constants.j 
        self.constants.nearest_neighbor_lattice_periodic_boundary = self.constants.periodic_boundary

    def initialize_constants_h_qc(self):
        self.constants.holstein_coupling_oscillator_frequency = self.constants.w * np.ones(self.constants.N)
        self.constants.holstein_coupling_dimensionless_coupling = self.constants.g * np.ones(self.constants.N)

    def initialize_constants_h_c(self):
        self.constants.harmonic_oscillator_frequency = self.constants.w * np.ones(self.constants.N)

    h_q = ingredients.nearest_neighbor_lattice_h_q
    dh_c_dzc = ingredients.harmonic_oscillator_dh_c_dzc
    h_c = ingredients.harmonic_oscillator_h_c
    h_qc = ingredients.holstein_coupling_h_qc
    dh_qc_dzc = ingredients.holstein_coupling_dh_qc_dzc
    init_classical = ingredients.harmonic_oscillator_boltzmann_init_classical
    hop_function = ingredients.harmonic_oscillator_hop
    initialization_functions = [
        initialize_constants_model,
        initialize_constants_h_c,
        initialize_constants_h_qc,
        initialize_constants_h_q,
    ]
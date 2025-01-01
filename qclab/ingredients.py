import numpy as np
import warnings


def harmonic_oscillator_h_c(model, **kwargs):
    """
    Calculate the classical Hamiltonian.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.pq_weight (np.ndarray): The weight parameters.\n
    """
    z_coord = kwargs['z_coord']
    h_c = np.sum(model.parameters.pq_weight * np.conjugate(z_coord) * z_coord)
    return h_c


def harmonic_oscillator_h_c_vectorized(model, **kwargs):
    """
    Calculate the vectorized classical Hamiltonian.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.pq_weight (np.ndarray): The weight parameters.\n
    """
    z_coord = kwargs['z_coord']
    h_c = np.sum(model.parameters.pq_weight[..., :] * np.conjugate(z_coord) * z_coord, axis=-1)
    return h_c


def harmonic_oscillator_dh_c_dzc(model, **kwargs):
    """
    Calculate the derivative of the classical Hamiltonian with respect to the z-coordinates.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.pq_weight (np.ndarray): The weight parameters.\n
    """
    z_coord = kwargs['z_coord']
    dh_c_dzc = model.parameters.pq_weight * z_coord + 0.0j
    return dh_c_dzc


def harmonic_oscillator_dh_c_dzc_vectorized(model, **kwargs):
    """
    Calculate the vectorized derivative of the classical Hamiltonian with respect to the z-coordinates.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.pq_weight (np.ndarray): The weight parameters.\n
    """
    z_coord = kwargs['z_coord']
    dh_c_dzc = model.parameters.pq_weight[..., :] * z_coord + 0.0j
    return dh_c_dzc


def two_level_system_h_q(model, **kwargs):
    """
    Calculate the quantum Hamiltonian for a two-level system.

    Model parameters:
        model.parameters.two_level_system_a (float): Parameter a.\n
        model.parameters.two_level_system_b (float): Parameter b.\n
        model.parameters.two_level_system_c (float): Parameter c.\n
        model.parameters.two_level_system_d (float): Parameter d.\n
    """
    h_q = np.zeros((2, 2), dtype=complex)
    h_q[0, 0] = model.parameters.two_level_system_a
    h_q[1, 1] = model.parameters.two_level_system_b
    h_q[0, 1] = model.parameters.two_level_system_c + 1j * model.parameters.two_level_system_d
    h_q[1, 0] = model.parameters.two_level_system_c - 1j * model.parameters.two_level_system_d
    return h_q


def two_level_system_h_q_vectorized(model, **kwargs):
    """
    Calculate the vectorized quantum Hamiltonian for a two-level system.

    Model parameters:
        model.parameters.two_level_system_a (float): Parameter a.\n
        model.parameters.two_level_system_b (float): Parameter b.\n
        model.parameters.two_level_system_c (float): Parameter c.\n
        model.parameters.two_level_system_d (float): Parameter d.\n
    """
    h_q = np.zeros((2, 2), dtype=complex)
    h_q[0, 0] = model.parameters.two_level_system_a
    h_q[1, 1] = model.parameters.two_level_system_b
    h_q[0, 1] = model.parameters.two_level_system_c + 1j * model.parameters.two_level_system_d
    h_q[1, 0] = model.parameters.two_level_system_c - 1j * model.parameters.two_level_system_d
    return h_q[np.newaxis, :, :]


def nearest_neighbor_lattice_h_q(model, **kwargs):
    """
    Calculate the quantum Hamiltonian for a nearest-neighbor lattice.

    Model parameters:
        model.parameters.nearest_neighbor_lattice_h_q_num_sites (int): Number of sites.\n
        model.parameters.nearest_neighbor_lattice_h_q_hopping_energy (complex): Hopping energy.\n
        model.parameters.nearest_neighbor_lattice_h_q_periodic_boundary (bool): Periodic boundary condition.\n
    """
    num_sites = model.parameters.nearest_neighbor_lattice_h_q_num_sites
    hopping_energy = model.parameters.nearest_neighbor_lattice_h_q_hopping_energy
    periodic_boundary = model.parameters.nearest_neighbor_lattice_h_q_periodic_boundary
    h_q = np.zeros((num_sites, num_sites), dtype=complex)

    # Fill the Hamiltonian matrix with hopping energies
    for n in range(num_sites - 1):
        h_q[n, n + 1] = -hopping_energy
        h_q[n + 1, n] = -np.conj(hopping_energy)

    # Apply periodic boundary conditions if specified
    if periodic_boundary:
        h_q[0, num_sites - 1] = -hopping_energy
        h_q[num_sites - 1, 0] = -np.conj(hopping_energy)

    return h_q


def nearest_neighbor_lattice_h_q_vectorized(model, **kwargs):
    """
    Calculate the vectorized quantum Hamiltonian for a nearest-neighbor lattice.

    Model parameters:
        model.parameters.nearest_neighbor_lattice_h_q_num_sites (int): Number of sites.\n
        model.parameters.nearest_neighbor_lattice_h_q_hopping_energy (complex): Hopping energy.\n
        model.parameters.nearest_neighbor_lattice_h_q_periodic_boundary (bool): Periodic boundary condition.\n
    """
    num_sites = model.parameters.nearest_neighbor_lattice_h_q_num_sites
    hopping_energy = model.parameters.nearest_neighbor_lattice_h_q_hopping_energy
    periodic_boundary = model.parameters.nearest_neighbor_lattice_h_q_periodic_boundary
    h_q = np.zeros((num_sites, num_sites), dtype=complex)

    # Fill the Hamiltonian matrix with hopping energies
    for n in range(num_sites - 1):
        h_q[n, n + 1] += -hopping_energy
        h_q[n + 1, n] += np.conj(h_q[n, n + 1])

    # Apply periodic boundary conditions if specified
    if periodic_boundary:
        h_q[0, num_sites - 1] += -hopping_energy
        h_q[num_sites - 1, 0] += np.conj(h_q[0, num_sites - 1])

    return h_q[..., :, :]


def holstein_lattice_h_qc(model, **kwargs):
    """
    Calculate the quantum-classical Hamiltonian for a Holstein lattice.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.\n
        model.parameters.holstein_lattice_h_qc_dimensionless_coupling (float): Dimensionless coupling.\n
    """
    z_coord = kwargs['z_coord']
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    h_qc = np.diag(dimensionless_coupling * oscillator_frequency) * (z_coord + np.conj(z_coord)) + 0.0j
    return h_qc


def holstein_lattice_h_qc_vectorized(model, **kwargs):
    """
    Calculate the vectorized quantum-classical Hamiltonian for a Holstein lattice.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.holstein_lattice_h_qc_num_sites (int): Number of sites.\n
        model.parameters.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.\n
        model.parameters.holstein_lattice_h_qc_dimensionless_coupling (float): Dimensionless coupling.\n
    """
    z_coord = kwargs['z_coord']
    num_sites = model.parameters.holstein_lattice_h_qc_num_sites
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    h_qc = np.zeros((*np.shape(z_coord)[:-1], num_sites, num_sites), dtype=complex)
    np.einsum('...ii->...i', h_qc)[...] = (dimensionless_coupling * oscillator_frequency)[..., :] * (
                z_coord + np.conj(z_coord)) + 0.0j
    return h_qc


def holstein_lattice_dh_qc_dzc(model, **kwargs):
    """
    Calculate the derivative of the quantum-classical Hamiltonian with respect to the z-coordinates.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.holstein_lattice_h_qc_num_sites (int): Number of sites.\n
        model.parameters.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.\n
        model.parameters.holstein_lattice_h_qc_dimensionless_coupling (float): Dimensionless coupling.\n
    """
    z_coord = kwargs['z_coord']
    num_sites = model.parameters.holstein_lattice_h_qc_num_sites
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    dh_qc_dzc = np.zeros((num_sites, num_sites, num_sites), dtype=complex)
    np.einsum('iii->i', dh_qc_dzc)[...] = dimensionless_coupling * oscillator_frequency * (np.ones_like(z_coord)) + 0.0j
    return dh_qc_dzc


def holstein_lattice_dh_qc_dzc_vectorized(model, **kwargs):
    """
    Calculate the vectorized derivative of the quantum-classical Hamiltonian with respect to the z-coordinates.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        model.parameters.holstein_lattice_h_qc_num_sites (int): Number of sites.\n
        model.parameters.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.\n
        model.parameters.holstein_lattice_h_qc_dimensionless_coupling (float): Dimensionless coupling.\n
    """
    z_coord = kwargs['z_coord']
    num_sites = model.parameters.holstein_lattice_h_qc_num_sites
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    dh_qc_dzc = np.zeros((*np.shape(z_coord)[:-1], num_sites, num_sites, num_sites), dtype=complex)
    np.einsum('...iii->...i', dh_qc_dzc)[...] = (dimensionless_coupling * oscillator_frequency)[..., :] * (
        np.ones_like(z_coord)) + 0.0j
    return dh_qc_dzc


def harmonic_oscillator_hop(model, **kwargs):
    """
    Perform a hopping operation for the harmonic oscillator.

    Required keyword arguments:
        z_coord (np.ndarray): The z-coordinates.
        delta_z_coord (np.ndarray): The change in z-coordinates.
        ev_diff (float): The energy difference.

    Model parameters:
        model.parameters.pq_weight (np.ndarray): The weight parameters.\n
    """
    z_coord = kwargs['z_coord']
    delta_z_coord = kwargs['delta_z_coord']
    ev_diff = kwargs['ev_diff']
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc = np.conj(z_coord)
    warnings.warn("Hop function excludes mass, check it", UserWarning)
    akj_z = np.real(np.sum(model.parameters.pq_weight * delta_zc_coord * delta_z_coord))
    bkj_z = np.real(np.sum(1j * model.parameters.pq_weight * (zc * delta_z_coord - z_coord * delta_zc_coord)))
    ckj_z = ev_diff
    disc = bkj_z ** 2 - 4 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = gamma / (2 * akj_z)
        # adjust classical coordinate
        z_coord = z_coord - 1.0j * np.real(gamma) * delta_z_coord
        hopped = True
    return z_coord, hopped


def harmonic_oscillator_boltzmann_init_classical(model, **kwargs):
    """
    Initialize classical coordinates according to Boltzmann statistics.

    Required keyword arguments:
        seed (int): The random seed.

    Model parameters:
        model.parameters.temp (float): Temperature.\n
        model.parameters.mass (float): Mass.\n
        model.parameters.pq_weight (np.ndarray): The weight parameters.\n
        model.parameters.num_classical_coordinates (int): Number of classical coordinates.\n
    """
    seed = kwargs.get('seed', None)
    np.random.seed(seed)
    q = np.random.normal(
        loc=0,
        scale=np.sqrt(model.parameters.temp / (model.parameters.mass * (model.parameters.pq_weight ** 2))),
        size=model.parameters.num_classical_coordinates
    )
    p = np.random.normal(
        loc=0,
        scale=np.sqrt(model.parameters.temp),
        size=model.parameters.num_classical_coordinates
    )
    z = np.sqrt(model.parameters.pq_weight * model.parameters.mass / 2) * (
            q + 1.0j * (p / (model.parameters.pq_weight * model.parameters.mass))
    )
    return z


def harmonic_oscillator_wigner_init_classical(model, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution of the ground state of a harmonic oscillator.

    Required model.parameters attributes:
        - pq_weight (float): The pq weight parameter.\n
        - mass (float): The mass of the harmonic oscillator.\n
        - temp (float): The temperature.\n
        - num_classical_coordinates (int): The number of classical coordinates.\n
    """
    seed = kwargs.get('seed', None)
    np.random.seed(seed)
    
    # Calculate the standard deviations for q and p
    std_q = np.sqrt(1 / (2 * model.parameters.pq_weight * model.parameters.mass * np.tanh(model.parameters.pq_weight / (2 * model.parameters.temp))))
    std_p = np.sqrt((model.parameters.mass * model.parameters.pq_weight) / (2 * np.tanh(model.parameters.pq_weight / (2 * model.parameters.temp))))
    
    # Generate random q and p values
    q = np.random.normal(loc=0, scale=std_q, size=model.parameters.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=std_p, size=model.parameters.num_classical_coordinates)
    
    # Calculate the classical coordinates z
    z = np.sqrt(model.parameters.pq_weight * model.parameters.mass / 2) * (q + 1.0j * (p / (model.parameters.pq_weight * model.parameters.mass)))
    
    return z
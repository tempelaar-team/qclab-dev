import numpy as np
import warnings


def harmonic_oscillator_h_c(model, **kwargs):
    """
    Calculate the classical Hamiltonian.

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments, including 'z_coord'.

    Returns:
        float: The classical Hamiltonian.
    """
    z_coord = kwargs['z_coord']
    h_c = np.sum(model.parameters.pq_weight * np.conjugate(z_coord) * z_coord)
    return h_c


def harmonic_oscillator_h_c_vectorized(model, **kwargs):
    """
    Calculate the vectorized classical Hamiltonian.

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments, including 'z_coord'.

    Returns:
        np.ndarray: The vectorized classical Hamiltonian.
    """
    z_coord = kwargs['z_coord']
    h_c = np.sum(model.parameters.pq_weight[..., :] * np.conjugate(z_coord) * z_coord, axis=-1)
    return h_c


def harmonic_oscillator_dh_c_dzc(model, **kwargs):
    """
    Calculate the derivative of the classical Hamiltonian with respect to the z-coordinates.

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments, including 'z_coord'.

    Returns:
        np.ndarray: The derivative of the classical Hamiltonian.
    """
    z_coord = kwargs['z_coord']
    dh_c_dzc = model.parameters.pq_weight * z_coord + 0.0j
    return dh_c_dzc


def harmonic_oscillator_dh_c_dzc_vectorized(model, **kwargs):
    """
    Calculate the vectorized derivative of the classical Hamiltonian with respect to the z-coordinates.

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments, including 'z_coord'.

    Returns:
        np.ndarray: The vectorized derivative of the classical Hamiltonian.
    """
    z_coord = kwargs['z_coord']
    dh_c_dzc = model.parameters.pq_weight[..., :] * z_coord + 0.0j
    return dh_c_dzc


def two_level_system_h_q(model, **kwargs):
    """
    Calculate the quantum Hamiltonian for a two-level system.

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The quantum Hamiltonian matrix.
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

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The vectorized quantum Hamiltonian matrix.
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

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The quantum Hamiltonian matrix.
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

    Args:
        model: The model object containing parameters.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: The vectorized quantum Hamiltonian matrix.
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
    z_coord = kwargs['z_coord']
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    h_qc = np.diag(dimensionless_coupling * oscillator_frequency) * (z_coord + np.conj(z_coord)) + 0.0j
    return h_qc


def holstein_lattice_h_qc_vectorized(model, **kwargs):
    z_coord = kwargs['z_coord']
    num_sites = model.parameters.holstein_lattice_h_qc_num_sites
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    h_qc = np.zeros((*np.shape(z_coord)[:-1], num_sites, num_sites), dtype=complex)
    np.einsum('...ii->...i', h_qc)[...] = (dimensionless_coupling * oscillator_frequency)[..., :] * (
                z_coord + np.conj(z_coord)) + 0.0j
    return h_qc


def holstein_lattice_dh_qc_dzc(model, **kwargs):
    z_coord = kwargs['z_coord']
    num_sites = model.parameters.holstein_lattice_h_qc_num_sites
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    dh_qc_dzc = np.zeros((num_sites, num_sites, num_sites), dtype=complex)
    np.einsum('iii->i', dh_qc_dzc)[...] = dimensionless_coupling * oscillator_frequency * (np.ones_like(z_coord)) + 0.0j
    return dh_qc_dzc


def holstein_lattice_dh_qc_dzc_vectorized(model, **kwargs):
    z_coord = kwargs['z_coord']
    num_sites = model.parameters.holstein_lattice_h_qc_num_sites
    oscillator_frequency = model.parameters.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = model.parameters.holstein_lattice_h_qc_dimensionless_coupling
    dh_qc_dzc = np.zeros((*np.shape(z_coord)[:-1], num_sites, num_sites, num_sites), dtype=complex)
    np.einsum('...iii->...i', dh_qc_dzc)[...] = (dimensionless_coupling * oscillator_frequency)[..., :] * (
        np.ones_like(z_coord)) + 0.0j
    return dh_qc_dzc


def harmonic_oscillator_hop(model, **kwargs):
    z_coord = kwargs['z_coord']
    delta_z_coord = kwargs['delta_z_coord']
    ev_diff = kwargs['ev_diff']
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc = np.conj(z_coord)
    warnings.warn("Hop function excludes mass, cehck it", UserWarning)
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

    Args:
        model: The model object containing parameters such as temperature, mass, and frequency.
        **kwargs: Additional keyword arguments, including 'seed'.

    Returns:
        np.ndarray: The initialized classical coordinates.
    """
    seed = kwargs['seed']
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

def harmonic_oscillator_wigner_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to Wigner distribution of the ground state of a harmonic oscillator
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed(seed)
    q = np.random.normal(loc=0, 
    scale=np.sqrt(1 / (2 * model.parameters.pq_weight * model.parameters.mass*np.tanh(model.parameters.pq_weight/(2*model.parameters.temp)))),
                         size=model.parameters.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=np.sqrt((model.parameters.mass * model.parameters.pq_weight) / (2 * np.tanh(model.parameters.pq_weight/(2*model.parameters.temp)))), size=model.parameters.num_classical_coordinates)
    z = np.sqrt(model.parameters.pq_weight * model.parameters.mass / 2) * (q + 1.0j * (p / (model.parameters.pq_weight * model.parameters.mass)))
    return z
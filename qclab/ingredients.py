"""
This file contains ingredient functions for use in Model classes.
"""
import warnings
import numpy as np


def harmonic_oscillator_h_c_vectorized(model, constants, parameters, **kwargs):
    """
    Harmonic oscillator classical Hamiltonian function.

    Model Ingredient:
        - model.h_c_vectorized

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.pq_weight (np.ndarray): The weight parameters.

    Related functions:
        - :func:`harmonic_oscillator_h_c`
    """
    z_coord = kwargs["z_coord"]
    h_c = np.sum(
        constants.pq_weight[..., :] * np.conjugate(z_coord) * z_coord, axis=-1
    )
    return h_c


def harmonic_oscillator_dh_c_dzc_vectorized(model, constants, parameters, **kwargs):
    """
    Calculate the vectorized derivative of the classical Hamiltonian
    with respect to the z-coordinates.

    Model Ingredient:
        - model.dh_c_dzc_vectorized

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.pq_weight (np.ndarray): The weight parameters.

    Related functions:
        - :func:`harmonic_oscillator_dh_c_dzc`
    """
    z_coord = kwargs["z_coord"]
    dh_c_dzc = constants.pq_weight[..., :] * z_coord + 0.0j
    return dh_c_dzc


def two_level_system_h_q_vectorized(model, constants, parameters, **kwargs):
    """
    Calculate the vectorized quantum Hamiltonian for a two-level system.

    Model Ingredient:
        - model.h_q_vectorized

    Model parameters:
        - constants.two_level_system_a (float): <0|H|0>
        - constants.two_level_system_b (float): <1|H|1>
        - constants.two_level_system_c (float): Re(<0|H|1>)
        - constants.two_level_system_d (float): Im(<0|H|1>)

    Related functions:
        - :func:`two_level_system_h_q`
    """
    del kwargs
    batch_size = parameters._size
    h_q = np.zeros((batch_size, 2, 2), dtype=complex)
    h_q[:, 0, 0] = constants.two_level_system_a
    h_q[:, 1, 1] = constants.two_level_system_b
    h_q[:, 0, 1] = (constants.two_level_system_c + 1j * constants.two_level_system_d)
    h_q[:, 1, 0] = (constants.two_level_system_c - 1j * constants.two_level_system_d)
    return h_q


def nearest_neighbor_lattice_h_q_vectorized(model, constants, parameters, **kwargs):
    """
    Calculate the vectorized quantum Hamiltonian for a nearest-neighbor lattice.

    Model Ingredient:
        - model.h_q_vectorized

    Model parameters:
        - constants.nearest_neighbor_lattice_h_q_num_sites (int): Number of sites.
        - constants.nearest_neighbor_lattice_h_q_hopping_energy (complex): Hopping energy.
        - constants.nearest_neighbor_lattice_h_q_periodic_boundary (bool):
          Periodic boundary condition.

    Related functions:
        - :func:`nearest_neighbor_lattice_h_q`
    """
    del kwargs
    num_sites = constants.nearest_neighbor_lattice_h_q_num_sites
    hopping_energy = constants.nearest_neighbor_lattice_h_q_hopping_energy
    periodic_boundary = constants.nearest_neighbor_lattice_h_q_periodic_boundary
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


def holstein_lattice_h_qc_vectorized(model, constants, parameters, **kwargs):
    """
    Calculate the vectorized quantum-classical Hamiltonian for a Holstein lattice.

    Model Ingredient:
        - model.h_qc_vectorized

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.holstein_lattice_h_qc_num_sites (int): Number of sites.
        - constants.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.
        - constants.holstein_lattice_h_qc_dimensionless_coupling
          (float): Dimensionless coupling.

    Related functions:
        - :func:`holstein_lattice_h_qc`
    """
    z_coord = kwargs["z_coord"]
    num_sites = constants.holstein_lattice_h_qc_num_sites
    oscillator_frequency = constants.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = (
        constants.holstein_lattice_h_qc_dimensionless_coupling
    )
    h_qc = np.zeros(
        (*np.shape(z_coord)[:-1], num_sites, num_sites), dtype=complex)
    np.einsum("...ii->...i", h_qc)[...] = (
        dimensionless_coupling * oscillator_frequency
    )[..., :] * (z_coord + np.conj(z_coord)) + 0.0j
    return h_qc


def holstein_lattice_dh_qc_dzc_vectorized(model, constants, parameters, **kwargs):
    """
    Calculate the vectorized derivative of the quantum-classical Hamiltonian with
    respect to the z-coordinates.

    Model Ingredient:
        - model.dh_qc_dzc_vectorized

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.holstein_lattice_h_qc_num_sites (int): Number of sites.
        - constants.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.
        - constants.holstein_lattice_h_qc_dimensionless_coupling (float):
          Dimensionless coupling.

    Related functions:
        - :func:`holstein_lattice_dh_qc_dzc`
    """
    z_coord = kwargs["z_coord"]
    num_sites = constants.holstein_lattice_h_qc_num_sites
    oscillator_frequency = constants.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = (
        constants.holstein_lattice_h_qc_dimensionless_coupling
    )
    dh_qc_dzc = np.zeros(
        (*np.shape(z_coord)[:-1], num_sites,
         num_sites, num_sites), dtype=complex
    )
    np.einsum("...iii->...i", dh_qc_dzc)[...] = (
        dimensionless_coupling * oscillator_frequency
    )[..., :] * (np.ones_like(z_coord)) + 0.0j
    return dh_qc_dzc


def harmonic_oscillator_hop(model, constants, parameters, **kwargs):
    """
    Perform a hopping operation for the harmonic oscillator.

    Model Ingredient:
        - model.hop

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.
        - delta_z_coord (np.ndarray): The change in z-coordinates.
        - ev_diff (float): The energy difference.

    Model parameters:
        - constants.pq_weight (np.ndarray): The weight parameters.

    Related functions:
        - :func:`harmonic_oscillator_boltzmann_init_classical`
        - :func:`harmonic_oscillator_wigner_init_classical`
    """
    z_coord = kwargs["z_coord"]
    delta_z_coord = kwargs["delta_z_coord"]
    ev_diff = kwargs["ev_diff"]
    hopped = False
    delta_zc_coord = np.conj(delta_z_coord)
    zc = np.conj(z_coord)
    warnings.warn("Hop function excludes mass, check it", UserWarning)
    akj_z = np.real(np.sum(constants.pq_weight *
                    delta_zc_coord * delta_z_coord))
    bkj_z = np.real(
        np.sum(
            1j
            * constants.pq_weight
            * (zc * delta_z_coord - z_coord * delta_zc_coord)
        )
    )
    ckj_z = ev_diff
    disc = bkj_z**2 - 4 * akj_z * ckj_z
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


def harmonic_oscillator_boltzmann_init_classical(model, constants, parameters, **kwargs):
    """
    Initialize classical coordinates according to Boltzmann statistics.

    Model Ingredient:
        - model.init_classical

    Required keyword arguments:
        - seed (int): The random seed.

    Model parameters:
        - constants.temp (float): Temperature.
        - constants.mass (float): Mass.
        - constants.pq_weight (np.ndarray): The weight parameters.
        - constants.num_classical_coordinates (int): Number of classical coordinates.

    Related functions:
        - :func:`harmonic_oscillator_wigner_init_classical`
    """
    seed = kwargs.get("seed", None)
    kBT = constants.temp
    m = constants.mass
    h = constants.pq_weight
    np.random.seed(seed)
    q = np.random.normal(
        loc=0,
        scale=np.sqrt(kBT / (m * (h**2))),
        size=constants.num_classical_coordinates)
    p = np.random.normal(
        loc=0,
        scale=np.sqrt(kBT),
        size=constants.num_classical_coordinates)
    z = np.sqrt(h * m / 2) * (q + 1.0j * (p / (h * m)))
    return z

def numerical_boltzmann_init_classical(model, constants, parameters, **kwargs):
    """
    This function samples a discrete probability distribution 
    approximating the Boltzmann distribution of the classical 
    Hamiltonian function.
    """
    seed = kwargs.get("seed", None)
    np.random.seed(seed)
    rand_val = np.random.rand()
    num_points = 1000
    amplitudes = 4*(np.random.rand(num_points, constants.num_classical_coordinates)-0.5)
    phases = np.exp(1.0j*2*np.pi*np.random.rand(num_points, constants.num_classical_coordinates))
    z_list = amplitudes * phases
    classical_energies = np.zeros(num_points)
    for n in range(num_points):
        classical_energies[n] = np.real(model.h_c(z_coord = z_list[n]))
    boltz_facs = np.exp(-classical_energies/constants.temp)
    boltz_facs = boltz_facs/np.sum(boltz_facs)
    cumulant = 0
    for n in range(num_points):
        cumulant += boltz_facs[n]
        if cumulant >= rand_val:
            z = z_list[n]
            break
    return z

def harmonic_oscillator_wigner_init_classical(model, constants, parameters, **kwargs):
    """
    Initialize classical coordinates according to the Wigner distribution
    of the ground state of a harmonic oscillator.

    Model Ingredient:
        - model.init_classical

    Required constants attributes:
        - pq_weight (float): The pq weight parameter.
        - mass (float): The mass of the harmonic oscillator.
        - temp (float): The temperature.
        - num_classical_coordinates (int): The number of classical coordinates.

    Related functions:
        - :func:`harmonic_oscillator_boltzmann_init_classical`
    """
    seed = kwargs.get("seed", None)
    np.random.seed(seed)

    # Calculate the standard deviations for q and p
    std_q = np.sqrt(
        1
        / (
            2
            * constants.pq_weight
            * constants.mass
            * np.tanh(constants.pq_weight / (2 * constants.temp))
        )
    )
    std_p = np.sqrt(
        (constants.mass * constants.pq_weight)
        / (2 * np.tanh(constants.pq_weight / (2 * constants.temp)))
    )

    # Generate random q and p values
    q = np.random.normal(
        loc=0, scale=std_q, size=constants.num_classical_coordinates
    )
    p = np.random.normal(
        loc=0, scale=std_p, size=constants.num_classical_coordinates
    )

    # Calculate the classical coordinates z
    z = np.sqrt(constants.pq_weight * constants.mass / 2) * (
        q + 1.0j * (p / (constants.pq_weight * constants.mass))
    )

    return z


def harmonic_oscillator_h_c(model, constants, parameters, **kwargs):
    """
    Calculate the classical Hamiltonian.

    Model Ingredient:
        - model.h_c

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.pq_weight (np.ndarray): The weight parameters.

    Related functions:
        - :func:`harmonic_oscillator_h_c_vectorized`
    """
    z_coord = kwargs["z_coord"]
    h = constants.pq_weight
    z = kwargs.get('z_coord', parameters.z_coord)
    h_c = np.sum(h * np.conjugate(z) * z)
    return h_c


def harmonic_oscillator_dh_c_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the classical Hamiltonian with respect to the z-coordinates.

    Model Ingredient:
        - model.dh_c_dzc

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.pq_weight (np.ndarray): The weight parameters.

    Related functions:
        - :func:`harmonic_oscillator_dh_c_dzc_vectorized`
    """
    z_coord = kwargs["z_coord"]
    dh_c_dzc = constants.pq_weight * z_coord + 0.0j
    return dh_c_dzc


def dh_c_dzc_finite_differences(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    # Approximate the gradient using finite differences
    delta_z = 1e-6
    offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
    offset_z_coord_im = (
        z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z
    )

    h_c_0 = model.h_c(constants, parameters, z_coord=z_coord)
    dh_c_dzc = np.zeros((len(z_coord), *np.shape(h_c_0)), dtype=complex)

    for n in range(len(z_coord)):
        h_c_offset_re = model.h_c(constants, parameters, z_coord=offset_z_coord_re[n])
        diff_re = (h_c_offset_re - h_c_0) / delta_z
        h_c_offset_im = model.h_c(constants, parameters, z_coord=offset_z_coord_im[n])
        diff_im = (h_c_offset_im - h_c_0) / delta_z
        dh_c_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
    return dh_c_dzc

def finite_differences_dh_c_dzc_vectorized(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    delta_z = np.ones((len(z_coord),))

def dh_qc_dzc_finite_differences(model, constants, parameters, **kwargs):
    z_coord = kwargs["z_coord"]
    # Approximate the gradient using finite differences
    delta_z = 1e-6
    offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
    offset_z_coord_im = (
        z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z
    )

    h_qc_0 = model.h_qc(constants, parameters, z_coord=z_coord)
    dh_qc_dzc = np.zeros((len(z_coord), *np.shape(h_qc_0)), dtype=complex)

    for n in range(len(z_coord)):
        h_qc_offset_re = model.h_qc(constants, parameters, z_coord=offset_z_coord_re[n])
        diff_re = (h_qc_offset_re - h_qc_0) / delta_z
        h_qc_offset_im = model.h_qc(constants, parameters, z_coord=offset_z_coord_im[n])
        diff_im = (h_qc_offset_im - h_qc_0) / delta_z
        dh_qc_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
    return dh_qc_dzc

def two_level_system_h_q(model, constants, parameters, **kwargs):
    """
    Calculate the quantum Hamiltonian for a two-level system.

    Model Ingredient:
        - model.h_q

    Model parameters:
        - constants.two_level_system_a (float): Parameter a.
        - constants.two_level_system_b (float): Parameter b.
        - constants.two_level_system_c (float): Parameter c.
        - constants.two_level_system_d (float): Parameter d.

    Related functions:
        - :func:`two_level_system_h_q_vectorized`
    """
    del kwargs
    h_q = np.zeros((2, 2), dtype=complex)
    h_q[0, 0] = constants.two_level_system_a
    h_q[1, 1] = constants.two_level_system_b
    h_q[0, 1] = (constants.two_level_system_c + 1j * constants.two_level_system_d)
    h_q[1, 0] = (constants.two_level_system_c - 1j * constants.two_level_system_d)
    return h_q


def nearest_neighbor_lattice_h_q(model, constants, parameters, **kwargs):
    """
    Calculate the quantum Hamiltonian for a nearest-neighbor lattice.

    Model Ingredient:
        - model.h_q

    Model parameters:
        - constants.nearest_neighbor_lattice_h_q_num_sites (int): Number of sites.
        - constants.nearest_neighbor_lattice_h_q_hopping_energy (complex): Hopping energy.
        - constants.nearest_neighbor_lattice_h_q_periodic_boundary (bool):
          Periodic boundary condition.

    Related functions:
        - :func:`nearest_neighbor_lattice_h_q_vectorized`
    """
    del kwargs
    num_sites = constants.nearest_neighbor_lattice_h_q_num_sites
    hopping_energy = constants.nearest_neighbor_lattice_h_q_hopping_energy
    periodic_boundary = constants.nearest_neighbor_lattice_h_q_periodic_boundary
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


def holstein_lattice_h_qc(model, constants, parameters, **kwargs):
    """
    Calculate the quantum-classical Hamiltonian for a Holstein lattice.

    Model Ingredient:
        - model.h_qc

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.
        - constants.holstein_lattice_h_qc_dimensionless_coupling (float): 
        Dimensionless coupling.

    Related functions:
        - :func:`holstein_lattice_h_qc_vectorized`
    """
    z_coord = kwargs["z_coord"]
    oscillator_frequency = constants.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = (
        constants.holstein_lattice_h_qc_dimensionless_coupling
    )
    h_qc = (
        np.diag(dimensionless_coupling * oscillator_frequency)
        * (z_coord + np.conj(z_coord))
        + 0.0j
    )
    return h_qc


def holstein_lattice_dh_qc_dzc(model, constants, parameters, **kwargs):
    """
    Calculate the derivative of the quantum-classical Hamiltonian with respect to the z-coordinates.

    Model Ingredient:
        - model.dh_qc_dzc

    Required keyword arguments:
        - z_coord (np.ndarray): The z-coordinates.

    Model parameters:
        - constants.holstein_lattice_h_qc_num_sites (int): Number of sites.
        - constants.holstein_lattice_h_qc_oscillator_frequency (float): Oscillator frequency.
        - constants.holstein_lattice_h_qc_dimensionless_coupling (float): 
        Dimensionless coupling.

    Related functions:
        - :func:`holstein_lattice_dh_qc_dzc_vectorized`
    """
    z_coord = kwargs["z_coord"]
    num_sites = constants.holstein_lattice_h_qc_num_sites
    oscillator_frequency = constants.holstein_lattice_h_qc_oscillator_frequency
    dimensionless_coupling = (
        constants.holstein_lattice_h_qc_dimensionless_coupling
    )
    dh_qc_dzc = np.zeros((num_sites, num_sites, num_sites), dtype=complex)
    np.einsum("iii->i", dh_qc_dzc)[...] = (
        dimensionless_coupling * oscillator_frequency *
        (np.ones_like(z_coord)) + 0.0j
    )
    return dh_qc_dzc

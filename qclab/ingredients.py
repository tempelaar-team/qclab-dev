import numpy as np

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
    h_c = np.sum(model.parameters.pq_weight[..., :] * np.conjugate(z_coord) * z_coord, axis=1)
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
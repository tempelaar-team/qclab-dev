import numpy as np
from numba import njit
import qclab.ingredients as ingredients
import warnings


def initialize_z_coord(sim, state, **kwargs):
    """
    Initialize the z-coordinate.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    seed = kwargs['seed']
    if hasattr(sim.model, 'init_classical'):
        state.modify('z_coord', sim.model.init_classical(seed=seed))
    else:
        state.modify('z_coord', ingredients.harmonic_oscillator_boltzmann_init_classical(sim.model, seed=state.seed))
    return state


def update_dh_c_dzc(sim, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian with respect to the z-coordinates.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    if hasattr(sim.model, 'dh_c_dzc'):
        # Use the model's built-in method if available
        state.modify('dh_c_dzc', sim.model.dh_c_dzc(z_coord=z_coord))
    else:
        # Approximate the gradient using finite differences
        delta_z = 1e-3
        offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
        offset_z_coord_im = z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z

        h_c_0 = sim.model.h_c(z_coord=z_coord)
        dh_c_dzc = np.zeros((len(z_coord), *np.shape(h_c_0)), dtype=complex)

        for n in range(len(z_coord)):
            h_c_offset_re = sim.model.h_c(z_coord=offset_z_coord_re[n])
            diff_re = (h_c_offset_re - h_c_0) / delta_z
            h_c_offset_im = sim.model.h_c(z_coord=offset_z_coord_im[n])
            diff_im = (h_c_offset_im - h_c_0) / (1j * delta_z)
            dh_c_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
        state.modify('dh_c_dzc', dh_c_dzc)
    return state


def update_dh_c_dzc_vectorized(sim, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian with respect to the z-coordinates (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    if hasattr(sim.model, 'dh_c_dzc_vectorized'):
        state.modify('dh_c_dzc', sim.model.dh_c_dzc_vectorized(z_coord=z_coord))
    else:
        state.modify('dh_c_dzc', np.array([sim.model.dh_c_dzc(z_coord=z_coord[n]) for n in range(len(z_coord))]))
        warnings.warn("dh_c_dzc_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
    return state


def update_dh_qc_dzc(sim, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian with respect to the z-coordinates.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    if hasattr(sim.model, 'dh_qc_dzc'):
        # Use the model's built-in method if available
        state.modify('dh_qc_dzc', sim.model.dh_qc_dzc(z_coord=z_coord))
    else:
        # Approximate the gradient using finite differences
        delta_z = 1e-3
        offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
        offset_z_coord_im = z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z

        # Check if model.h_qc is callable
        if not hasattr(sim.model, 'h_qc') or not callable(sim.model.h_qc):
            raise AttributeError("model must have a callable h_qc method.")

        h_qc_0 = sim.model.h_qc(z_coord=z_coord)
        dh_qc_dzc = np.zeros((len(z_coord), *np.shape(h_qc_0)), dtype=complex)

        for n in range(len(state.z_coord)):
            h_qc_offset_re = sim.model.h_qc(z_coord=offset_z_coord_re[n])
            diff_re = (h_qc_offset_re - h_qc_0) / delta_z
            h_qc_offset_im = sim.model.h_qc(z_coord=offset_z_coord_im[n])
            diff_im = (h_qc_offset_im - h_qc_0) / (1j * delta_z)
            dh_qc_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
        state.modify('dh_qc_dzc', dh_qc_dzc)
    return state


def update_dh_qc_dzc_vectorized(sim, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian with respect to the z-coordinates (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    if hasattr(sim.model, 'dh_qc_dzc_vectorized'):
        state.modify('dh_qc_dzc', sim.model.dh_qc_dzc_vectorized(z_coord=z_coord))
    else:
        state.modify('dh_qc_dzc', np.array([sim.model.dh_qc_dzc(z_coord=z_coord[n]) for n in range(len(z_coord))]))
        warnings.warn("dh_qc_dzc_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
    return state


def update_classical_forces(sim, state, **kwargs):
    """
    Update the classical forces.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    state = update_dh_c_dzc(sim, state, z_coord=z_coord)
    state.modify('classical_forces', state.get('dh_c_dzc'))
    return state


def update_classical_forces_vectorized(sim, state, **kwargs):
    """
    Update the classical forces (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    state = update_dh_c_dzc_vectorized(sim, state, z_coord=z_coord)
    state.modify('classical_forces', state.dh_c_dzc)
    return state


def update_quantum_classical_forces(sim, state, **kwargs):
    """
    Update the quantum-classical forces.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    wf = kwargs['wf']
    state = update_dh_qc_dzc(sim, state, z_coord=z_coord)
    state.modify('quantum_classical_forces', np.dot(np.dot(state.dh_qc_dzc, wf), np.conj(wf)))
    return state


def update_quantum_classical_forces_vectorized(sim, state, **kwargs):
    """
    Update the quantum-classical forces (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    wf = kwargs['wf']
    state = update_dh_qc_dzc_vectorized(sim, state, z_coord=z_coord)
    state.modify('quantum_classical_forces', np.einsum('bnj,bj->bn', np.einsum('bnji,bi->bnj', state.dh_qc_dzc, wf), np.conj(wf)))
    return state


def update_z_coord_rk4(sim, state, **kwargs):
    """
    Update the z-coordinates using the 4th-order Runge-Kutta method.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    dt = sim.parameters.dt
    wf = kwargs['wf']
    update_quantum_classical_forces_bool = kwargs['update_quantum_classical_forces_bool']
    state = update_classical_forces(sim, state, z_coord=state.z_coord)
    state = update_quantum_classical_forces(sim, state, wf=wf, z_coord=state.z_coord)
    k1 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state = update_classical_forces(sim, state, z_coord=state.z_coord + 0.5 * dt * k1)
    if update_quantum_classical_forces_bool:
        state = update_quantum_classical_forces(sim, state, wf=wf, z_coord=state.z_coord + 0.5 * dt * k1)
    k2 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state = update_classical_forces(sim, state, z_coord=state.z_coord + 0.5 * dt * k2)
    if update_quantum_classical_forces_bool:
        state = update_quantum_classical_forces(sim, state, wf=wf, z_coord=state.z_coord + 0.5 * dt * k2)
    k3 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state = update_classical_forces(sim, state, z_coord=state.z_coord + dt * k3)
    if update_quantum_classical_forces_bool:
        state = update_quantum_classical_forces(sim, state, wf=wf, z_coord=state.z_coord + dt * k3)
    k4 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state.modify('z_coord', state.z_coord + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
    return state


def update_z_coord_rk4_vectorized(sim, state, **kwargs):
    """
    Update the z-coordinates using the 4th-order Runge-Kutta method (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    dt = sim.parameters.dt
    wf = kwargs['wf']
    update_quantum_classical_forces_bool = kwargs['update_quantum_classical_forces_bool']
    z_coord_0 = state.z_coord
    state = update_classical_forces_vectorized(sim, state, z_coord=z_coord_0)
    state = update_quantum_classical_forces_vectorized(sim, state, wf=wf, z_coord=z_coord_0)
    k1 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state = update_classical_forces_vectorized(sim, state, z_coord=z_coord_0 + 0.5 * dt * k1)
    if update_quantum_classical_forces_bool:
        state = update_quantum_classical_forces_vectorized(sim, state, wf=wf, z_coord=z_coord_0 + 0.5 * dt * k1)
    k2 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state = update_classical_forces_vectorized(sim, state, z_coord=z_coord_0 + 0.5 * dt * k2)
    if update_quantum_classical_forces_bool:
        state = update_quantum_classical_forces_vectorized(sim, state, wf=wf, z_coord=z_coord_0 + 0.5 * dt * k2)
    k3 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state = update_classical_forces_vectorized(sim, state, z_coord=z_coord_0 + dt * k3)
    if update_quantum_classical_forces_bool:
        state = update_quantum_classical_forces_vectorized(sim, state, wf=wf, z_coord=z_coord_0 + dt * k3)
    k4 = -1.0j * (state.classical_forces + state.quantum_classical_forces)
    state.modify('z_coord', z_coord_0 + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
    return state


def update_h_quantum(sim, state, **kwargs):
    """
    Update the quantum Hamiltonian.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    state.modify('h_quantum', sim.model.h_q() + sim.model.h_qc(z_coord=z_coord) + 0j)
    return state


def update_h_quantum_vectorized(sim, state, **kwargs):
    """
    Update the quantum Hamiltonian (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    if hasattr(sim.model, 'h_q_vectorized') and hasattr(sim.model, 'h_qc_vectorized'):
        state.modify('h_quantum', sim.model.h_q_vectorized() + sim.model.h_qc_vectorized(z_coord=z_coord) + 0j)
    else:
        state.modify('h_quantum', np.array([(sim.model.h_q() + sim.model.h_qc(z_coord=z_coord[n]) + 0j) for n in range(len(z_coord))]))
        warnings.warn("h_quantum_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
        warnings.warn("h_qc_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
    return state


def update_wf_db_rk4(sim, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    dt = sim.parameters.dt
    k1 = (-1j * np.dot(state.h_quantum, state.wf_db))
    k2 = (-1j * np.dot(state.h_quantum, state.wf_db + 0.5 * dt * k1))
    k3 = (-1j * np.dot(state.h_quantum, state.wf_db + 0.5 * dt * k2))
    k4 = (-1j * np.dot(state.h_quantum, state.wf_db + dt * k3))
    state.modify('wf_db', state.get('wf_db') + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
    return state


@njit
def mat_vec_branch(mat, vec):
    """
    Perform matrix-vector multiplication for each branch.

    Args:
        mat (ndarray): The matrix.
        vec (ndarray): The vector.

    Returns:
        ndarray: The result of the matrix-vector multiplication.
    """
    return np.sum(mat * vec[:, np.newaxis, :], axis=-1)


def update_wf_db_rk4_vectorized(sim, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    dt = sim.parameters.dt
    wf_db = state.wf_db
    h_quantum = state.h_quantum
    k1 = (-1j * mat_vec_branch(h_quantum, wf_db))
    k2 = (-1j * mat_vec_branch(h_quantum, (wf_db + 0.5 * dt * k1)))
    k3 = (-1j * mat_vec_branch(h_quantum, (wf_db + 0.5 * dt * k2)))
    k4 = (-1j * mat_vec_branch(h_quantum, (wf_db + dt * k3)))
    state.modify('wf_db', wf_db + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
    return state


def update_dm_db_mf(sim, state, **kwargs):
    """
    Update the density matrix in the mean-field approximation.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    wf_db = state.wf_db
    state.modify('dm_db', np.outer(wf_db, np.conj(wf_db)))
    return state


def update_dm_db_mf_vectorized(sim, state, **kwargs):
    """
    Update the density matrix in the mean-field approximation (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    wf_db = state.wf_db
    state.modify('dm_db', np.einsum('bi,bj->bij', wf_db, np.conj(wf_db)))
    return state


def update_classical_energy(sim, state, **kwargs):
    """
    Update the classical energy.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    state.modify('classical_energy', np.real(sim.model.h_c(z_coord=z_coord))[np.newaxis])
    return state


def update_classical_energy_vectorized(sim, state, **kwargs):
    """
    Update the classical energy (vectorized).

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    z_coord = kwargs['z_coord']
    if hasattr(sim.model, 'h_c_vectorized'):
        state.modify('classical_energy', np.real(sim.model.h_c_vectorized(z_coord=z_coord)))
    else:
        state.modify('classical_energy', np.array([np.real(sim.model.h_c_vectorized(z_coord=z_coord[n])) for n in range(len(z_coord))]))
        warnings.warn("h_c_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
    return state


def update_quantum_energy(sim, state, **kwargs):
    """
    Update the quantum energy.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    wf = kwargs['wf']
    state.modify('quantum_energy', np.real(np.matmul(np.conj(wf), np.matmul(state.h_quantum, wf)))[np.newaxis])
    return state
import numpy as np
from numba import njit
import qclab.ingredients as ingredients
import warnings


def vector_apply_all_but_last(func, arg):
    """
    Vectorizes a function over all but the last index of the input array.

    Parameters:
    func (callable): The function to be vectorized. It should take a 1D array as input and return an array.
    arg (ndarray): The input array to be processed. The function will be applied to all slices along the last axis.

    Returns:
    ndarray: The output array with the function applied to each slice along the last axis of the input array.
    """
    arg_vec = arg.reshape((np.prod(np.shape(arg)[:-1]), np.shape(arg)[-1]))  # Reshape to 2D array
    val_vec = np.array([func(arg_vec[n]) for n in range(len(arg_vec))])  # Vectorize function
    return val_vec.reshape((*np.shape(arg)[:-1], *np.shape(val_vec)[1:]))


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

def dh_c_dzc_finite_differences(sim, state, **kwargs):
    z_coord = kwargs['z_coord']
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
        diff_im = (h_c_offset_im - h_c_0) / delta_z
        dh_c_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
    return dh_c_dzc


def dh_qc_dzc_finite_differences(sim, state, **kwargs):
    z_coord = kwargs['z_coord']
    # Approximate the gradient using finite differences
    delta_z = 1e-3
    offset_z_coord_re = z_coord[np.newaxis, :] + np.identity(len(z_coord)) * delta_z
    offset_z_coord_im = z_coord[np.newaxis, :] + 1j * np.identity(len(z_coord)) * delta_z

    h_qc_0 = sim.model.h_qc(z_coord=z_coord)
    dh_qc_dzc = np.zeros((len(z_coord), *np.shape(h_qc_0)), dtype=complex)

    for n in range(len(z_coord)):
        h_qc_offset_re = sim.model.h_qc(z_coord=offset_z_coord_re[n])
        diff_re = (h_qc_offset_re - h_qc_0) / delta_z
        h_qc_offset_im = sim.model.h_qc(z_coord=offset_z_coord_im[n])
        diff_im = (h_qc_offset_im - h_qc_0) / delta_z
        dh_qc_dzc[n] = 0.5 * (diff_re + 1j * diff_im)
    return dh_qc_dzc

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
        warnings.warn("dh_c_dzc_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
        if hasattr(sim.model, 'dh_c_dzc'):
            state.modify('dh_c_dzc', vector_apply_all_but_last(lambda z: sim.model.dh_c_dzc(z_coord=z), z_coord))
        else:
            warnings.warn("dh_c_dzc not implemented for this model. Using finite differences.", UserWarning)
            state.modify('dh_c_dzc', vector_apply_all_but_last(lambda z: dh_c_dzc_finite_differences(sim, state, z_coord=z), z_coord))

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
        warnings.warn("dh_qc_dzc_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
        if hasattr(sim.model, 'dh_qc_dzc'):
            state.modify('dh_qc_dzc', vector_apply_all_but_last(lambda z: sim.model.dh_qc_dzc(z_coord=z), z_coord))
        else:
            warnings.warn("dh_qc_dzc not implemented for this model. Using finite differences.", UserWarning)
            state.modify('dh_qc_dzc', vector_apply_all_but_last(lambda z: dh_qc_dzc_finite_differences(sim, state, z_coord=z), z_coord))

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
    state.modify('classical_forces', state.dh_c_dzc)
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
    state.modify('quantum_classical_forces',
                 np.einsum('...nj,...j->...n', np.einsum('...nji,...i->...nj', state.dh_qc_dzc, wf), np.conj(wf)))
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
    z_coord_0 = kwargs['z_coord']
    output_name = kwargs['output_name']
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
    state.modify(output_name, z_coord_0 + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
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
    if hasattr(sim.model, 'h_q_vectorized'):
        h_q = sim.model.h_q_vectorized()
    else:
        warnings.warn("h_q_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
        h_q = sim.model.h_q()
    if hasattr(sim.model, 'h_qc_vectorized'):
        h_tot = h_q + sim.model.h_qc_vectorized(z_coord=z_coord) + 0.0j
    else:
        warnings.warn("h_qc_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
        h_tot = np.array([(h_q + sim.model.h_qc(z_coord=z_coord[n]) + 0j) for n in range(len(z_coord))]) 
    state.modify('h_quantum', h_tot)
    #if hasattr(sim.model, 'h_q_vectorized') and hasattr(sim.model, 'h_qc_vectorized'):
    #    state.modify('h_quantum', sim.model.h_q_vectorized() + sim.model.h_qc_vectorized(z_coord=z_coord) + 0j)
    #else:
    #    state.modify('h_quantum', np.array([(sim.model.h_q() + sim.model.h_qc(z_coord=z_coord[n]) + 0j) for n in range(len(z_coord))]))
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
    state.modify('wf_db', state.wf_db + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4))
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
    state.modify('dm_db', np.einsum('...i,...j->...ij', wf_db, np.conj(wf_db)))
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
    state.modify('classical_energy', sim.model.h_c(z_coord=z_coord)[np.newaxis])
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
        state.modify('classical_energy', sim.model.h_c_vectorized(z_coord=z_coord)[:,np.newaxis])
    else:
        warnings.warn("h_c_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
        state.modify('classical_energy', vector_apply_all_but_last(lambda z: sim.model.h_c(z_coord=z), z_coord)[:,np.newaxis])
    return state


def update_classical_energy_fssh_vectorized(sim, state, **kwargs):
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
        state.modify('classical_energy',np.sum(sim.model.h_c_vectorized(z_coord=z_coord), axis=-1)[:, np.newaxis])
    else:
        state.modify('classical_energy', vector_apply_all_but_last(lambda z: sim.model.h_c(z_coord=z), z_coord))
        warnings.warn("h_c_vectorized not implemented for this model. Using non-vectorized method.", UserWarning)
    return state


def update_quantum_energy_mf(sim, state, **kwargs):
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
    state.modify('quantum_energy', np.matmul(np.conj(wf), np.matmul(state.h_quantum, wf))[np.newaxis])
    return state


def update_quantum_energy_mf_vectorized(sim, state, **kwargs):
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
    state.modify('quantum_energy', np.einsum('...i,...ij,...j->...', np.conj(wf), state.h_quantum, wf)[:, np.newaxis])
    return state


def update_quantum_energy_fssh_vectorized(sim, state, **kwargs):
    state.modify('quantum_energy', 
        np.einsum('...bi,...bij,...bj->...', np.conj(state.act_surf_wf), state.h_quantum, state.act_surf_wf)[:,
                                   np.newaxis])
    return state


def broadcast_var_to_branch_vectorized(sim, state, **kwargs):
    name = kwargs['name']
    val = kwargs['val']
    out = np.zeros((len(val), sim.algorithm.parameters.num_branches, *np.shape(val)[1:])) + val[:, np.newaxis, ...]
    state.modify(name, out)
    return state


def diagonalize_matrix_vectorized(sim, state, **kwargs):
    matrix = kwargs['matrix']
    eigvals_name = kwargs['eigvals_name']
    eigvecs_name = kwargs['eigvecs_name']
    eigvals, eigvecs = np.linalg.eigh(matrix + 0.0j)
    state.modify(eigvals_name, eigvals)
    state.modify(eigvecs_name, eigvecs)
    return state


def analytic_der_couple_phase(sim, state, eigvals, eigvecs):
    der_couple_q_phase = np.ones(np.shape(eigvals), dtype=complex)
    der_couple_p_phase = np.ones(np.shape(eigvals), dtype=complex)
    for i in range(np.shape(eigvals)[-1] - 1):
        j = i + 1
        evec_i = eigvecs[..., i]
        evec_j = eigvecs[..., j]
        eval_i = eigvals[..., i]
        eval_j = eigvals[..., j]
        ev_diff = eval_j - eval_i
        plus = np.zeros_like(ev_diff)
        if np.any(np.abs(ev_diff) < 1e-12):
            plus[np.where(np.abs(ev_diff) < 1e-12)] = 1
            warnings.warn("Degenerate eigenvalues detected.")
        der_couple_zc = np.ascontiguousarray(
            np.einsum('...i,...nij,...j->...n', np.conj(evec_i), state.dh_qc_dzc, evec_j) / (
                (ev_diff + plus)[..., np.newaxis]))
        der_couple_z = np.ascontiguousarray(
            np.einsum('...i,...nij,...j->...n', np.conj(evec_i), np.einsum('...nij->...nji', state.dh_qc_dzc).conj(),
                      evec_j) / ((ev_diff + plus)[..., np.newaxis]))
        der_couple_p = np.sqrt(1 / (2 * sim.model.parameters.pq_weight * sim.model.parameters.mass))[..., :] * (
                der_couple_z - der_couple_zc)
        der_couple_q = np.sqrt(sim.model.parameters.pq_weight * sim.model.parameters.mass / 2)[..., :] * (
                der_couple_z + der_couple_zc)
        der_couple_q_angle = np.angle(der_couple_q[np.arange(der_couple_q.shape[0])[:, None], np.arange(
            der_couple_q.shape[1]), np.argmax(np.abs(der_couple_q), axis=-1)])
        der_couple_p_angle = np.angle(der_couple_p[np.arange(der_couple_p.shape[0])[:, None], np.arange(
            der_couple_p.shape[1]), np.argmax(np.abs(der_couple_p), axis=-1)])
        der_couple_q_angle[np.where(np.abs(der_couple_q_angle) < 1e-12)] = 0
        der_couple_p_angle[np.where(np.abs(der_couple_p_angle) < 1e-12)] = 0
        der_couple_q_phase[..., i + 1:] = np.exp(1.0j * der_couple_q_angle[..., np.newaxis]) * der_couple_q_phase[...,
                                                                                               i + 1:]
        der_couple_p_phase[..., i + 1:] = np.exp(1.0j * der_couple_p_angle[..., np.newaxis]) * der_couple_p_phase[...,
                                                                                               i + 1:]
    return der_couple_q_phase, der_couple_p_phase


def gauge_fix_eigs_vectorized(sim, state, **kwargs):
    eigvals = kwargs['eigvals']
    eigvecs = kwargs['eigvecs']
    eigvecs_previous = kwargs['eigvecs_previous']
    output_eigvecs_name = kwargs['output_eigvecs_name']
    if kwargs['gauge_fixing'] >= 1:
        phase = np.exp(-1.0j * np.angle(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)))
        eigvecs = np.einsum('...ai,...i->...ai', eigvecs, phase)
    if kwargs['gauge_fixing'] >= 2:
        z_coord = kwargs['z_coord']
        update_dh_qc_dzc_vectorized(sim, state, z_coord=z_coord)
        der_couple_q_phase, der_couple_p_phase = analytic_der_couple_phase(sim, state, eigvals, eigvecs)
        eigvecs = np.einsum('...ai,...i->...ai', eigvecs, np.conj(der_couple_q_phase))
    if kwargs['gauge_fixing'] >= 0:
        signs = np.sign(np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2))
        eigvecs = np.einsum('...ai,...i->...ai', eigvecs, signs)
    if kwargs['gauge_fixing'] == 2:
        der_couple_q_phase_new, der_couple_p_phase_new = analytic_der_couple_phase(sim, state, eigvals, eigvecs)
        if np.sum(np.abs(np.imag(der_couple_q_phase_new)) ** 2 + np.abs(np.imag(der_couple_p_phase_new)) ** 2) > 1e-10:
            # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
            # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            warnings.warn("Phase error encountered when fixing gauge analytically.", UserWarning)

    state.modify(output_eigvecs_name, eigvecs)
    return state


def copy_value_vectorized(sim, state, **kwargs):
    name = kwargs['name']
    val = kwargs['val']
    state.modify(name, np.copy(val))
    return state


def basis_transform_vec_vectorized(sim, state, **kwargs):
    # default is adb to db
    input_vec = kwargs['input_vec']
    basis = kwargs['basis']
    output_name = kwargs['output_name']
    state.modify(output_name, np.einsum('...ij,...j->...i', basis, input_vec, optimize='greedy'))
    return state


def basis_transform_mat_vectorized(sim, state, **kwargs):
    # default is adb to db
    input_mat = kwargs['input_mat']
    basis = kwargs['basis']
    output_name = kwargs['output_name']
    state.modify(output_name,
                 np.einsum('...ij,...jk,...lk->...il', basis, input_mat, np.conj(basis), optimize='greedy'))
    return state


def initialize_active_surface(sim, state, **kwargs):
    num_states = np.shape(state.wf_db)[-1]
    if sim.algorithm.parameters.fssh_deterministic:
        if sim.algorithm.parameters.num_branches != num_states:
            raise ValueError("num_branches must be equal to the quantum dimension for deterministic FSSH.")
        act_surf_ind_0 = np.arange(sim.algorithm.parameters.num_branches, dtype=int)
    else:
        intervals = np.zeros(num_states)
        for state_n in range(num_states):
            intervals[state_n] = np.real(np.sum((np.abs(state.wf_adb_branch[0]) ** 2)[0:state_n + 1]))
        act_surf_ind_0 = np.zeros((sim.algorithm.parameters.num_branches), dtype=int)
        for branch_n in range(sim.algorithm.parameters.num_branches):
            act_surf_ind_0[branch_n] = \
                np.arange(num_states, dtype=int)[intervals > state.stochastic_sh_rand_vals[branch_n]][0]
        act_surf_ind_0 = np.sort(act_surf_ind_0)
    # initialize active surface and active surface index in each branch 
    state.modify('act_surf_ind_0', act_surf_ind_0.astype(int))
    state.modify('act_surf_ind', act_surf_ind_0.astype(int))
    act_surf = np.zeros((sim.algorithm.parameters.num_branches, num_states), dtype=int)
    act_surf[np.arange(sim.algorithm.parameters.num_branches, dtype=int), state.act_surf_ind] = 1
    state.modify('act_surf', act_surf.reshape((sim.algorithm.parameters.num_branches, num_states)).astype(int))
    return state


def initialize_random_values_fssh(sim, state, **kwargs):
    np.random.seed(state.seed)
    state.modify('hopping_probs_rand_vals', np.random.rand(len(sim.parameters.tdat)))
    state.modify('stochastic_sh_rand_vals', np.random.rand(sim.algorithm.parameters.num_branches))
    return state


def initialize_dm_adb_0_fssh_vectorized(sim, state, **kwargs):
    state.modify('dm_adb_0', np.einsum('...i,...j->...ij', state.wf_adb_branch, np.conj(state.wf_adb_branch)) + 0.0j)
    return state


def update_act_surf_wf_vectorized(sim, state, **kwargs):
    init_shape = np.shape(state.act_surf_ind)
    act_surf_ind_flat = state.act_surf_ind.reshape((np.prod(init_shape)))
    evecs_flat = state.eigvecs.reshape((np.prod(init_shape), *np.shape(state.eigvecs)[-2:]))[
                 np.arange(len(act_surf_ind_flat)), :, act_surf_ind_flat]
    act_surf_wf = evecs_flat.reshape((*init_shape, np.shape(state.eigvecs)[-1]))
    state.modify('act_surf_wf', act_surf_wf)
    return state


def update_dm_db_fssh_vectorized(sim, state, **kwargs):
    dm_adb_branch = np.einsum('...i,...j->...ij', state.wf_adb_branch, np.conj(state.wf_adb_branch), optimize='greedy')
    for nt in range(len(dm_adb_branch)):
        np.einsum('...jj->...j', dm_adb_branch[nt])[...] = state.act_surf[nt]
    if sim.algorithm.parameters.fssh_deterministic:
        dm_adb_branch = np.einsum('tbb->tb', state.dm_adb_0[..., 0, :, :])[..., np.newaxis, np.newaxis] * dm_adb_branch
    else:
        dm_adb_branch = dm_adb_branch / sim.algorithm.parameters.num_branches
    state.modify('dm_adb', np.sum(dm_adb_branch, axis=-3) + 0.0j)
    basis_transform_mat_vectorized(sim, state, input_mat=dm_adb_branch, basis=state.eigvecs, output_name='dm_db_branch')
    state.modify('dm_db', np.sum(state.dm_db_branch, axis=-3) + 0.0j)
    return state


def update_wf_db_eigs_vectorized(sim, state, **kwargs):
    wf_db = kwargs['wf_db']
    adb_name = kwargs['adb_name']
    output_name = kwargs['output_name']
    eigvals = kwargs['eigvals']
    eigvecs = kwargs['eigvecs']
    evals_exp = np.exp(-1.0j * eigvals * sim.parameters.dt)
    basis_transform_vec_vectorized(sim=sim, state=state, input_vec=wf_db,
                                   basis=np.einsum('...ij->...ji', eigvecs).conj(), output_name=adb_name)
    state.modify(adb_name, (state.wf_adb_branch * evals_exp))
    basis_transform_vec_vectorized(sim=sim, state=state, input_vec=state.wf_adb_branch, basis=eigvecs,
                                   output_name=output_name)
    return state


def initialize_timestep_index(sim, state, **kwargs):
    """
    Initialize the timestep index for the simulation.

    This function sets the timestep index (`t_ind`) in the state object to an array with a single element [0].

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    state.modify('t_ind', np.array([0]))
    return state


def update_timestep_index(sim, state, **kwargs):
    """
    Update the timestep index for the simulation.

    This function increments the timestep index (`t_ind`) in the state object.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    state.modify('t_ind', state.t_ind + 1)
    return state


@njit
def nan_num(num):
    """
    converts nan to a large or small number using numba acceleration
    """
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num


def update_active_surface_fssh(sim, state, **kwargs):
    """
    Execute the fewest-switches surface hopping (FSSH) procedure for updating the active surface.

    Args:
        sim (Simulation): The simulation object.
        state (State): The state object.
        **kwargs: Additional keyword arguments.

    Returns:
        State: The updated state object.
    """
    rand = state.hopping_probs_rand_vals[state.t_ind[0]]
    z_coord_branch = state.z_coord_branch
    act_surf_ind = state.act_surf_ind
    act_surf = state.act_surf

    for i in range(sim.algorithm.parameters.num_branches):
        # Calculate hopping probabilities
        prod = np.matmul(np.conj(state.eigvecs[i][:, state.act_surf_ind[i]]), state.eigvecs_previous[i])
        hop_prob = -2 * np.real(prod * (state.wf_adb_branch[i] / (state.wf_adb_branch[i][state.act_surf_ind[i]])))
        hop_prob[state.act_surf_ind[i]] = 0

        bin_edge = 0
        for k in range(len(hop_prob)):
            hop_prob[k] = nan_num(hop_prob[k])
            bin_edge += hop_prob[k]
            if rand < bin_edge:
                # Perform hopping
                evec_k = state.eigvecs[i][:, state.act_surf_ind[i]]
                evec_j = state.eigvecs[i][:, k]
                eval_k = state.eigvals[i][state.act_surf_ind[i]]
                eval_j = state.eigvals[i][k]
                ev_diff = eval_j - eval_k

                dkj_z = np.einsum('...i,...nij,...j->...n', np.conj(evec_k),
                                  np.einsum('...nij->...nji', state.dh_qc_dzc[i]).conj(), evec_j) / ev_diff[
                            ..., np.newaxis]
                dkj_zc = np.einsum('...i,...nij,...j->...n', np.conj(evec_k), state.dh_qc_dzc[i], evec_j) / ev_diff[
                    ..., np.newaxis]
                dkj_p = np.sqrt(1 / (2 * sim.model.parameters.pq_weight * sim.model.parameters.mass)) * (dkj_z - dkj_zc)
                dkj_q = np.sqrt(sim.model.parameters.pq_weight * sim.model.parameters.mass / 2) * (dkj_z + dkj_zc)

                max_pos_q = np.argmax(np.abs(dkj_q))
                max_pos_p = np.argmax(np.abs(dkj_p))

                # Check for complex nonadiabatic couplings
                if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(np.sin(np.angle(dkj_q[max_pos_q]))) > 1e-2:
                    warnings.warn('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!', UserWarning)
                if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(np.sin(np.angle(dkj_p[max_pos_p]))) > 1e-2:
                    warnings.warn('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!', UserWarning)

                delta_z = dkj_zc

                # Perform hopping using the model's hop function or the default harmonic oscillator hop function
                if hasattr(sim.model, 'hop_function'):
                    z_coord_branch_i, hopped = sim.model.hop_function(sim.model, z_coord_branch[i], delta_z, ev_diff)
                else:
                    z_coord_branch_i, hopped = ingredients.harmonic_oscillator_hop(sim.model, z_coord=z_coord_branch[i],
                                                                                   delta_z_coord=delta_z,
                                                                                   ev_diff=ev_diff)

                if hopped:
                    act_surf_ind[i] = k
                    act_surf[i] = np.zeros_like(act_surf[i])
                    act_surf[i][act_surf_ind[i]] = 1
                    z_coord_branch[i] = z_coord_branch_i
                break

    state.modify('act_surf_ind', act_surf_ind)
    state.modify('act_surf', act_surf)
    state.modify('z_coord_branch', z_coord_branch)
    return state

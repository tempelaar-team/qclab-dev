"""
This module contains tasks that update the state and parameters objects
during propagation.
"""

import logging
import numpy as np
from qc_lab import functions
import qc_lab.numerical_constants as numerical_constants

logger = logging.getLogger(__name__)


def update_t(sim, state, parameters, **kwargs):
    """
    Updates the time in the state object with the time index in each trajectory
    multiplied by the update timestep.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.t : ndarray
        Time of each trajectory.
    """
    batch_size = sim.settings.batch_size
    state.t = np.broadcast_to(sim.t_ind * sim.settings.dt_update, (batch_size,))
    return state, parameters


def update_dh_c_dzc_finite_differences(sim, state, parameters, **kwargs):
    """
    Calculates the gradient of the classical Hamiltonian using finite differences.

    Required Constants
    ------------------
    dh_c_dzc_finite_difference_delta : float, default : numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    Keyword Arguments
    -----------------
    z : ndarray
        Classical coordinates at which to evaluate the gradient.
    name : str, default: "dh_c_dzc"
        Name under which to store the finite-difference gradient in the state object.

    Variable Modifications
    -------------------
    state.{name} : ndarray
        Gradient of the classical Hamiltonian.
    """
    z = kwargs["z"]
    name = kwargs.get("name", "dh_c_dzc")
    delta_z = sim.model.constants.get(
        "dh_c_dzc_finite_difference_delta", numerical_constants.FINITE_DIFFERENCE_DELTA
    )
    batch_size = len(z)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    # Calculate increments in the real and imaginary directions.
    dz_re = delta_z * np.eye(num_classical_coordinates, dtype=z.dtype)
    dz_im = 1j * dz_re
    # Stack real/imag offset z coordinates.
    z_offset = z[:, None, :]
    z_offset_all = np.concatenate((z_offset + dz_re, z_offset + dz_im), axis=1).reshape(
        -1, num_classical_coordinates
    )
    # Get the quantum-classical Hamiltonian function.
    h_c, _ = sim.model.get("h_c")
    # Calculate it at the original z coordinate.
    h_c_0 = h_c(sim.model, parameters, z=z)
    # Calculate h_c at the offset coordinates.
    h_c_all = h_c(sim.model, parameters, z=z_offset_all).reshape(
        batch_size, 2 * num_classical_coordinates
    )
    # Split real/imag blocks of the offset h_c.
    h_c_re = h_c_all[:, :num_classical_coordinates]
    h_c_im = h_c_all[:, num_classical_coordinates:]
    # Calculate finite-difference derivatives.
    h_c_0_exp = h_c_0[:, None]
    diff_re = (h_c_re - h_c_0_exp) / delta_z
    diff_im = (h_c_im - h_c_0_exp) / delta_z
    dh_c_dzc = 0.5 * (diff_re + 1j * diff_im)
    setattr(state, name, dh_c_dzc)
    return state, parameters


def update_classical_forces(sim, state, parameters, **kwargs):
    """
    Updates the gradient of the classical Hamiltonian w.r.t. the conjugate classical
    coordinate.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of the classical coordinates in the state object.

    Variable Modifications
    -------------------
    state.classical_forces : ndarray
            Gradient of the classical Hamiltonian.
    """
    z = getattr(state, kwargs["z"])
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        state.classical_forces = dh_c_dzc(sim.model, parameters, z=z)
        return state, parameters
    if sim.settings.debug:
        logger.info("dh_c_dzc not found; using finite differences.")
    return update_dh_c_dzc_finite_differences(
        sim, state, parameters, name="classical_forces", z=z
    )


def update_dh_qc_dzc_finite_differences(sim, state, parameters, **kwargs):
    """
    Calculates the gradient of the quantum-classical Hamiltonian using finite
    differences.

    Required Constants
    ------------------
    dh_qc_dzc_finite_difference_delta : float, default : numerical_constants.FINITE_DIFFERENCE_DELTA
        Finite-difference step size.

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in the state object.

    Variable Modifications
    -------------------
    state.dh_qc_dzc : tuple
        Gradient of the quantum-classical Hamiltonian.
    """
    z = getattr(state, kwargs["z"])
    batch_size = len(z)
    delta_z = sim.model.constants.get(
        "dh_qc_dzc_finite_difference_delta", numerical_constants.FINITE_DIFFERENCE_DELTA
    )
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    num_quantum_states = sim.model.constants.num_quantum_states
    # Calculate increments in the real and imaginary directions.
    dz_re = delta_z * np.eye(num_classical_coordinates, dtype=z.dtype)
    dz_im = 1j * dz_re
    # Stack real/imag offset z coordinates.
    z_offset = z[:, None, :]
    z_offset_all = np.concatenate((z_offset + dz_re, z_offset + dz_im), axis=1).reshape(
        -1, num_classical_coordinates
    )
    # Get the quantum-classical Hamiltonian function.
    h_qc, _ = sim.model.get("h_qc")
    # Calculate it at the original z coordinate.
    h_qc_0 = h_qc(sim.model, parameters, z=z)
    # Calculate h_qc at the offset coordinates.
    h_qc_all = h_qc(sim.model, parameters, z=z_offset_all).reshape(
        batch_size,
        2 * num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    # Split real/imag blocks of the offset h_qc.
    h_qc_re = h_qc_all[:, :num_classical_coordinates, :, :]
    h_qc_im = h_qc_all[:, num_classical_coordinates:, :, :]
    # Calculate finite-difference derivatives.
    h_qc_0_exp = h_qc_0[:, None, :, :]
    diff_re = (h_qc_re - h_qc_0_exp) / delta_z
    diff_im = (h_qc_im - h_qc_0_exp) / delta_z
    dh_qc_dzc = 0.5 * (diff_re + 1j * diff_im)
    # Get sparse representation of dh_qc_dzc.
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    # Update it in the state object.
    state.dh_qc_dzc = (inds, mels, shape)
    return state, parameters


def update_dh_qc_dzc(sim, state, parameters, **kwargs):
    """
    Updates the gradient of the quantum-classical Hamiltonian w.r.t. the conjugate
    classical coordinate.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.

    Variable Modifications
    -------------------
    state.dh_qc_dzc : tuple
        Gradient of the quantum-classical Hamiltonian.
    """
    z = getattr(state, kwargs["z"])
    if state.dh_qc_dzc is None or sim.model.update_dh_qc_dzc:
        # If dh_qc_dzc has not been calculated yet, or if the
        # model requires it to be updated, calculate it.
        dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
        if has_dh_qc_dzc:
            state.dh_qc_dzc = dh_qc_dzc(sim.model, parameters, z=z)
            return state, parameters
        if sim.settings.debug:
            logger.info("dh_qc_dzc not found; using finite differences.")
        return update_dh_qc_dzc_finite_differences(
            sim, state, parameters, z=kwargs["z"]
        )
    # If dh_qc_dzc has already been calculated and does not need to be updated,
    # return the existing parameters and state objects.
    return state, parameters


def update_quantum_classical_forces(sim, state, parameters, **kwargs):
    """
    Updates the quantum-classical forces w.r.t. the wavefunction defined by ``wf_db``.

    If the model has a ``gauge_field_force`` ingredient, this term will be added
    to the quantum-classical forces.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.
    wf_db : str
        Name of the wavefunction (in the diabatic basis) in the state object.
    adb_state_ind: int
        Index of the adiabatic state from which to obtain the gauge field force.
        Required if ``algorithm.settings.use_gauge_field_force`` is ``True``.
    wf_changed : bool, default: True
        If ``True``, the wavefunction has changed since the last time the forces were calculated.

    Variable Modifications
    -------------------
    state.dh_qc_dzc : tuple
        Gradient of the quantum-classical Hamiltonian.
    state.quantum_classical_forces : ndarray
        Quantum-classical forces.
    """
    z = getattr(state, kwargs["z"])
    wf_db = getattr(state, kwargs["wf_db"])
    wf_changed = kwargs.get("wf_changed", True)
    adb_state_ind = kwargs.get("adb_state_ind", None)
    # Update the gradient of h_qc.
    state, parameters = update_dh_qc_dzc(sim, state, parameters, z=kwargs["z"])
    # Calculate the expectation value w.r.t. the wavefunction.
    # If not(wf_changed) and sim.model.update_dh_qc_dzc then recalculate.
    # If wf_changed then recalculate.
    # If state.quantum_classical_forces is None then recalculate.
    if (
        (state.quantum_classical_forces is None)
        or wf_changed
        or (not (wf_changed) and sim.model.update_dh_qc_dzc)
    ):
        if state.quantum_classical_forces is None:
            state.quantum_classical_forces = np.zeros(np.shape(z), dtype=complex)
        state.quantum_classical_forces = functions.calc_sparse_inner_product(
            *state.dh_qc_dzc,
            wf_db.conj(),
            wf_db,
            out=state.quantum_classical_forces.reshape(-1),
        ).reshape(np.shape(z))
    if sim.algorithm.settings.get("use_gauge_field_force"):
        state, parameters = add_gauge_field_force_fssh(
            sim, state, parameters, z=kwargs["z"], adb_state_ind=adb_state_ind
        )
    return state, parameters


def add_gauge_field_force_fssh(sim, state, parameters, **kwargs):
    """
    Adds the gauge field force to the quantum-classical forces if the model has a
    ``gauge_field_force`` ingredient.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.
    adb_state_ind : str, default: "act_surf_ind"
        Index of the adiabatic state from which to obtain the gauge field force.

    Variable Modifications
    -------------------
    state.quantum_classical_forces : ndarray
        Quantum-classical forces with gauge field force added.
    """
    z = getattr(state, kwargs["z"])
    adb_state_ind = getattr(state, kwargs.get("adb_state_ind", "act_surf_ind"))
    gauge_field_force, has_gauge_field_force = sim.model.get("gauge_field_force")
    if has_gauge_field_force:
        state.quantum_classical_forces += gauge_field_force(
            parameters, z=z, adb_state_ind=adb_state_ind
        )
    else:
        if sim.settings.debug:
            logger.warning("gauge_field_force not found; skipping.")
    return state, parameters


def diagonalize_matrix(sim, state, parameters, **kwargs):
    """
    Diagonalizes a given matrix from the state object and stores the eigenvalues and
    eigenvectors in the state object.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    matrix : str
        Name of the matrix to diagonalize in the state object.
    eigvals : str
        Name of the eigenvalues in the state object.
    eigvecs : str
        Name of the eigenvectors in the state object.

    Variable Modifications
    -------------------
    state.{eigvals} : ndarray
        Eigenvalues of the matrix.
    state.{eigvecs} : ndarray
        Eigenvectors of the matrix.
    """
    matrix = getattr(state, kwargs["matrix"])
    eigvals, eigvecs = np.linalg.eigh(matrix)
    setattr(state, kwargs["eigvals"], eigvals)
    setattr(state, kwargs["eigvecs"], eigvecs)
    return state, parameters


def gauge_fix_eigs(sim, state, parameters, **kwargs):
    """
    Fixes the gauge of the eigenvectors as specified by the gauge_fixing parameter.


    if gauge_fixing == "sign_overlap":
        The sign of the eigenvector is changed so the real part of its
        overlap with the previous eigenvector is positive.

    if gauge_fixing == "phase_overlap":
        The phase of the eigenvector is determined from its overlap
        with the previous eigenvector and used to maximize the real
        part of the overlap. The sign is then changed so the real
        part of the overlap is positive.

    if gauge_fixing == "phase_der_couple":
        The phase of the eigenvector is determined by calculating the
        derivative couplings and changed so that all the derivative
        couplings are real-valued.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    eigvals : str
        Name of the eigenvalues in the state object.
    eigvecs : str
        Name of the eigenvectors in the state object.
    eigvecs_previous : str
        Name of the previous eigenvectors in the state object.
    output_eigvecs : str
        Name of the output gauge-fixed eigenvectors in the state object.
    z : str
        Name of classical coordinates in the state object.
    gauge_fixing : str, default: sim.algorithm.settings.gauge_fixing
        Gauge-fixing method to use.

    Variable Modifications
    -------------------
    state.{output_eigvecs} : ndarray
        Gauge-fixed eigenvectors.
    """
    eigvals = getattr(state, kwargs["eigvals"])
    eigvecs = getattr(state, kwargs["eigvecs"])
    eigvecs_previous = getattr(state, kwargs["eigvecs_previous"])
    gauge_fixing = kwargs.get("gauge_fixing", sim.algorithm.settings.gauge_fixing)
    gauge_fixing_numerical_values = {
        "sign_overlap": 0,
        "phase_overlap": 1,
        "phase_der_couple": 2,
    }
    gauge_fixing_value = gauge_fixing_numerical_values[gauge_fixing]
    if gauge_fixing_value not in {0, 1, 2}:
        logger.critical("Invalid gauge_fixing value: %s", gauge_fixing)
        raise ValueError(f"Invalid gauge_fixing value: {gauge_fixing}")
    if gauge_fixing_value == 1:
        # Maximize the real part of the overlap (guaranteed to be positive).
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        phase = np.exp(-1j * np.angle(overlap))
        eigvecs *= phase[:, None, :]
    if gauge_fixing_value == 2:
        # Make the derivative couplings real-valued (but not necessarily positive).
        state, parameters = update_dh_qc_dzc(sim, state, parameters, z=kwargs["z"])
        der_couple_q_phase, _ = functions.analytic_der_couple_phase(
            sim, state.dh_qc_dzc, eigvals, eigvecs
        )
        eigvecs *= np.conj(der_couple_q_phase)[:, None, :]
    if gauge_fixing_value == 0 or gauge_fixing_value == 2:
        # Make the real part positive based on the sign of the real part of the overlap.
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        signs = np.sign(np.real(overlap))
        eigvecs *= signs[:, None, :]

    if gauge_fixing_value == 2 and sim.settings.debug:
        der_couple_q_phase_new, der_couple_p_phase_new = (
            functions.analytic_der_couple_phase(sim, state.dh_qc_dzc, eigvals, eigvecs)
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_q_phase_new)) ** 2
                + np.abs(np.imag(der_couple_p_phase_new)) ** 2
            )
            > numerical_constants.SMALL
        ):
            logger.error(
                "Phase error encountered when fixing gauge analytically. %s",
                np.sum(
                    np.abs(np.imag(der_couple_q_phase_new)) ** 2
                    + np.abs(np.imag(der_couple_p_phase_new)) ** 2
                ),
            )
    setattr(state, kwargs["output_eigvecs"], eigvecs)
    return state, parameters


def basis_transform_vec(sim, state, parameters, **kwargs):
    """
    Transforms a vector ``state.{input}`` to a new basis defined by
    ``state.{basis}`` and stores it in ``state.{output}``.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    input : str
        Name of the vector to transform in the state object.
    basis : str
        Name of the basis to transform to in the state object.
        Assumed to be column vectors corresponding to adiabatic
        states.
    output : str
        Name of the output vector in the state object.
    adb_to_db : bool, default: False
        If True, transforms from adiabatic to diabatic. If False, transforms from
        adiabatic to diabatic.

    Variable Modifications
    -------------------
    state.{output} : ndarray
        Vector expressed in the new basis.
    """
    input_vec = getattr(state, kwargs["input"])
    basis = getattr(state, kwargs["basis"])
    adb_to_db = kwargs["adb_to_db"]
    setattr(
        state,
        kwargs["output"],
        functions.transform_vec(input_vec, basis, adb_to_db=adb_to_db),
    )
    return state, parameters


def update_act_surf_wf(sim, state, parameters, **kwargs):
    """
    Updates the wavefunction corresponding to the active surface.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.act_surf_wf : ndarray
        Wavefunction of the active surface.
    """
    num_trajs = sim.settings.batch_size
    act_surf_wf = state.eigvecs[
        np.arange(num_trajs, dtype=int),
        :,
        state.act_surf_ind,
    ]
    state.act_surf_wf = act_surf_wf
    return state, parameters


def update_wf_db_eigs(sim, state, parameters, **kwargs):
    """
    Evolves the diabatic wavefunction ``state.{wf_db}`` using the
    eigenvalues ``state.{eigvals}`` and eigenvectors ``state.{eigvecs}``.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf_db : str
        Name of the diabatic wavefunction in the state object.
    eigvals : str
        Name of the eigenvalues in the state object.
    eigvecs : str
        Name of the eigenvectors in the state object.

    Variable Modifications
    -------------------
    state.{wf_db} : ndarray
        Updated diabatic wavefunction.
    """
    eigvals = getattr(state, kwargs["eigvals"])
    eigvecs = getattr(state, kwargs["eigvecs"])
    wf_db = getattr(state, kwargs["wf_db"])
    batch_size = len(wf_db)
    num_quantum_states = sim.model.constants.num_quantum_states
    # Calculate the propagator in the adiabatic basis.
    prop_adb = np.zeros(
        (batch_size, num_quantum_states, num_quantum_states), dtype=complex
    )
    idx = np.arange(num_quantum_states)
    prop_adb[:, idx, idx] = np.exp(-1j * eigvals * sim.settings.dt_update)
    # Transform propagator to the diabatic basis.
    prop_db = functions.transform_mat(prop_adb, eigvecs, adb_to_db=True)
    # Apply the propagator to the diabatic wavefunction.
    setattr(
        state,
        kwargs["wf_db"],
        functions.multiply_matrix_vector(prop_db, wf_db),
    )
    return state, parameters


def update_wf_db_rk4(sim, state, parameters, **kwargs):
    """
    Updates the wavefunction ``state.{wf_db}`` using the 4th-order Runge-Kutta method.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf_db : str
        Name of the diabatic wavefunction in the state object.

    Variable Modifications
    -------------------
    state.{wf_db} : ndarray
        Updated diabatic wavefunction.
    """
    dt_update = sim.settings.dt_update
    wf_db = getattr(state, kwargs["wf_db"])
    h_quantum = state.h_quantum
    setattr(state, kwargs["wf_db"], functions.wf_db_rk4(h_quantum, wf_db, dt_update))
    return state, parameters


def update_hop_probs_fssh(sim, state, parameters, **kwargs):
    """
    Calculates the hopping probabilities for FSSH.

    :math:`P_{a\\rightarrow b} = -2\\Re((C_{b}/C_{a})\\langle a(t)| b(t-dt)\\rangle)`

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.hop_prob : ndarray
        Hopping probabilities between the active surface and all other surfaces.
    """
    act_surf_ind = state.act_surf_ind
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    # Check if any of the coefficients on the active surface are zero.
    if sim.settings.debug:
        if np.any(
            np.abs(state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind])
            == 0.0
        ):
            logger.warning(
                "Zero coefficient on active surface encountered when calculating hopping probabilities."
            )
    # Calculates < act_surf(t) | b(t-dt) >
    prod = functions.multiply_matrix_vector(
        np.swapaxes(state.eigvecs_previous, -1, -2),
        np.conj(
            state.eigvecs[
                np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind
            ]
        ),
    )
    # Calculates -2*Re( (C_b / C_act_surf) * < act_surf(t) | b(t-dt) > )
    hop_prob = -2.0 * np.real(
        prod
        * state.wf_adb
        / state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind][:, np.newaxis]
    )
    # Sets hopping probabilities to 0 at the active surface.
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind] = 0.0
    state.hop_prob = hop_prob

    return state, parameters


def update_hop_inds_fssh(sim, state, parameters, **kwargs):
    """
    Determines which trajectories hop based on the hopping probabilities and which state
    they hop to. Note that these will only hop if they are not frustrated by the hopping
    function.

    Stores the indices of the hopping trajectories in ``state.hop_ind``. Stores the
    destination indices of the hops in ``state.hop_dest``.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.hop_ind : ndarray
        Indices of trajectories that hop.
    state.hop_dest : ndarray
        Destination surface for each hop.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    hop_prob = state.hop_prob
    rand = state.hopping_probs_rand_vals[:, sim.t_ind]
    cumulative_probs = np.cumsum(
        np.nan_to_num(hop_prob, nan=0, posinf=100e100, neginf=-100e100, copy=False),
        axis=1,
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    hop_ind = np.where(
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )[0]
    hop_dest = np.argmax(
        (cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1
    )[hop_ind]
    state.hop_ind = hop_ind
    state.hop_dest = hop_dest
    return state, parameters


def update_z_shift_fssh(sim, state, parameters, **kwargs):
    """
    Determines if a hop occurs and calculates the shift in the classical coordinate
    at the single trajectory level.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : ndarray
        Classical coordinates at which to evaluate the hop.
    delta_z : ndarray
        Direction in which to rescale the coordinates.
    eigval_diff : float
        Difference in eigenvalues between the initial and final states (e_final - e_initial).

    Variable Modifications
    -------------------
    state.hop_successful_traj : bool
        Flag indicating if the hop was successful.
    state.z_shift_traj : ndarray
        Shift required to conserve energy.
    """
    z = kwargs["z"]
    delta_z = kwargs["delta_z"]
    eigval_diff = kwargs["eigval_diff"]
    hop, has_hop = sim.model.get("hop")
    if has_hop:
        z_shift, hopped = hop(
            sim.model,
            parameters,
            z=z,
            delta_z=delta_z,
            eigval_diff=eigval_diff,
        )
    else:
        z_shift, hopped = functions.numerical_fssh_hop(
            sim.model,
            parameters,
            z=z,
            delta_z=delta_z,
            eigval_diff=eigval_diff,
        )
    state.hop_successful_traj = hopped
    state.z_shift_traj = z_shift
    return state, parameters


def update_hop_vals_fssh(sim, state, parameters, **kwargs):
    """
    Executes the hopping function for the hopping trajectories.

    Stores the rescaled coordinates in ``state.z_rescaled`` and a Boolean registering
    if the hop was successful in ``state.hop_successful``.

    If the model has a ``rescaling_direction_fssh`` ingredient, it will be used to
    determine the direction in which to rescale the coordinates. Otherwise, the
    direction will be calculated with ``functions.calc_delta_z_fssh``.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.z_shift : ndarray
        Shift in coordinates for each hopping trajectory.
    state.hop_successful : ndarray
        Flags indicating if each hop was successful.
    """

    hop_ind = state.hop_ind
    hop_dest = state.hop_dest
    state.z_shift = np.zeros(
        (len(hop_ind), sim.model.constants.num_classical_coordinates), dtype=complex
    )
    state.hop_successful = np.zeros(len(hop_ind), dtype=bool)
    eigvals = state.eigvals
    eigvecs = state.eigvecs
    z = state.z
    act_surf_ind = state.act_surf_ind
    hop_traj_ind = 0
    for traj_ind in hop_ind:
        final_state_ind = hop_dest[hop_traj_ind]
        init_state_ind = act_surf_ind[traj_ind]
        eigval_init_state = eigvals[traj_ind][init_state_ind]
        eigval_final_state = eigvals[traj_ind][final_state_ind]
        eigval_diff = eigval_final_state - eigval_init_state
        eigvec_init_state = eigvecs[traj_ind, :, init_state_ind]
        eigvec_final_state = eigvecs[traj_ind, :, final_state_ind]
        rescaling_direction_fssh, has_rescaling_direction_fssh = sim.model.get(
            "rescaling_direction_fssh"
        )
        if has_rescaling_direction_fssh:
            delta_z = rescaling_direction_fssh(
                parameters,
                z=state.z[traj_ind],
                init_state_ind=init_state_ind,
                final_state_ind=final_state_ind,
            )
        else:
            inds, mels, shape = state.dh_qc_dzc
            dh_qc_dzc_traj_ind = inds[0] == traj_ind
            inds_traj = (
                inds[0][dh_qc_dzc_traj_ind],
                inds[1][dh_qc_dzc_traj_ind],
                inds[2][dh_qc_dzc_traj_ind],
                inds[3][dh_qc_dzc_traj_ind],
            )
            mels_traj = mels[dh_qc_dzc_traj_ind]
            shape_traj = (1, shape[1], shape[2], shape[3])
            dh_qc_dzc_traj = (inds_traj, mels_traj, shape_traj)
            delta_z = functions.calc_delta_z_fssh(
                sim, eigval_diff, eigvec_init_state, eigvec_final_state, dh_qc_dzc_traj
            )
        state, parameters = update_z_shift_fssh(
            sim,
            state,
            parameters,
            z=z[traj_ind],
            delta_z=delta_z,
            eigval_diff=eigval_diff,
        )
        state.hop_successful[hop_traj_ind] = state.hop_successful_traj
        state.z_shift[hop_traj_ind] = state.z_shift_traj
        hop_traj_ind += 1
    return state, parameters


def update_z_hop_fssh(sim, state, parameters, **kwargs):
    """
    Applies the shift in coordinates after successful hops.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.z : ndarray
        Classical coordinates.
    """
    state.z[state.hop_ind] += state.z_shift
    return state, parameters


def update_act_surf_hop_fssh(sim, state, parameters, **kwargs):
    """
    Updates the active surface, active surface index, and active surface wavefunction
    following a hop in FSSH.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.act_surf_ind : ndarray
        Active surface indices.
    state.act_surf : ndarray
        Active surface wavefunctions.
    """
    # Get the index of the trajectories that successfully hopped.
    hop_successful_traj_ind = state.hop_ind[state.hop_successful]
    # Get their destination states.
    hop_dest_ind = state.hop_dest[state.hop_successful]
    # Zero out the active surface in the ones that hopped.
    state.act_surf[hop_successful_traj_ind] = 0
    # Set the new active surface to 1.
    state.act_surf[hop_successful_traj_ind, hop_dest_ind] = 1
    # Update the active surface index.
    state.act_surf_ind[hop_successful_traj_ind] = hop_dest_ind

    return state, parameters


def update_h_quantum(sim, state, parameters, **kwargs):
    """
    Updates the Hamiltonian matrix of the quantum subsystem.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.

    Variable Modifications
    -------------------
    state.h_q : ndarray
        Quantum Hamiltonian matrix.
    state.h_qc : ndarray
        Quantum-classical coupling matrix.
    state.h_quantum : ndarray
        Total Hamiltonian of the quantum subsystem.
    """
    z = getattr(state, kwargs["z"])
    h_q, _ = sim.model.get("h_q")
    h_qc, _ = sim.model.get("h_qc")
    if sim.model.update_h_q or state.h_q is None:
        # Update the quantum Hamiltonian if required or if it is not set.
        state.h_q = h_q(sim.model, parameters, batch_size=sim.settings.batch_size)
    # Update the quantum-classical Hamiltonian.
    state.h_qc = h_qc(sim.model, parameters, z=z)
    state.h_quantum = state.h_q + state.h_qc
    return state, parameters


def update_z_rk4_k1(sim, state, parameters, **kwargs):
    """
    Computes the first RK4 intermediate for evolving the classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    state.{output} : ndarray
        Output coordinates after half step.
    state.z_rk4_k1 : ndarray
        First RK4 slope.
    """
    dt_update = sim.settings.dt_update
    z_k = getattr(state, kwargs["z"])
    output = kwargs["output"]
    out, k1 = functions.update_z_rk4_k123_sum(
        z_k, state.classical_forces, state.quantum_classical_forces, 0.5 * dt_update
    )
    setattr(state, output, out)
    state.z_rk4_k1 = k1
    return state, parameters


def update_z_rk4_k2(sim, state, parameters, **kwargs):
    """
    Computes the second RK4 intermediate for evolving the classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    state.{output} : ndarray
        Output coordinates after half step.
    state.z_rk4_k2 : ndarray
        Second RK4 slope.
    """
    dt_update = sim.settings.dt_update
    z_k = getattr(state, kwargs["z"])
    output = kwargs["output"]
    out, k2 = functions.update_z_rk4_k123_sum(
        z_k, state.classical_forces, state.quantum_classical_forces, 0.5 * dt_update
    )
    setattr(state, output, out)
    state.z_rk4_k2 = k2
    return state, parameters


def update_z_rk4_k3(sim, state, parameters, **kwargs):
    """
    Computes the third RK4 intermediate for evolving the classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    state.{output} : ndarray
        Output coordinates after a full step.
    state.z_rk4_k3 : ndarray
        Third RK4 slope.
    """
    dt_update = sim.settings.dt_update
    z_k = getattr(state, kwargs["z"])
    output = kwargs["output"]
    out, k3 = functions.update_z_rk4_k123_sum(
        z_k, state.classical_forces, state.quantum_classical_forces, dt_update
    )
    setattr(state, output, out)
    state.z_rk4_k3 = k3
    return state, parameters


def update_z_rk4_k4(sim, state, parameters, **kwargs):
    """
    Computes the final RK4 update for evolving the classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    state.{output} : ndarray
        Output coordinates after a full step.
    """
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, kwargs["z"])
    output = kwargs["output"]
    out = functions.update_z_rk4_k4_sum(
        z_0,
        state.z_rk4_k1,
        state.z_rk4_k2,
        state.z_rk4_k3,
        state.classical_forces,
        state.quantum_classical_forces,
        dt_update,
    )
    setattr(state, output, out)
    return state, parameters


def update_dm_db_mf(sim, state, parameters, **kwargs):
    """
    Updates the diabatic density matrix based on the wavefunction.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf_db : str
        Name of the diabatic wavefunction in the state object.
    dm_db : str, default: "dm_db"
        Name of the diabatic density matrix in the state object.

    Variable Modifications
    -------------------
    state.{dm_db} : ndarray
        Diabatic density matrix.
    """
    wf_db = getattr(state, kwargs["wf_db"])
    dm_db_name = kwargs.get("dm_db", "dm_db")
    setattr(state, dm_db_name, np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy"))
    return state, parameters


def update_classical_energy(sim, state, parameters, **kwargs):
    """
    Updates the classical energy.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.

    Variable Modifications
    -------------------
    state.classical_energy : ndarray
        Energy of the classical subsystem.
    """
    z = getattr(state, kwargs["z"])
    h_c, _ = sim.model.get("h_c")
    state.classical_energy = np.real(h_c(sim.model, parameters, z=z, batch_size=len(z)))
    return state, parameters


def update_classical_energy_fssh(sim, state, parameters, **kwargs):
    """
    Updates the classical energy for FSSH simulations. If deterministic, the energy in
    each branch is summed together with weights determined by the initial adiabatic
    populations. If not deterministic (and so there is only one branch), the energy is
    computed for the single branch.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.

    Variable Modifications
    -------------------
    state.classical_energy : ndarray
        Energy of the classical subsystem.
    """
    z = getattr(state, kwargs["z"])
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_states = sim.model.constants.num_quantum_states
    h_c, _ = sim.model.get("h_c")
    if sim.algorithm.settings.fssh_deterministic:
        state.classical_energy = 0.0
        branch_weights = num_branches * np.einsum(
            "tbbb->tb",
            state.dm_adb_0.reshape((batch_size, num_branches, num_states, num_states)),
        )
        for branch_ind in range(num_branches):
            z_branch = z[state.branch_ind == branch_ind]
            state.classical_energy = state.classical_energy + branch_weights[
                :, branch_ind
            ] * h_c(
                sim.model,
                parameters,
                z=z_branch,
                batch_size=len(z_branch),
            )
    else:
        state.classical_energy = 0.0
        z_branch = z[state.branch_ind == 0]
        state.classical_energy = state.classical_energy + h_c(
            sim.model,
            parameters,
            z=z_branch,
            batch_size=len(z_branch),
        )
        state.classical_energy = state.classical_energy
    state.classical_energy = np.real(state.classical_energy)
    return state, parameters


def update_quantum_energy(sim, state, parameters, **kwargs):
    """
    Updates the quantum energy w.r.t. the wavefunction specified by ``wf_db`` by taking
    the expectation value of ``state.h_quantum``.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf_db : str
        Name of the wavefunction in the state object.

    Variable Modifications
    -------------------
    state.quantum_energy : ndarray
        Quantum energy.
    """
    wf_db = getattr(state, kwargs["wf_db"])
    state.quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf_db), state.h_quantum, wf_db, optimize="greedy")
    )
    return state, parameters


def update_quantum_energy_fssh(sim, state, parameters, **kwargs):
    """
    Updates the quantum energy w.r.t. the wavefunction specified by ``wf_db``.
    Accounts for both stochastic and deterministic FSSH modes.
    Typically, the wavefunction used is that of the active surface.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf_db : str
        Name of the wavefunction in the state object.

    Variable Modifications
    -------------------
    state.quantum_energy : ndarray
        Quantum energy.
    """
    wf_db = getattr(state, kwargs["wf_db"])

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf_db = wf_db * np.sqrt(
            num_branches
            * np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_states, num_states)
                ),
                optimize="greedy",
            ).flatten()[:, np.newaxis]
        )
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf_db), state.h_quantum, wf_db, optimize="greedy"
        )
    else:
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf_db), state.h_quantum, wf_db, optimize="greedy"
        )
        state.quantum_energy = state.quantum_energy
    state.quantum_energy = np.real(state.quantum_energy)
    return state, parameters


def update_dm_db_fssh(sim, state, parameters, **kwargs):
    """
    Updates the diabatic density matrix for FSSH. Accounts for both stochastic and
    deterministic FSSH modes.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    state.dm_adb_branch_flat : ndarray
        Flattened branch density matrices.
    state.dm_db : ndarray
        Diabatic density matrix.
    """
    dm_adb_branch = np.einsum(
        "ti,tj->tij",
        state.wf_adb,
        np.conj(state.wf_adb),
        optimize="greedy",
    )
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    batch_size = sim.settings.batch_size // num_branches
    num_quantum_states = sim.model.constants.num_quantum_states
    for nt, _ in enumerate(dm_adb_branch):
        np.einsum("jj->j", dm_adb_branch[nt])[...] = state.act_surf[nt]
    if sim.algorithm.settings.fssh_deterministic:
        dm_adb_branch = (
            np.einsum(
                "tbbb->tb",
                state.dm_adb_0.reshape(
                    (batch_size, num_branches, num_quantum_states, num_quantum_states)
                ),
            ).flatten()[:, np.newaxis, np.newaxis]
            * dm_adb_branch
        )
    else:
        dm_adb_branch = dm_adb_branch / num_branches
    state.dm_adb_branch_flat = dm_adb_branch.reshape(
        (
            batch_size * num_branches,
            num_quantum_states,
            num_quantum_states,
        )
    )
    state.dm_db_branch = functions.transform_mat(
        state.dm_adb_branch_flat, state.eigvecs, adb_to_db=True
    )
    state.dm_db = num_branches * np.sum(
        state.dm_db_branch.reshape(
            (
                batch_size,
                num_branches,
                num_quantum_states,
                num_quantum_states,
            )
        ),
        axis=-3,
    )
    return state, parameters

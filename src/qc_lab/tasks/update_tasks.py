"""
This module contains tasks that update the state and parameters objects
during propagation.
"""

import logging
import numpy as np
from qc_lab import functions
from qc_lab.numerical_constants import SMALL

logger = logging.getLogger(__name__)


def update_t(algorithm, sim, parameters, state, **kwargs):
    """
    Update the time in the state object with the time index in each trajectory
    multiplied by the update timestep.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.t`` : float
        Time of each trajectory.
    """
    batch_size = len(parameters.seed)
    state.t = np.ones(batch_size) * sim.t_ind * sim.settings.dt_update
    return parameters, state


def update_dh_c_dzc_finite_differences(algorithm, sim, parameters, state, **kwargs):
    """
    Calculate the gradient of the classical Hamiltonian using finite differences.

    Required Constants
    ------------------
    ``dh_c_dzc_finite_difference_delta`` : float, optional, default: 1e-6
        Finite-difference step size.

    Keyword Arguments
    -----------------
    z : ndarray
        Classical coordinates at which to evaluate the gradient.
    name : str, optional, default: "dh_c_dzc"
        Name under which to store the finite-difference gradient in the state object.

    Variable Modifications
    -------------------
    ``state.{name}`` : ndarray
        Gradient of the classical Hamiltonian.
    """
    z = kwargs["z"]
    name = kwargs.get("name", "dh_c_dzc")
    delta_z = sim.model.constants.get("dh_c_dzc_finite_difference_delta", 1e-6)
    batch_size = len(parameters.seed)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    offset_z_re = (
        z[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_im = (
        z[:, np.newaxis, :]
        + 1j * np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_c, _ = sim.model.get("h_c")
    h_c_0 = h_c(sim.model, parameters, z=z, batch_size=len(z))
    h_c_offset_re = h_c(
        sim.model,
        parameters,
        z=offset_z_re,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(batch_size, num_classical_coordinates)
    h_c_offset_im = h_c(
        sim.model,
        parameters,
        z=offset_z_im,
        batch_size=batch_size * num_classical_coordinates,
    ).reshape(batch_size, num_classical_coordinates)
    diff_re = (h_c_offset_re - h_c_0[:, np.newaxis]) / delta_z
    diff_im = (h_c_offset_im - h_c_0[:, np.newaxis]) / delta_z
    dh_c_dzc = 0.5 * (diff_re + 1j * diff_im)
    setattr(state, name, dh_c_dzc)
    return parameters, state


def update_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the classical Hamiltonian w.r.t. the conjugate classical
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
    ``state.classical_forces`` : ndarray
            Gradient of the classical Hamiltonian.
    """
    z = getattr(state, kwargs["z"])
    dh_c_dzc, has_dh_c_dzc = sim.model.get("dh_c_dzc")
    if has_dh_c_dzc:
        state.classical_forces = dh_c_dzc(sim.model, parameters, z=z)
        return parameters, state
    return update_dh_c_dzc_finite_differences(
        algorithm, sim, parameters, state, name="classical_forces", z=z
    )


def update_dh_qc_dzc_finite_differences(algorithm, sim, parameters, state, **kwargs):
    """
    Calculate the gradient of the quantum-classical Hamiltonian using finite
    differences.

    Required Constants
    ------------------
    ``dh_qc_dzc_finite_difference_delta`` : float, optional, default: 1e-6
        Finite-difference step size.

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in the state object.

    Variable Modifications
    -------------------
    ``state.dh_qc_dzc`` : tuple
        Gradient of the quantum-classical Hamiltonian.
    """
    z = getattr(state, kwargs["z"])
    batch_size = len(z)
    delta_z = sim.model.constants.get("dh_qc_dzc_finite_difference_delta", 1e-6)
    num_classical_coordinates = sim.model.constants.num_classical_coordinates
    num_quantum_states = sim.model.constants.num_quantum_states
    offset_z_re = (
        z[:, np.newaxis, :]
        + np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    offset_z_im = (
        z[:, np.newaxis, :]
        + 1j * np.identity(num_classical_coordinates)[np.newaxis, :, :] * delta_z
    ).reshape(
        (
            batch_size * num_classical_coordinates,
            num_classical_coordinates,
        )
    )
    h_qc, _ = sim.model.get("h_qc")
    h_qc_0 = h_qc(sim.model, parameters, z=z)
    h_qc_offset_re = h_qc(sim.model, parameters, z=offset_z_re).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    h_qc_offset_im = h_qc(
        sim.model,
        parameters,
        z=offset_z_im,
    ).reshape(
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    diff_re = (h_qc_offset_re - h_qc_0[:, np.newaxis, :, :]) / delta_z
    diff_im = (h_qc_offset_im - h_qc_0[:, np.newaxis, :, :]) / delta_z
    dh_qc_dzc = 0.5 * (diff_re + 1j * diff_im)
    inds = np.where(dh_qc_dzc != 0)
    mels = dh_qc_dzc[inds]
    shape = np.shape(dh_qc_dzc)
    state.dh_qc_dzc = (inds, mels, shape)
    return parameters, state


def update_dh_qc_dzc(algorithm, sim, parameters, state, **kwargs):
    """
    Update the gradient of the quantum-classical Hamiltonian w.r.t. the conjugate
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
    ``state.dh_qc_dzc`` : tuple
        Gradient of the quantum-classical Hamiltonian.
    """
    z = getattr(state, kwargs["z"])
    if state.dh_qc_dzc is None or sim.model.update_dh_qc_dzc:
        # If dh_qc_dzc has not been claculated yet, or if the
        # model requires it to be updated, calculate it.
        dh_qc_dzc, has_dh_qc_dzc = sim.model.get("dh_qc_dzc")
        if has_dh_qc_dzc:
            state.dh_qc_dzc = dh_qc_dzc(sim.model, parameters, z=z)
            return parameters, state
        return update_dh_qc_dzc_finite_differences(
            algorithm, sim, parameters, state, z=kwargs["z"]
        )
    # If dh_qc_dzc has already been calculated and does not need to be updated,
    # return the existing parameters and state objects.
    return parameters, state


def update_quantum_classical_forces(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum-classical forces w.r.t. the state defined by wf.

    If the model has a gauge_field_force ingredient, this term will be added
    to the quantum-classical forces.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.
    wf : str
        Name of the wavefunction in the state object.

    Variable Modifications
    -------------------
    ``state.dh_qc_dzc`` : tuple
        Gradient of the quantum-classical Hamiltonian.
    ``state.quantum_classical_forces`` : ndarray
        Quantum-classical forces.
    """
    z = getattr(state, kwargs["z"])
    wf = getattr(state, kwargs["wf"])
    use_gauge_field_force = kwargs.get("use_gauge_field_force", False)
    # Update the gradient of h_qc.
    parameters, state = update_dh_qc_dzc(
        algorithm, sim, parameters, state, z=kwargs["z"]
    )
    # inds, mels, shape = state.dh_qc_dzc
    # Calculate the expectation value w.r.t. the wavefunction.
    if state.quantum_classical_forces is None:
        state.quantum_classical_forces = np.zeros(np.shape(z), dtype=complex)
    state.quantum_classical_forces = functions.calc_sparse_inner_product(
        *state.dh_qc_dzc, wf.conj(), wf, out=state.quantum_classical_forces.flatten()
    ).reshape(np.shape(z))
    # Add the gauge field force if it exists and is requested.
    gauge_field_force, has_gauge_field_force = sim.model.get("gauge_field_force")
    if has_gauge_field_force and use_gauge_field_force:
        state.quantum_classical_forces += gauge_field_force(parameters, z=z, wf=wf)
    return parameters, state


def diagonalize_matrix(algorithm, sim, parameters, state, **kwargs):
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
    ``state.{eigvals}`` : ndarray
        Eigenvalues of the matrix.
    ``state.{eigvecs}`` : ndarray
        Eigenvectors of the matrix.
    """
    matrix = getattr(state, kwargs["matrix"])
    eigvals, eigvecs = np.linalg.eigh(matrix)
    setattr(state, kwargs["eigvals"], eigvals)
    setattr(state, kwargs["eigvecs"], eigvecs)
    return parameters, state


def gauge_fix_eigs(algorithm, sim, parameters, state, **kwargs):
    """
    Fixes the gauge of the eigenvectors as specified by the gauge_fixing parameter.

    if gauge_fixing == "sign_overlap":
        The sign of the eigenvector is changed so the real part of its
        overlap with the previous eigenvector is positive.

    if gauge_fixing == "phase_overlap":
        The phase of the eigenvector is determined from its overlap
        with the previous eigenvector and used to maximize the real
        part of the overlap.

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
    output_eigvecs_name : str
        Name of the output gauge-fixed eigenvectors in the state object.
    z : str
        Name of classical coordinates in the state object.
    gauge_fixing : str, optional, default: sim.algorithm.settings.gauge_fixing
        Gauge-fixing method to use.

    Variable Modifications
    -------------------
    - ``state.{output_eigvecs_name}``: gauge-fixed eigenvectors.
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
    if gauge_fixing_value >= 1:
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        phase = np.exp(-1j * np.angle(overlap))
        eigvecs *= phase[:, None, :]
    if gauge_fixing_value >= 2:
        parameters, state = update_dh_qc_dzc(
            algorithm, sim, parameters, state, z=kwargs["z"]
        )
        der_couple_q_phase, _ = functions.analytic_der_couple_phase(
            state.dh_qc_dzc,
            eigvals,
            eigvecs,
            sim.model.constants.classical_coordinate_mass,
            sim.model.constants.classical_coordinate_weight,
        )
        eigvecs *= np.conj(der_couple_q_phase)[:, None, :]
    if gauge_fixing_value >= 0:
        overlap = np.sum(np.conj(eigvecs_previous) * eigvecs, axis=-2)
        signs = np.sign(np.real(overlap))
        eigvecs *= signs[:, None, :]

    if gauge_fixing_value == 2:
        der_couple_q_phase_new, der_couple_p_phase_new = (
            functions.analytic_der_couple_phase(
                state.dh_qc_dzc,
                eigvals,
                eigvecs,
                sim.model.constants.classical_coordinate_mass,
                sim.model.constants.classical_coordinate_weight,
            )
        )
        if (
            np.sum(
                np.abs(np.imag(der_couple_q_phase_new)) ** 2
                + np.abs(np.imag(der_couple_p_phase_new)) ** 2
            )
            > SMALL
        ):
            logger.error(
                "Phase error encountered when fixing gauge analytically."
                + str(
                    np.sum(
                        np.abs(np.imag(der_couple_q_phase_new)) ** 2
                        + np.abs(np.imag(der_couple_p_phase_new)) ** 2
                    )
                )
            )
    setattr(state, kwargs["output_eigvecs_name"], eigvecs)
    return parameters, state


def basis_transform_vec(algorithm, sim, parameters, state, **kwargs):
    """
    Transforms a vector "input_name" to a new basis defined by "basis_name".

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    input_name : str
        Name of the vector to transform in the state object.
    basis_name : str
        Name of the basis to transform to in the state object.
        Assumed to be column vectors corresponding to adiabatic
        states.
    output_name : str
        Name of the output vector in the state object.
    adb_to_db : bool, optional, default: False
        If True, transforms from adiabatic to diabatic. If False, transforms from
        adiabatic to diabatic.

    Variable Modifications
    -------------------
    - ``state.{output_name}``: vector expressed in the new basis.
    """
    input_vec = getattr(state, kwargs["input_name"])
    basis = getattr(state, kwargs["basis_name"])
    adb_to_db = kwargs["adb_to_db"]
    setattr(
        state,
        kwargs["output_name"],
        functions.transform_vec(input_vec, basis, adb_to_db=adb_to_db),
    )
    return parameters, state


def update_act_surf_wf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the wavefunction corresponding to the active surface.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.act_surf_wf`` : ndarray
        Wavefunction of the active surface.
    """
    num_trajs = sim.settings.batch_size
    act_surf_wf = state.eigvecs[
        np.arange(num_trajs, dtype=int),
        :,
        state.act_surf_ind.flatten().astype(int),
    ]
    state.act_surf_wf = act_surf_wf
    return parameters, state


def update_wf_db_eigs(algorithm, sim, parameters, state, **kwargs):
    """
    Evolve the diabatic wavefunction using the electronic eigenbasis.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf_db_name : str
        Name of the diabatic wavefunction in the state object.
    eigvals_name : str
        Name of the eigenvalues in the state object.
    eigvecs_name : str
        Name of the eigenvectors in the state object.

    Variable Modifications
    -------------------
    ``state.{wf_db_name}`` : ndarray
        Updated diabatic wavefunction.
    """
    eigvals = getattr(state, kwargs["eigvals_name"])
    eigvecs = getattr(state, kwargs["eigvecs_name"])
    wf_db = getattr(state, kwargs["wf_db_name"])
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
        kwargs["wf_db_name"],
        functions.multiply_matrix_vector(prop_db, wf_db),
    )
    return parameters, state


def update_wf_db_rk4(algorithm, sim, parameters, state, **kwargs):
    """
    Update the wavefunction using the 4th-order Runge-Kutta method.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.wf_db`` : ndarray
        Diabatic wavefunction.
    """
    dt_update = sim.settings.dt_update
    wf_db = state.wf_db
    h_quantum = state.h_quantum
    state.wf_db = functions.wf_db_rk4(h_quantum, wf_db, dt_update)
    return parameters, state


def update_hop_probs_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Calculate the hopping probabilities for FSSH.

    :math:`P_{a\\rightarrow b} = -2\\Re((C_{b}/C_{a})\\langle a(t)| b(t-dt)\\rangle)`

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.hop_prob`` : ndarray
        Hopping probabilities between the active surface and all other surfaces.
    """
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    # Calculates < act_surf(t) | b(t-dt) >
    prod = functions.multiply_matrix_vector(
        np.swapaxes(state.eigvecs_previous, -1, -2),
        np.conj(
            state.eigvecs[
                np.arange(num_trajs * num_branches, dtype=int), :, act_surf_ind_flat
            ]
        ),
    )
    # Calculates -2*Re( (C_b / C_act_surf) * < act_surf(t) | b(t-dt) > )
    hop_prob = -2.0 * np.real(
        prod
        * state.wf_adb
        / state.wf_adb[np.arange(num_trajs * num_branches), act_surf_ind_flat][
            :, np.newaxis
        ]
    )
    # Sets hopping probabilities to 0 at the active surface.
    hop_prob[np.arange(num_branches * num_trajs), act_surf_ind_flat] *= 0.0
    state.hop_prob = hop_prob

    return parameters, state


def update_hop_inds_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Determine which trajectories hop based on the hopping probabilities and which state
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
    ``state.hop_ind`` : ndarray
        Indices of trajectories that hop.
    ``state.hop_dest`` : ndarray
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
        np.nan_to_num(hop_prob, nan=0, posinf=100e100, neginf=-100e100, copy=False), axis=1
    )
    rand_branch = (rand[:, np.newaxis] * np.ones((num_trajs, num_branches))).flatten()
    hop_ind = np.where(
        np.sum((cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1) > 0
    )[
        0
    ]  # trajectory indices that hop
    # destination indices of the hops in each hoping trajectory
    hop_dest = np.argmax(
        (cumulative_probs > rand_branch[:, np.newaxis]).astype(int), axis=1
    )[hop_ind]
    state.hop_ind = hop_ind
    state.hop_dest = hop_dest
    return parameters, state


def _update_hop_inds_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Determine which trajectories hop based on the hopping probabilities and which state
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
    ``state.hop_ind`` : ndarray
        Indices of trajectories that hop.
    ``state.hop_dest`` : ndarray
        Destination surface for each hop.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1

    hop_prob = state.hop_prob

    cumulative = np.cumsum(hop_prob, axis=1)

    # Repeat the per-trajectory random once per branch.
    rand = state.hopping_probs_rand_vals[:, sim.t_ind]
    rand_branch = np.repeat(rand, num_branches)

    # # A hop happens iff final cumulative prob exceeds the random

    hop_check = cumulative > rand_branch[:, None]

    hop_mask = (
        np.sum(hop_check.astype(int), axis=1) > 0
    )  # cumulative[:, -1] > rand_branch                      # (rows,)

    # First destination where cumulative > rand (compute once)
    # Note: creates a single boolean temp; much cheaper than doing it twice.
    first_idx = np.argmax(hop_check, axis=1)

    hop_ind = np.nonzero(hop_mask)[0]
    hop_dest = first_idx[hop_mask]

    state.hop_ind = hop_ind
    state.hop_dest = hop_dest
    return parameters, state


def update_z_shift_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Determine if a hop occurs and calculates the shift in the z-coordinate
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
    ``state.hop_successful_traj`` : bool
        Flag indicating if the hop was successful.
    ``state.z_shift_traj`` : ndarray
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
    return parameters, state


def update_hop_vals_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Executes the hopping function for the hopping trajectories.

    Stores the rescaled coordinates in ``state.z_rescaled`` and the a Boolean registering
    if the hop was successful in ``state.hop_successful``.

    If the model has a rescaling_direction_fssh ingredient, it will be used to
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
    ``state.z_shift`` : ndarray
        Shift in coordinates for each hopping trajectory.
    ``state.hop_successful`` : ndarray
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
    z = np.copy(state.z)
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)
    hop_traj_ind = 0
    for traj_ind in hop_ind:
        final_state_ind = hop_dest[hop_traj_ind]
        init_state_ind = act_surf_ind_flat[traj_ind]
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
                eigval_diff,
                eigvec_init_state,
                eigvec_final_state,
                dh_qc_dzc_traj,
                sim.model.constants.classical_coordinate_mass,
                sim.model.constants.classical_coordinate_weight,
            )
        parameters, state = update_z_shift_fssh(
            algorithm,
            sim,
            parameters,
            state,
            z=z[traj_ind],
            delta_z=delta_z,
            eigval_diff=eigval_diff,
        )
        state.hop_successful[hop_traj_ind] = state.hop_successful_traj
        state.z_shift[hop_traj_ind] = state.z_shift_traj
        hop_traj_ind += 1
    return parameters, state


def update_z_hop_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Apply coordinate changes after successful hops.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.z`` : ndarray
        Classical coordinates.
    """
    state.z[state.hop_ind] += state.z_shift
    return parameters, state


def update_act_surf_hop_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the active surface, active surface index, and active surface wavefunction
    following a hop in FSSH.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.act_surf_ind`` : ndarray
        Active surface indices.
    ``state.act_surf`` : ndarray
        Active surface wavefunctions.
    """
    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
    else:
        num_branches = 1
    num_trajs = sim.settings.batch_size // num_branches
    act_surf_flat = state.act_surf
    act_surf_ind = state.act_surf_ind
    act_surf_ind_flat = act_surf_ind.flatten().astype(int)

    idx = state.hop_ind[state.hop_successful]
    act_surf_ind_flat[idx] = state.hop_dest[state.hop_successful]
    act_surf_flat[idx] = np.zeros_like(act_surf_flat[idx])
    act_surf_flat[idx, state.hop_dest[state.hop_successful]] = 1
    state.act_surf_ind = np.copy(act_surf_ind_flat.reshape((num_trajs, num_branches)))
    state.act_surf = np.copy(act_surf_flat)
    return parameters, state


def update_h_quantum(algorithm, sim, parameters, state, **kwargs):
    """
    Update the Hamiltonian matrix of the quantum subsystem.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.


    Variable Modifications
    -------------------
    ``state.h_q`` : ndarray
        Quantum Hamiltonian matrix.
    ``state.h_qc`` : ndarray
        Quantum-classical coupling matrix.
    ``state.h_quantum`` : ndarray
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
    return parameters, state


def update_z_rk4_k1(algorithm, sim, parameters, state, **kwargs):
    """
    Compute the first RK4 intermediate for classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output_name : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    ``state.{output_name}``` : ndarray
        Output coordinates after half step.
    ``state.z_rk4_k1`` : ndarray
        First RK4 slope.
    """
    dt_update = sim.settings.dt_update
    z_k = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    out, k1 = functions.update_z_rk4_k123_sum(
        z_k, state.classical_forces, state.quantum_classical_forces, 0.5 * dt_update
    )
    setattr(state, output_name, out)
    state.z_rk4_k1 = k1
    return parameters, state


def update_z_rk4_k2(algorithm, sim, parameters, state, **kwargs):
    """
    Compute the second RK4 intermediate for classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output_name : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    ``state.{output_name}``` : ndarray
        Output coordinates after half step.
    ``state.z_rk4_k2`` : ndarray
        Second RK4 slope.
    """
    dt_update = sim.settings.dt_update
    z_k = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    out, k2 = functions.update_z_rk4_k123_sum(
        z_k, state.classical_forces, state.quantum_classical_forces, 0.5 * dt_update
    )
    setattr(state, output_name, out)
    state.z_rk4_k2 = k2
    return parameters, state


def update_z_rk4_k3(algorithm, sim, parameters, state, **kwargs):
    """
    Compute the third RK4 intermediate for classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output_name : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    ``state.{output_name}``` : ndarray
        Output coordinates after a full step.
    ``state.z_rk4_k3`` : ndarray
        Third RK4 slope.
    """
    dt_update = sim.settings.dt_update
    z_k = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    out, k3 = functions.update_z_rk4_k123_sum(
        z_k, state.classical_forces, state.quantum_classical_forces, dt_update
    )
    setattr(state, output_name, out)
    state.z_rk4_k3 = k3
    return parameters, state


def update_z_rk4_k4(algorithm, sim, parameters, state, **kwargs):
    """
    Compute the final RK4 update for classical coordinates.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of input coordinates in state object.
    output_name : str
        Name of the output coordinates in the state object.

    Variable Modifications
    -------------------
    ``state.{output_name}`` : ndarray
        Output coordinates after a full step.
    """
    dt_update = sim.settings.dt_update
    z_0 = getattr(state, kwargs["z"])
    output_name = kwargs["output_name"]
    out = functions.update_z_rk4_k4_sum(
        z_0,
        state.z_rk4_k1,
        state.z_rk4_k2,
        state.z_rk4_k3,
        state.classical_forces,
        state.quantum_classical_forces,
        dt_update,
    )
    setattr(state, output_name, out)
    return parameters, state


def update_dm_db_mf(algorithm, sim, parameters, state, **kwargs):
    """
    Update the diabatic density matrix based on the wavefunction.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    - ``state.dm_db``: diabatic density matrix.
    """
    wf_db = state.wf_db
    state.dm_db = np.einsum("ti,tj->tij", wf_db, np.conj(wf_db), optimize="greedy")
    return parameters, state


def update_classical_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    z : str
        Name of classical coordinates in state object.

    Variable Modifications
    -------------------
    ``state.classical_energy`` : ndarray
        Energy of the classical subsystem.
    """
    z = getattr(state, kwargs["z"])
    h_c, _ = sim.model.get("h_c")
    state.classical_energy = np.real(h_c(sim.model, parameters, z=z, batch_size=len(z)))
    return parameters, state


def update_classical_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the classical energy for FSSH simulations. If deterministic, the energy in
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
    ``state.classical_energy`` : ndarray
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
    return parameters, state


def update_quantum_energy(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t. the wavefunction specified by wf by taking
    the expectation value of ``state.h_quantum``.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf : str
        Name of the wavefunction in the state object.

    Variable Modifications
    -------------------
    ``state.quantum_energy`` : ndarray
        Quantum energy.
    """
    wf = getattr(state, kwargs["wf"])
    state.quantum_energy = np.real(
        np.einsum("ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy")
    )
    return parameters, state


def update_quantum_energy_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the quantum energy w.r.t. the wavefunction specified by wf.
    Accounts for both stochastic and deterministic FSSH modes.
    Typically, the wavefunction used is that of the active surface.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    wf : str
        Name of the wavefunction in the state object.

    Variable Modifications
    -------------------
    ``state.quantum_energy`` : ndarray
        Quantum energy.
    """
    wf = getattr(state, kwargs["wf"])

    if sim.algorithm.settings.fssh_deterministic:
        num_branches = sim.model.constants.num_quantum_states
        batch_size = sim.settings.batch_size // num_branches
        num_states = sim.model.constants.num_quantum_states
        wf = wf * np.sqrt(
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
            "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
        )
    else:
        state.quantum_energy = np.einsum(
            "ti,tij,tj->t", np.conj(wf), state.h_quantum, wf, optimize="greedy"
        )
        state.quantum_energy = state.quantum_energy
    state.quantum_energy = np.real(state.quantum_energy)
    return parameters, state


def update_dm_db_fssh(algorithm, sim, parameters, state, **kwargs):
    """
    Update the diabatic density matrix for FSSH. Accounts for both stochastic and
    deterministic FSSH modes.

    Required Constants
    ------------------
    None

    Keyword Arguments
    -----------------
    None

    Variable Modifications
    -------------------
    ``state.dm_adb_branch_flat`` : ndarray
        Flattened branch density matrices.
    ``state.dm_db`` : ndarray
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
    return parameters, state

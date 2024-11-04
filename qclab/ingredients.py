import numpy as np
import qclab.auxiliary as auxiliary
import copy
import pyscf
from pyscf import gto, scf, ci, grad, nac


############################################################
#                  MEAN-FIELD INGREDIENTS                  #
############################################################


def initialize_wf_db(state, model, params):
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]

    """
    state.wf_db = (np.zeros((params.batch_size, params.num_branches, model.num_states), dtype=complex)
                   + model.wf_db[np.newaxis, :])
    return state, model, params


def initialize_z_coord(state, model, params):
    state.z_coord = np.zeros((params.batch_size, params.num_branches,
                              model.num_classical_coordinates), dtype=complex)
    # load initial values of the z coordinate 
    for traj_n in range(params.batch_size):
        state.z_coord[traj_n, :, :] = model.init_classical(model, params.seeds[traj_n])
    return state, model, params


def update_h_quantum(state, model, params):
    state.h_quantum = np.zeros((params.batch_size, params.num_branches,
                                model.num_states, model.num_states), dtype=complex) \
                      + model.h_q(state, model, params) + model.h_qc(state, model, params, state.z_coord)
    return state, model, params


def update_quantum_force_wf_db(state, model, params):
    state.quantum_force = model.dh_qc_dzc(state, model, params, state.z_coord, state.wf_db, state.wf_db)
    return state, model, params


def update_z_coord_rk4(state, model, params):
    state.z_coord = auxiliary.rk4_c(state, model, params, state.z_coord, state.quantum_force, params.dt)
    return state, model, params


def update_wf_db_rk4(state, model, params):
    # evolve wf_db using an RK4 solver
    state.wf_db = auxiliary.rk4_q(state.h_quantum, state.wf_db, params.dt)
    return state, model, params


def update_dm_db_mf(state, model, params):
    state.dm_db = np.einsum('tbi,tbj->ij', state.wf_db, np.conj(state.wf_db), optimize='greedy')
    return state, model, params


def update_e_c(state, model, params):
    state.e_c = np.sum(model.h_c(state, model, params, state.z_coord))
    return state, model, params


def update_e_q_mf(state, model, params):
    state.e_q = np.einsum('tbj,tbji,tbi', np.conj(state.wf_db),
                          state.h_quantum, state.wf_db, optimize='greedy')
    return state, model, params


############################################################
#                     FSSH INGREDIENTS                    #
############################################################


def initialize_random_values(state, model, params):
    # initialize random numbers needed in each trajectory
    state.hopping_probs_rand_vals = np.zeros((params.batch_size, len(params.tdat)))
    state.stochastic_sh_rand_vals = np.zeros((params.batch_size, params.num_branches))
    for nt in range(params.batch_size):
        np.random.seed(params.seeds[nt])
        state.hopping_probs_rand_vals[nt, :] = np.random.rand(len(params.tdat))
        state.stochastic_sh_rand_vals[nt, :] = np.random.rand(params.num_branches)
    return state, model, params


def update_eigs(state, model, params):
    state.eigvals, state.eigvecs = np.linalg.eigh(state.h_quantum)
    return state, model, params


def analytic_gauge_fix_eigs(state, model, params):
    # compute initial eigenvalues and eigenvectors in each branch
    for traj_n in range(params.batch_size):
        for branch_n in range(params.num_branches):
            # compute initial gauge shift for real-valued derivative couplings
            der_couple_q_phase, der_couple_p_phase = (
                auxiliary.get_der_couple_phase(state, model, params, state.z_coord[traj_n, branch_n], state.eigvals[traj_n, branch_n], state.eigvecs[traj_n, branch_n]))
            # execute phase shift
            state.eigvecs[traj_n, branch_n] = np.copy(
                np.matmul(state.eigvecs[traj_n, branch_n], np.diag(np.conjugate(der_couple_q_phase))))
            # recalculate phases and check that they are zero
            der_couple_q_phase, der_couple_p_phase = (
                auxiliary.get_der_couple_phase(state, model, params, state.z_coord[traj_n, branch_n], state.eigvals[traj_n, branch_n], state.eigvecs[traj_n, branch_n]))
            if np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2) > 1e-10:
                # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
                # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
                print('Warning: phase init',
                      np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2))
    return state, model, params


def update_eigs_previous(state, model, params):
    state.eigvecs_previous = np.copy(state.eigvecs)
    return state, model, params


def initialize_wf_adb(state, model, params):
    # state.wf_adb = auxiliary.psi_db_to_adb_branch(state.wf_db, state.eigvecs)
    state.wf_adb = auxiliary.vec_db_to_adb(state.wf_db, state.eigvecs)
    return state, model, params


def initialize_active_surface(state, model, params):
    # Options for deterministic branch modelulation, num_branches==num_states
    if params.sh_deterministic:
        assert params.num_branches == model.num_states
        act_surf_ind_0 = np.zeros((params.batch_size, params.num_branches)) + np.arange(params.num_branches, dtype=int)[
                                                                           np.newaxis, :]
    else:
        # determine initial active surfaces
        intervals = np.zeros((params.batch_size, model.num_states))
        for traj_n in range(params.batch_size):
            for state_n in range(model.num_states):
                intervals[traj_n, state_n] = np.real(np.sum((np.abs(state.wf_adb[traj_n, 0]) ** 2)[0:state_n + 1]))
        # initialize active surface index
        act_surf_ind_0 = np.zeros((params.batch_size, params.num_branches), dtype=int)
        for traj_n in range(params.batch_size):
            for branch_n in range(params.num_branches):
                act_surf_ind_0[traj_n, branch_n] = np.arange(model.num_states, dtype=int)[
                    intervals[traj_n] > state.stochastic_sh_rand_vals[traj_n, branch_n]][0]
            act_surf_ind_0[traj_n] = np.sort(act_surf_ind_0[traj_n])
    # initialize active surface and active surface index in each branch
    state.act_surf_ind_0 = act_surf_ind_0.astype(int)
    state.act_surf_ind = np.copy(state.act_surf_ind_0).astype(int)
    act_surf = np.zeros((params.batch_size, params.num_branches, model.num_states), dtype=int)
    for nt in range(params.batch_size):
        act_surf[nt][np.arange(params.num_branches, dtype=int), state.act_surf_ind[nt]] = 1
    state.act_surf = act_surf.reshape((params.batch_size, params.num_branches, model.num_states)).astype(int)
    return state, model, params


def update_quantum_force_act_surf(state, model, params):
    traj_ind = np.arange(params.batch_size, dtype=int)[:, np.newaxis] + np.zeros((params.batch_size, params.num_branches),
                                                                               dtype=int)
    branch_ind = np.arange(params.num_branches, dtype=int)[np.newaxis, :] + np.zeros(
        (params.batch_size, params.num_branches), dtype=int)
    state.quantum_force = model.dh_qc_dzc(state, model, params, state.z_coord, state.eigvecs[traj_ind, branch_ind, :, state.act_surf_ind],
                                          state.eigvecs[traj_ind, branch_ind, :, state.act_surf_ind])
    return state, model, params


def initialize_dm_adb_0_fssh(state, model, params):
    state.dm_adb_0 = np.zeros((params.batch_size, model.num_states, model.num_states), dtype=complex)

    for n in range(params.batch_size):
        # use the first branch, since all branches are identical at t=0
        state.dm_adb_0[n] = np.outer(np.conj(state.wf_adb[n, 0]), state.wf_adb[n, 0])
    return state, model, params


def update_wf_db_eigs(state, model, params):
    state.wf_db, state.wf_adb = auxiliary.evolve_wf_eigs(state.wf_db, state.eigvals, state.eigvecs, params.dt)
    return state, model, params


def gauge_fix_eigs(state, model, params):
    for nt in range(params.batch_size):  # TODO can this loop be eliminated? Make sign_adjust_branch not branch dependent?
        # adjust gauge of eigenvectors
        state.eigvecs[nt], _ = auxiliary.sign_adjust_branch(state, model, params, state.z_coord[nt], state.eigvecs[nt], 
                                                            state.eigvecs_previous[nt], state.eigvals[nt])
    return state, model, params


def update_active_surface_fssh(state, model, params):
    ############################################################
    #                         HOPPING PROCEDURE                #
    ############################################################
    for nt in range(params.batch_size):
        rand = state.hopping_probs_rand_vals[nt, state.t_ind]
        for i in range(params.num_branches):
            # compute hopping probabilities
            prod = np.matmul(
                np.conjugate(state.eigvecs[nt, i][:, state.act_surf_ind[nt, i]]),state.eigvecs_previous[nt, i])

            hop_prob = -2 * np.real(prod * (state.wf_adb[nt, i] / state.wf_adb[nt, i][state.act_surf_ind[nt, i]]))
            hop_prob[state.act_surf_ind[nt, i]] = 0
            bin_edge = 0
            # hop if possible
            for k in range(len(hop_prob)):
                hop_prob[k] = auxiliary.nan_num(hop_prob[k])
                bin_edge = bin_edge + hop_prob[k]
                if rand < bin_edge:
                    # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                    evec_k = state.eigvecs[nt, i][:, state.act_surf_ind[nt, i]]
                    evec_j = state.eigvecs[nt, i][:, k]
                    eval_k = state.eigvals[nt, i][state.act_surf_ind[nt, i]]
                    eval_j = state.eigvals[nt, i][k]
                    ev_diff = eval_j - eval_k
                    # dkj_q is wrt q dkj_p is wrt p.
                    #dkj_z, dkj_zc = auxiliary.get_der_couple(state, state.z_coord, evec_k, evec_j, ev_diff)
                    dkj_z = model.dh_qc_dz(state, model, params, state.z_coord, evec_k, evec_j) / (ev_diff)
                    dkj_zc = model.dh_qc_dzc(state, model, params, state.z_coord, evec_k, evec_j) / (ev_diff)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(model.pq_weight * model.mass / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * model.pq_weight * model.mass)) * 1.0j * (dkj_z - dkj_zc)
                    max_pos_q = np.argmax(np.abs(dkj_q))
                    max_pos_p = np.argmax(np.abs(dkj_p))
                    if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        raise Exception('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!')
                    if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                        raise Exception('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!')
                    delta_z = dkj_zc
                    state.z_coord[nt, i], hopped = model.hop(state, model, params, state.z_coord[nt, i], delta_z, ev_diff)
                    if hopped:  # adjust active surfaces if a hop has occurred
                        state.act_surf_ind[nt, i] = k
                        state.act_surf[nt, i] = np.zeros_like(state.act_surf[nt, i])
                        state.act_surf[nt, i][state.act_surf_ind[nt, i]] = 1
                    break
    return state, model, params


def update_dm_db_fssh(state, model, params):
    state.dm_adb = np.einsum('tbi,tbj->tbij', state.wf_adb, np.conj(state.wf_adb), optimize='greedy')
    for nt in range(params.batch_size):
        np.einsum('...jj->...j', state.dm_adb[nt])[...] = state.act_surf[nt]
    if params.sh_deterministic:
        state.dm_adb = np.einsum('tbb->tb', state.dm_adb_0, optimize='greedy')[:, :, np.newaxis, np.newaxis] * state.dm_adb
    else:
        state.dm_adb = state.dm_adb / params.num_branches

    state.dm_db = np.sum(auxiliary.mat_adb_to_db(state.dm_adb, state.eigvecs), axis=(0, 1))
    return state, model, params


def update_e_q_fssh(state, model, params):
    traj_ind = (np.arange(params.batch_size, dtype=int)[:, np.newaxis] + \
                np.zeros((params.batch_size, params.num_branches), dtype=int))
    branch_ind = (np.arange(params.num_branches, dtype=int)[np.newaxis, :] + \
                  np.zeros((params.batch_size, params.num_branches), dtype=int))
    state.e_q = np.sum(state.eigvals[traj_ind, branch_ind, state.act_surf_ind])
    return state, model, params


############################################################
#                     CFSSH INGREDIENTS                    #
############################################################


def update_branch_pair_eigs(state, model, params):
    state.eigvals_branch_pair = np.zeros((params.batch_size, params.num_branches, 
                                          params.num_branches, model.num_states), dtype=float)
    state.eigvecs_branch_pair = np.zeros((params.batch_size, params.num_branches, params.num_branches, 
                                          model.num_states, model.num_states), dtype=complex)
    h_q = model.h_q(state, model, params)
    for nt in range(params.batch_size):
        for i in range(params.num_branches):
            for j in range(i, params.num_branches):
                z_coord_nt_ij = (state.z_coord[nt, i] + state.z_coord[nt, j]) / 2
                state.eigvals_branch_pair[nt, i, j], state.eigvecs_branch_pair[nt, i, j] = np.linalg.eigh(h_q + model.h_qc(state, model, params, z_coord_nt_ij))
                if i != j:
                    state.eigvals_branch_pair[nt, j, i], state.eigvecs_branch_pair[nt, j, i] = state.eigvals_branch_pair[nt, i, j], state.eigvecs_branch_pair[nt, i, j]
    return state, model, params


def analytic_gauge_fix_branch_pair_eigs(state, model, params):
    # compute initial eigenvalues and eigenvectors in each branch
    for n in range(params.batch_size):
        for b in range(params.num_branches):
            for d in range(params.num_branches):
                # compute initial gauge shift for real-valued derivative couplings
                z_coord_ij = (state.z_coord[n,b] + state.z_coord[n,d])/2
                der_couple_q_phase, der_couple_p_phase = (auxiliary.get_der_couple_phase(state, model, params, z_coord_ij, state.eigvals_branch_pair[n, b, d], state.eigvecs_branch_pair[n, b, d]))
                # execute phase shift
                state.eigvecs_branch_pair[n, b, d] = np.copy(np.matmul(state.eigvecs_branch_pair[n, b, d], np.diag(np.conjugate(der_couple_q_phase))))
                # recalculate phases and check that they are zero
                der_couple_q_phase, der_couple_p_phase = (
                    auxiliary.get_der_couple_phase(state, model, params, z_coord_ij, state.eigvals_branch_pair[n, b, d], state.eigvecs_branch_pair[n, b, d]))
                if np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2) > 1e-10:
                    # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
                    # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
                    print('Warning: phase init',
                        np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2))
    return state, model, params


def update_branch_pair_eigs_previous(state, model, params):
    state.eigvecs_branch_pair_previous = np.copy(state.eigvecs_branch_pair)
    return state, model, params


def initialize_wf_adb_delta(state, model, params):
    # initialize wavefunction as a delta function in each branch
    wf_adb_delta = np.zeros((params.batch_size, params.num_branches, model.num_states), dtype=complex)
    for nt in range(params.batch_size):
        wf_adb_delta[nt][np.arange(params.num_branches, dtype=int), state.act_surf_ind_0[nt]] = 1.0 + 0.j
    # transform to diabatic basis
    state.wf_adb_delta = wf_adb_delta
    state.wf_db_delta = auxiliary.vec_adb_to_db(wf_adb_delta, state.eigvecs)
    return state, model, params


def initialize_branch_phase(state, model, params):
    state.branch_phase = np.zeros((params.batch_size, params.num_branches))
    return state, model, params


def update_wf_db_delta_eigs(state, model, params):
    state.wf_db_delta, state.wf_adb_delta = auxiliary.evolve_wf_eigs(state.wf_db_delta, state.eigvals, state.eigvecs, params.dt)
    return state, model, params


def update_branch_phase(state, model, params):
    traj_ind = np.arange(params.batch_size, dtype=int)[:, np.newaxis] + \
        np.zeros((params.batch_size, params.num_branches), dtype=int)
    branch_ind = np.arange(params.num_branches, dtype=int)[np.newaxis, :] + \
        np.zeros((params.batch_size, params.num_branches), dtype=int)
    state.branch_phase = state.branch_phase + params.dt * state.eigvals[traj_ind, branch_ind, state.act_surf_ind_0]
    return state, model, params


def update_active_surface_cfssh(state, model, params):
    ############################################################
    #                         HOPPING PROCEDURE                #
    ############################################################
    for nt in range(params.batch_size):
        rand = state.hopping_probs_rand_vals[nt, state.t_ind]
        for i in range(params.num_branches):
            # compute hopping probabilities
            prod = np.matmul(np.conjugate(state.eigvecs[nt, i][:, state.act_surf_ind[nt, i]]),
                             state.eigvecs_previous[nt, i])
            hop_prob = -2 * np.real(prod * (state.wf_adb_delta[nt, i] / state.wf_adb_delta[nt, i][state.act_surf_ind[nt, i]]))
            hop_prob[state.act_surf_ind[nt, i]] = 0
            bin_edge = 0
            # hop if possible
            for k in range(len(hop_prob)):
                hop_prob[k] = auxiliary.nan_num(hop_prob[k])
                bin_edge = bin_edge + hop_prob[k]
                if rand < bin_edge:
                    # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                    evec_k = state.eigvecs[nt, i][:, state.act_surf_ind[nt, i]]
                    evec_j = state.eigvecs[nt, i][:, k]
                    eval_k = state.eigvals[nt, i][state.act_surf_ind[nt, i]]
                    eval_j = state.eigvals[nt, i][k]
                    ev_diff = eval_j - eval_k
                    # dkj_q is wrt q dkj_p is wrt p.
                    #dkj_z, dkj_zc = auxiliary.get_der_couple(evec_k, evec_j, ev_diff, state.z_coord[nt, i], model)
                    dkj_z = model.dh_qc_dz(state, model, params, state.z_coord, evec_k, evec_j) / (ev_diff)
                    dkj_zc = model.dh_qc_dzc(state, model, params, state.z_coord, evec_k, evec_j) / (ev_diff)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(model.pq_weight * model.mass / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * model.pq_weight * model.mass)) * 1.0j * (dkj_z - dkj_zc)
                    max_pos_q = np.argmax(np.abs(dkj_q))
                    max_pos_p = np.argmax(np.abs(dkj_p))
                    if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        raise Exception('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!')
                    if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                        raise Exception('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!')
                    delta_z = dkj_zc
                    state.z_coord[nt, i], hopped = model.hop(state, model, params, state.z_coord[nt, i], delta_z, ev_diff)
                    if hopped:  # adjust active surfaces if a hop has occured
                        state.act_surf_ind[nt, i] = k
                        state.act_surf[nt, i] = np.zeros_like(state.act_surf[nt, i])
                        state.act_surf[nt, i][state.act_surf_ind[nt, i]] = 1
                    break
    return state, model, params


def gauge_fix_branch_pair_eigs(state, model, params):
    # fix the gauge of each set of branches in each batch of trajectories
    for nt in range(params.batch_size):
        state.eigvals_branch_pair[nt], state.eigvecs_branch_pair[nt] = \
        auxiliary.sign_adjust_branch_pair_eigs(state, model, params, state.z_coord[nt], state.eigvecs_branch_pair[nt],
            state.eigvals_branch_pair[nt], state.eigvecs_branch_pair_previous[nt])
    return state, model, params


def update_classical_overlap(state, model, params):
    # calculate overlap matrix
    state.overlap = np.zeros((params.batch_size, params.num_branches, params.num_branches))
    for nt in range(params.batch_size):
        state.overlap[nt] = auxiliary.get_classical_overlap(state, model, params, state.z_coord[nt])
    return state, model, params


def update_dm_db_cfssh(state, model, params):
    # state.dm_adb_branch = np.einsum('ni,nj->nij', state.wf_adb, np.conj(state.wf_adb))
    # for nt in range(params.batch_size):
    #    np.einsum('...jj->...j', state.dm_adb_branch[nt * params.num_branches:(nt + 1) * params.num_branches])[
    #        ...] = state.act_surf[nt * params.num_branches:(nt + 1) * params.num_branches]
    state.dm_adb = np.zeros((params.batch_size, params.num_branches, model.num_states, model.num_states), dtype=complex)
    dm_adb_coh = np.zeros((params.batch_size, model.num_states, model.num_states), dtype=complex)
    dm_db_coh = np.zeros((params.batch_size, model.num_states, model.num_states), dtype=complex)
    for nt in range(params.batch_size):
        dm_adb_coh_ij = np.zeros((model.num_states, model.num_states), dtype=complex)
        branch_phase_nt = state.branch_phase[nt]
        act_surf_ind_nt = state.act_surf_ind[nt]
        act_surf_nt = state.act_surf[nt]
        act_surf_ind_0_nt = state.act_surf_ind_0[nt]
        #z_coord_nt = state.z_coord[nt]
        for i in range(params.num_branches):
            for j in range(i + 1, params.num_branches):
                a_i = act_surf_ind_nt[i]
                a_j = act_surf_ind_nt[j]
                a_i_0 = act_surf_ind_0_nt[i]
                a_j_0 = act_surf_ind_0_nt[j]
                if a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(state.dm_adb_0[nt][a_i, a_j]) > 1e-12:
                    if params.sh_deterministic:
                        prob_fac = 1
                    else:
                        prob_fac = 1 / (state.dm_adb_0[nt][a_i, a_i] * state.dm_adb_0[nt][a_j, a_j] * (
                                    params.num_branches - 1))
                    coh_ij_tmp = prob_fac * state.dm_adb_0[nt][a_i, a_j] * state.overlap[nt][i, j] * np.exp(
                        -1.0j * (branch_phase_nt[i] - branch_phase_nt[j]))
                    dm_adb_coh_ij[a_i, a_j] += coh_ij_tmp
                    dm_adb_coh_ij[a_j, a_i] += np.conj(coh_ij_tmp)
                    dm_db_coh_ij = coh_ij_tmp * np.outer(state.eigvecs_branch_pair[nt][i, j][:, a_i],
                                                         np.conj(state.eigvecs_branch_pair[nt][i, j][:, a_j])) + \
                                   np.conj(coh_ij_tmp) * np.outer(state.eigvecs_branch_pair[nt][i, j][:, a_j],
                                                                  np.conj(state.eigvecs_branch_pair[nt][i, j][:, a_i]))
                    dm_db_coh[nt] = dm_db_coh[nt] + dm_db_coh_ij
                    dm_adb_coh[nt] = dm_adb_coh[nt] + dm_adb_coh_ij
                    dm_adb_coh_ij = np.zeros((model.num_states, model.num_states), dtype=complex)

        if params.sh_deterministic:
            rho_diag = np.diag(state.dm_adb_0[nt]).reshape((-1, 1)) * act_surf_nt
            np.einsum('...jj->...j', state.dm_adb[nt], optimize='greedy')[...] = rho_diag
        else:
            # TODO check this stochastic density matrix construction
            for n in range(params.num_branches):
                state.dm_adb[n, state.act_surf_ind[n], state.act_surf_ind[n]] += 1 # This is probably wrong with the indices
    state.dm_db = auxiliary.mat_adb_to_db(state.dm_adb, state.eigvecs)
    if params.sh_deterministic:
        state.dm_db = (state.dm_db + (dm_db_coh[:, np.newaxis, :, :] / params.num_branches))
    else:
        state.dm_db = (state.dm_db + (dm_db_coh[:, np.newaxis, :, :] / params.num_branches)) / params.num_branches
    
    state.dm_db = np.sum(state.dm_db, axis=(0, 1))
    return state, model, params



############################################################
#              MANYBODY MEAN-FIELD INGREDIENTS             #
############################################################


def initialize_wf_db_mb(state, model, params):
    state.wf_db_MB = np.zeros((params.batch_size, params.num_branches, 
                               model.num_states, model.num_particles), dtype=complex) + model.wf_db_MB[..., :, :]
    return state, model, params



def update_quantum_force_wf_db_mbmf(state, model, params):
    state.quantum_force = np.zeros((params.batch_size, params.num_branches, model.num_classical_coordinates), dtype=complex)
    for n in range(model.num_particles):
        state.quantum_force += model.dh_qc_dzc(state, state.z_coord, state.wf_db_MB[..., n], state.wf_db_MB[..., n])
    return state, model, params


def update_wf_db_mb_rk4(state, model, params):
    # evolve wf_db using an RK4 solver
    for n in range(model.num_particles):
        state.wf_db_MB[..., n] = auxiliary.rk4_q(state.h_quantum, state.wf_db_MB[..., n], params.dt)
    return state, model, params


def update_e_q_mbmf(state, model, params):
    state.e_q = np.einsum('tbin,tbij,tbjn', np.conj(state.wf_db_MB), state.h_quantum, state.wf_db_MB, optimize='greedy')
    return state, model, params

def update_rdm1(state, model, params):
    state.rdm1 = np.einsum('tbin,tbjn->ij', state.wf_db_MB, np.conj(state.wf_db_MB), optimize='greedy')
    return state, model, params 

def update_rdm2(state, model, params):
    #rdm1 = np.einsum('tbin,tbjn->tbij', state.wf_db_MB, np.conj(state.wf_db_MB), optimize='greedy')
    #state.rdm2 = np.einsum('tbil,jk->ijlk',rdm1, np.identity(model.num_states), optimize='greedy') - \
    #    np.einsum('tbik,tbjl->ijlk', rdm1, rdm1, optimize='greedy')
    state.rdm2 = np.einsum('tbin,tbjm,tbkm,tbln->ijkl',np.conj(state.wf_db_MB), np.conj(state.wf_db_MB), state.wf_db_MB, state.wf_db_MB, optimize='greedy') - \
        np.einsum('tbin,tbjm,tbkn,tblm->ijkl',np.conj(state.wf_db_MB), np.conj(state.wf_db_MB), state.wf_db_MB, state.wf_db_MB, optimize='greedy')
    return state, model, params


    



############################################################
#              AB INITIO MEAN-FIELD DYNAMICS              #
############################################################


def initialize_wf_adb_ab_initio(state, model, params):
    state.wf_adb = (np.zeros((params.batch_size, params.num_branches, model.num_states), dtype=complex)
                   + model.wf_adb[np.newaxis, :])
    return state, model, params


def update_quantum_force_wf_adb(state, model, params):
    state.quantum_force = model.dh_qc_dzc(state, model, params, state.z_coord, state.wf_adb, state.wf_adb)
    return state, model, params 


def update_wf_adb_rk4(state, model, params):
    # evolve wf_db using an RK4 solver
    state.wf_adb = auxiliary.rk4_q(state.h_quantum, state.wf_adb, params.dt)
    print(np.sum(np.abs(state.h_quantum - np.einsum('tbij->tbji',np.conj(state.h_quantum)))))
    return state, model, params

def update_ab_initio_ham_prev(state, model, params):
    state.ab_initio_hams_posthf_prev = copy.copy(state.ab_initio_hams_posthf)
    state.ab_initio_hams_mf_prev = copy.copy(state.ab_initio_hams_mf)
    return state, model, params

def update_q_coord(state, model, params):
    state.q_coord = np.sum(np.real(state.z_coord + np.conj(state.z_coord))/np.sqrt(2*model.mass*model.pq_weight),axis=(0,1))
    #print(state.q_coord.reshape(model.num_atoms, 3))
    print(state.t_ind/len(params.tdat_n), np.sum(np.abs(np.sum(state.wf_adb, axis=(0,1)))**2))
    return state, model, params

def update_ab_initio_ham(state, model, params):
    ab_initio_hams_posthf = np.zeros((params.batch_size, params.num_branches), dtype=object)
    ab_initio_hams_mf = np.zeros((params.batch_size, params.num_branches), dtype=object)
    for traj in range(params.batch_size):
        for branch in range(params.num_branches):
            q_coord = np.real((1 / np.sqrt(2 * model.mass * model.pq_weight)) * (state.z_coord[traj, branch] + np.conj(state.z_coord[traj, branch])))
            q_coord = q_coord.reshape((model.num_atoms, 3))
            atom_coords = [['H', tuple(q_coord[n])] for n in range(model.num_atoms)]
            mol = pyscf.gto.M(atom = atom_coords, basis=model.basis)
            mol.verbose=0
            mf = pyscf.scf.RHF(mol).run()
            myci = pyscf.ci.CISD(mf)
            #myci = pyscf.fci.FCI(mf)
            myci.nroots = model.num_states
            myci.run()
            ab_initio_hams_posthf[traj][branch] = myci
            ab_initio_hams_mf[traj][branch] = mf


    state.ab_initio_hams_posthf = ab_initio_hams_posthf
    state.ab_initio_hams_mf = ab_initio_hams_mf
    return state, model, params 

def update_diag_eq(state, model, params):
    state.diag_eq = np.diag(np.sum(state.h_quantum, axis=(0,1)))
    return state, model, params


from functools import reduce
def sort_surfaces(state, model, params):
    for traj in range(params.batch_size):
        for branch in range(params.num_branches):
            myci = state.ab_initio_hams_posthf[traj][branch]
            myci_prev = state.ab_initio_hams_posthf_prev[traj][branch]
            #eris = myci.ao2mo()
            mf = state.ab_initio_hams_mf[traj][branch]
            mf_prev = state.ab_initio_hams_mf_prev[traj][branch]
            s12 = pyscf.gto.intor_cross('cint1e_ovlp_sph', mf.mol, mf_prev.mol)
            s12 = reduce(np.dot, (mf.mo_coeff.T, s12, mf_prev.mo_coeff))
            nmo = mf_prev.mo_energy.size 
            nocc = mf.mol.nelectron // 2
            if model.num_states > 1:
                overlap_mat = np.zeros((model.num_states, model.num_states), dtype=complex)
                for n in range(model.num_states):
                    for m in range(model.num_states):
                        #overlap_mat[m, n] = pyscf.ci.cisd.overlap(myci.ci[m], myci_prev.ci[n], nmo, nocc, s12)
                        overlap_mat[m, n] = pyscf.fci.addons.overlap(myci.ci[m], myci_prev.ci[n], myci.norb, myci.nelec)
              
                order = np.argmax(np.abs(overlap_mat), axis=1)
                #print(np.round(np.abs(overlap_mat)**2,1))
                #print(np.round(np.diag(out_mat[traj,branch]),2))
                #print(np.round(np.abs(overlap_mat)**2,1)[:,order][order,:])
                #out_mat[traj,branch] = out_mat[traj, branch][:,order][order,:]
                #print(np.round(np.diag(out_mat[traj,branch]),2))
                state.ab_initio_hams_posthf[traj][branch].ci = list(np.array(state.ab_initio_hams_posthf[traj][branch].ci)[order])
    return state, model, params




############################################################
#           MANYBODY SURFACE HOPPING INGREDIENTS          #
############################################################


def initialize_wf_adb_mb(state, model, params):
    state.wf_adb_MB = np.einsum('...ji->...ij',auxiliary.vec_db_to_adb(np.einsum('...ij->...ji',state.wf_db_MB,optimize='greedy'), state.eigvecs),optimize='greedy')
    return state, model, params 


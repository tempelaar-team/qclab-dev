import numpy as np
import qclab.auxiliary as auxiliary


############################################################
#                  MEAN-FIELD INGREDIENTS                  #
############################################################


def initialize_wf_db(state):
    state.wf_db = (np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_states), dtype=complex)
                   + state.model.wf_db[np.newaxis, :])
    return state


def initialize_z_coord(state):
    state.z_coord = np.zeros((state.model.batch_size, state.model.num_branches,
                              state.model.num_classical_coordinates), dtype=complex)
    # load initial values of the z coordinate 
    for traj_n in range(state.model.batch_size):
        state.z_coord[traj_n, :, :] = state.model.init_classical(state.model, state.model.seeds[traj_n])
    return state


def update_h_quantum(state):
    state.h_quantum = np.zeros((state.model.batch_size, state.model.num_branches,
                                state.model.num_states, state.model.num_states), dtype=complex) \
                      + state.model.h_q(state) + state.model.h_qc(state, state.z_coord)
    return state


def update_quantum_force_wf_db(state):
    state.quantum_force = state.model.dh_qc_dzc(state, state.z_coord, state.wf_db, state.wf_db)
    return state


def update_z_coord_rk4(state):
    state.z_coord = auxiliary.rk4_c(state, state.z_coord, state.quantum_force, state.model.dt)
    return state



def update_wf_db_rk4(state):
    # evolve wf_db using an RK4 solver
    state.wf_db = auxiliary.rk4_q(state.h_quantum, state.wf_db, state.model.dt)
    return state


def update_dm_db_mf(state):
    state.dm_db = np.einsum('tbi,tbj->ij', state.wf_db, np.conj(state.wf_db), optimize='greedy')
    return state


def update_e_c(state):
    state.e_c = np.sum(state.model.h_c(state, state.z_coord))
    return state


def update_e_q_mf(state):
    state.e_q = np.einsum('tbj,tbji,tbi', np.conj(state.wf_db),
                          state.h_quantum, state.wf_db, optimize='greedy')
    return state


############################################################
#                     FSSH INGREDIENTS                    #
############################################################


def initialize_random_values(state):
    # initialize random numbers needed in each trajectory
    state.hopping_probs_rand_vals = np.zeros((state.model.batch_size, len(state.model.tdat)))
    state.stochastic_sh_rand_vals = np.zeros((state.model.batch_size, state.model.num_branches))
    for nt in range(state.model.batch_size):
        np.random.seed(state.model.seeds[nt])
        state.hopping_probs_rand_vals[nt, :] = np.random.rand(len(state.model.tdat))
        state.stochastic_sh_rand_vals[nt, :] = np.random.rand(state.model.num_branches)
    return state


def update_eigs(state):
    state.eigvals, state.eigvecs = np.linalg.eigh(state.h_quantum)
    return state


def analytic_gauge_fix_eigs(state):
    # compute initial eigenvalues and eigenvectors in each branch
    for traj_n in range(state.model.batch_size):
        for branch_n in range(state.model.num_branches):
            # compute initial gauge shift for real-valued derivative couplings
            der_couple_q_phase, der_couple_p_phase = (
                auxiliary.get_der_couple_phase(state, state.z_coord[traj_n, branch_n], state.eigvals[traj_n, branch_n], state.eigvecs[traj_n, branch_n]))
            # execute phase shift
            state.eigvecs[traj_n, branch_n] = np.copy(
                np.matmul(state.eigvecs[traj_n, branch_n], np.diag(np.conjugate(der_couple_q_phase))))
            # recalculate phases and check that they are zero
            der_couple_q_phase, der_couple_p_phase = (
                auxiliary.get_der_couple_phase(state, state.z_coord[traj_n, branch_n], state.eigvals[traj_n, branch_n], state.eigvecs[traj_n, branch_n]))
            if np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2) > 1e-10:
                # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
                # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
                print('Warning: phase init',
                      np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2))
    return state


def update_eigs_previous(state):
    state.eigvecs_previous = np.copy(state.eigvecs)
    return state


def initialize_wf_adb(state):
    # state.wf_adb = auxiliary.psi_db_to_adb_branch(state.wf_db, state.eigvecs)
    state.wf_adb = auxiliary.vec_db_to_adb(state.wf_db, state.eigvecs)
    return state


def initialize_active_surface(state):
    # Options for deterministic branch modelulation, num_branches==num_states
    if state.model.sh_deterministic:
        assert state.model.num_branches == state.model.num_states
        act_surf_ind_0 = np.zeros((state.model.batch_size, state.model.num_branches)) + np.arange(state.model.num_branches, dtype=int)[
                                                                           np.newaxis, :]
    else:
        # determine initial active surfaces
        intervals = np.zeros((state.model.batch_size, state.model.num_states))
        for traj_n in range(state.model.batch_size):
            for state_n in range(state.model.num_states):
                intervals[traj_n, state_n] = np.real(np.sum((np.abs(state.wf_adb[traj_n]) ** 2)[0:state_n + 1]))
        # initialize active surface index
        act_surf_ind_0 = np.zeros((state.model.batch_size, state.model.num_branches), dtype=int)
        for traj_n in range(state.model.batch_size):
            for branch_n in range(state.model.num_branches):
                act_surf_ind_0[traj_n, branch_n] = np.arange(state.model.num_states, dtype=int)[
                    intervals[traj_n] > state.stochastic_sh_rand_vals[traj_n, branch_n]][0]
            act_surf_ind_0[traj_n] = np.sort(act_surf_ind_0[traj_n])
    # initialize active surface and active surface index in each branch
    state.act_surf_ind_0 = act_surf_ind_0.astype(int)
    state.act_surf_ind = np.copy(state.act_surf_ind_0).astype(int)
    act_surf = np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_states), dtype=int)
    for nt in range(state.model.batch_size):
        act_surf[nt][np.arange(state.model.num_branches, dtype=int), state.act_surf_ind[nt]] = 1
    state.act_surf = act_surf.reshape((state.model.batch_size, state.model.num_branches, state.model.num_states)).astype(int)
    return state


def update_quantum_force_act_surf(state):
    traj_ind = np.arange(state.model.batch_size, dtype=int)[:, np.newaxis] + np.zeros((state.model.batch_size, state.model.num_branches),
                                                                               dtype=int)
    branch_ind = np.arange(state.model.num_branches, dtype=int)[np.newaxis, :] + np.zeros(
        (state.model.batch_size, state.model.num_branches), dtype=int)
    state.quantum_force = state.model.dh_qc_dzc(state, state.z_coord, state.eigvecs[traj_ind, branch_ind, :, state.act_surf_ind],
                                          state.eigvecs[traj_ind, branch_ind, :, state.act_surf_ind])
    return state


def initialize_dm_adb_0_fssh(state):
    state.dm_adb_0 = np.zeros((state.model.batch_size, state.model.num_states, state.model.num_states), dtype=complex)

    for n in range(state.model.batch_size):
        # use the first branch, since all branches are identical at t=0
        state.dm_adb_0[n] = np.outer(np.conj(state.wf_adb[n, 0]), state.wf_adb[n, 0])
    return state


def update_wf_db_eigs(state):
    state.wf_db, state.wf_adb = auxiliary.evolve_wf_eigs(state.wf_db, state.eigvals, state.eigvecs, state.model.dt)
    return state


def gauge_fix_eigs(state):
    for nt in range(state.model.batch_size):  # TODO can this loop be eliminated? Make sign_adjust_branch not branch dependent?
        # adjust gauge of eigenvectors
        state.eigvecs[nt], _ = auxiliary.sign_adjust_branch(state, state.z_coord[nt], state.eigvecs[nt], 
                                                            state.eigvecs_previous[nt], state.eigvals[nt])
    return state


def update_active_surface_fssh(state):
    ############################################################
    #                         HOPPING PROCEDURE                #
    ############################################################
    for nt in range(state.model.batch_size):
        rand = state.hopping_probs_rand_vals[nt, state.t_ind]
        for i in range(state.model.num_branches):
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
                    dkj_z = state.model.dh_qc_dz(state, state.z_coord, evec_k, evec_j) / (ev_diff)
                    dkj_zc = state.model.dh_qc_dzc(state, state.z_coord, evec_k, evec_j) / (ev_diff)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(state.model.pq_weight * state.model.mass / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * state.model.pq_weight * state.model.mass)) * 1.0j * (dkj_z - dkj_zc)
                    max_pos_q = np.argmax(np.abs(dkj_q))
                    max_pos_p = np.argmax(np.abs(dkj_p))
                    if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        raise Exception('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!')
                    if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                        raise Exception('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!')
                    delta_z = dkj_zc
                    state.z_coord[nt, i], hopped = state.model.hop(state, state.z_coord[nt, i], delta_z, ev_diff)
                    if hopped:  # adjust active surfaces if a hop has occurred
                        state.act_surf_ind[nt, i] = k
                        state.act_surf[nt, i] = np.zeros_like(state.act_surf[nt, i])
                        state.act_surf[nt, i][state.act_surf_ind[nt, i]] = 1
                    break
    return state


def update_dm_db_fssh(state):
    state.dm_adb = np.einsum('tbi,tbj->tbij', state.wf_adb, np.conj(state.wf_adb), optimize='greedy')
    for nt in range(state.model.batch_size):
        for nb in range(state.model.num_branches):
            np.einsum('...jj->...j', state.dm_adb[nt, nb])[...] = state.act_surf[nt, nb]
    if state.model.sh_deterministic:
        state.dm_adb = np.einsum('tbb->tb', state.dm_adb_0, optimize='greedy')[:, :, np.newaxis, np.newaxis] * state.dm_adb
    else:
        state.dm_adb = state.dm_adb / state.model.num_branches

    state.dm_db = np.sum(auxiliary.mat_adb_to_db(state.dm_adb, state.eigvecs), axis=(0, 1))
    return state


def update_e_q_fssh(state):
    traj_ind = (np.arange(state.model.batch_size, dtype=int)[:, np.newaxis] + \
                np.zeros((state.model.batch_size, state.model.num_branches), dtype=int))
    branch_ind = (np.arange(state.model.num_branches, dtype=int)[np.newaxis, :] + \
                  np.zeros((state.model.batch_size, state.model.num_branches), dtype=int))
    state.e_q = np.sum(state.eigvals[traj_ind, branch_ind, state.act_surf_ind])
    return state


############################################################
#                     CFSSH INGREDIENTS                    #
############################################################


def update_wf_db_delta_eigs(state):
    state.wf_db_delta, state.wf_adb_delta = auxiliary.evolve_wf_eigs(state.wf_db_delta, state.eigvals, state.eigvecs, state.model.dt)
    return state


def initialize_branch_phase(state):
    state.branch_phase = np.zeros((state.model.batch_size, state.model.num_branches))
    return state


def update_branch_phase(state):
    traj_ind = np.arange(state.model.batch_size, dtype=int)[:, np.newaxis] + \
        np.zeros((state.model.batch_size, state.model.num_branches), dtype=int)
    branch_ind = np.arange(state.model.num_branches, dtype=int)[np.newaxis, :] + \
        np.zeros((state.model.batch_size, state.model.num_branches), dtype=int)
    state.branch_phase = state.branch_phase + state.model.dt * state.eigvals[traj_ind, branch_ind, state.act_surf_ind_0]
    return state


def initialize_wf_adb_delta(state):
    # initialize wavefunction as a delta function in each branch
    wf_adb_delta = np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_states), dtype=complex)
    for nt in range(state.model.batch_size):
        wf_adb_delta[nt][np.arange(state.model.num_branches, dtype=int), state.act_surf_ind_0[nt]] = 1.0 + 0.j
    # transform to diabatic basis
    state.wf_adb_delta = wf_adb_delta
    state.wf_db_delta = auxiliary.vec_adb_to_db(wf_adb_delta, state.eigvecs)
    return state


def update_active_surface_cfssh(state):
    ############################################################
    #                         HOPPING PROCEDURE                #
    ############################################################
    for nt in range(state.model.batch_size):
        rand = state.hopping_probs_rand_vals[nt, state.t_ind]
        for i in range(state.model.num_branches):
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
                    dkj_z = state.model.dh_qc_dz(state, state.z_coord, evec_k, evec_j) / (ev_diff)
                    dkj_zc = state.model.dh_qc_dzc(state, state.z_coord, evec_k, evec_j) / (ev_diff)
                    # check that nonadiabatic couplings are real-valued
                    dkj_q = np.sqrt(state.model.pq_weight * state.model.mass / 2) * (dkj_z + dkj_zc)
                    dkj_p = np.sqrt(1 / (2 * state.model.pq_weight * state.model.mass)) * 1.0j * (dkj_z - dkj_zc)
                    max_pos_q = np.argmax(np.abs(dkj_q))
                    max_pos_p = np.argmax(np.abs(dkj_p))
                    if np.abs(dkj_q[max_pos_q]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2:
                        raise Exception('dkj_q Nonadiabatic coupling is complex, needs gauge fixing!')
                    if np.abs(dkj_p[max_pos_p]) > 1e-8 and np.abs(
                            np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                        raise Exception('dkj_p Nonadiabatic coupling is complex, needs gauge fixing!')
                    delta_z = dkj_zc
                    state.z_coord[nt, i], hopped = state.model.hop(state, state.z_coord[nt, i], delta_z, ev_diff)
                    if hopped:  # adjust active surfaces if a hop has occured
                        state.act_surf_ind[nt, i] = k
                        state.act_surf[nt, i] = np.zeros_like(state.act_surf[nt, i])
                        state.act_surf[nt, i][state.act_surf_ind[nt, i]] = 1
                    break
    return state


def update_classical_overlap(state):
    # calculate overlap matrix
    state.overlap = np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_branches))
    for nt in range(state.model.batch_size):
        state.overlap[nt] = auxiliary.get_classical_overlap(state, state.z_coord[nt])
    return state


def update_dm_db_cfssh(state):
    # state.dm_adb_branch = np.einsum('ni,nj->nij', state.wf_adb, np.conj(state.wf_adb))
    # for nt in range(state.model.batch_size):
    #    np.einsum('...jj->...j', state.dm_adb_branch[nt * state.model.num_branches:(nt + 1) * state.model.num_branches])[
    #        ...] = state.act_surf[nt * state.model.num_branches:(nt + 1) * state.model.num_branches]
    state.dm_adb = np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_states, state.model.num_states), dtype=complex)
    dm_adb_coh = np.zeros((state.model.batch_size, state.model.num_states, state.model.num_states), dtype=complex)
    dm_db_coh = np.zeros((state.model.batch_size, state.model.num_states, state.model.num_states), dtype=complex)
    for nt in range(state.model.batch_size):
        dm_adb_coh_ij = np.zeros((state.model.num_states, state.model.num_states), dtype=complex)
        branch_phase_nt = state.branch_phase[nt]
        act_surf_ind_nt = state.act_surf_ind[nt]
        act_surf_nt = state.act_surf[nt]
        act_surf_ind_0_nt = state.act_surf_ind_0[nt]
        #z_coord_nt = state.z_coord[nt]
        for i in range(state.model.num_branches):
            for j in range(i + 1, state.model.num_branches):
                a_i = act_surf_ind_nt[i]
                a_j = act_surf_ind_nt[j]
                a_i_0 = act_surf_ind_0_nt[i]
                a_j_0 = act_surf_ind_0_nt[j]
                if a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(state.dm_adb_0[nt][a_i, a_j]) > 1e-12:
                    if state.model.sh_deterministic:
                        prob_fac = 1
                    else:
                        prob_fac = 1 / (state.dm_adb_0[nt][a_i, a_i] * state.dm_adb_0[nt][a_j, a_j] * (
                                    state.model.num_branches - 1))
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
                    dm_adb_coh_ij = np.zeros((state.model.num_states, state.model.num_states), dtype=complex)

        if state.model.sh_deterministic:
            rho_diag = np.diag(state.dm_adb_0[nt]).reshape((-1, 1)) * act_surf_nt
            np.einsum('...jj->...j', state.dm_adb[nt], optimize='greedy')[...] = rho_diag
        else:
            # TODO check this stochastic density matrix construction
            for n in range(state.model.num_branches):
                state.dm_adb[n, state.act_surf_ind[n], state.act_surf_ind[n]] += 1 # This is probably wrong with the indices
    state.dm_db = auxiliary.mat_adb_to_db(state.dm_adb, state.eigvecs)
    if state.model.sh_deterministic:
        state.dm_db = (state.dm_db + (dm_db_coh[:, np.newaxis, :, :] / state.model.num_branches))
    else:
        state.dm_db = (state.dm_db + (dm_db_coh[:, np.newaxis, :, :] / state.model.num_branches)) / state.model.num_branches
    
    state.dm_db = np.sum(state.dm_db, axis=(0, 1))
    return state


def update_branch_pair_eigs(state):
    state.eigvals_branch_pair = np.zeros((state.model.batch_size, state.model.num_branches, 
                                          state.model.num_branches, state.model.num_states), dtype=float)
    state.eigvecs_branch_pair = np.zeros((state.model.batch_size, state.model.num_branches, state.model.num_branches, 
                                          state.model.num_states, state.model.num_states), dtype=complex)
    h_q = state.model.h_q(state)
    for nt in range(state.model.batch_size):
        for i in range(state.model.num_branches):
            for j in range(i, state.model.num_branches):
                z_coord_nt_ij = (state.z_coord[nt, i] + state.z_coord[nt, j]) / 2
                state.eigvals_branch_pair[nt, i, j], state.eigvecs_branch_pair[nt, i, j] = np.linalg.eigh(h_q + state.model.h_qc(state, z_coord_nt_ij))
                if i != j:
                    state.eigvals_branch_pair[nt, j, i], state.eigvecs_branch_pair[nt, j, i] = state.eigvals_branch_pair[nt, i, j], state.eigvecs_branch_pair[nt, i, j]
    return state


def gauge_fix_branch_pair_eigs(state):
    # fix the gauge of each set of branches in each batch of trajectories
    for nt in range(state.model.batch_size):
        state.eigvals_branch_pair[nt], state.eigvecs_branch_pair[nt] = \
        auxiliary.sign_adjust_branch_pair_eigs(state,state.z_coord[nt], state.eigvecs_branch_pair[nt],
            state.eigvals_branch_pair[nt], state.eigvecs_branch_pair_previous[nt])
    return state


def update_branch_pair_eigs_previous(state):
    state.eigvecs_branch_pair_previous = np.copy(state.eigvecs_branch_pair)
    return state


def analytic_gauge_fix_branch_pair_eigs(state):
    # compute initial eigenvalues and eigenvectors in each branch
    for n in range(state.model.batch_size):
        for b in range(state.model.num_branches):
            for d in range(state.model.num_branches):
                # compute initial gauge shift for real-valued derivative couplings
                z_coord_ij = (state.z_coord[n,b] + state.z_coord[n,d])/2
                der_couple_q_phase, der_couple_p_phase = (auxiliary.get_der_couple_phase(state, z_coord_ij, state.eigvals_branch_pair[n, b, d], state.eigvecs_branch_pair[n, b, d]))
                # execute phase shift
                state.eigvecs_branch_pair[n, b, d] = np.copy(np.matmul(state.eigvecs_branch_pair[n, b, d], np.diag(np.conjugate(der_couple_q_phase))))
                # recalculate phases and check that they are zero
                der_couple_q_phase, der_couple_p_phase = (
                    auxiliary.get_der_couple_phase(state, z_coord_ij, state.eigvals_branch_pair[n, b, d], state.eigvecs_branch_pair[n, b, d]))
                if np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2) > 1e-10:
                    # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
                    # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
                    print('Warning: phase init',
                        np.sum(np.abs(np.imag(der_couple_q_phase)) ** 2 + np.abs(np.imag(der_couple_p_phase)) ** 2))
    return state

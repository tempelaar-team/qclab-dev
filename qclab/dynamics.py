import numpy as np
import qclab.auxilliary as auxilliary
import qclab.simulation as simulation
from tqdm import tqdm

    

def dynamics(dyn, sim, traj=simulation.Trajectory()):
    sim = auxilliary.load_defaults(sim)
    dyn = dyn(sim)
    # initialize dynamics
    dyn.initialize_dynamics(sim)
    # generate t=0 observables
    dyn.calculate_observables(sim)
    for key in dyn.observables_t.keys():
        traj.new_observable(key, (len(dyn.tdat_output), *np.shape(dyn.observables_t[key])), dyn.observables_t[key].dtype)
    # Begin dynamics loops
    t_output_ind = 0
    for dyn.t_ind in tqdm(np.arange(0, len(dyn.tdat))):
        if t_output_ind == len(dyn.tdat_output):
            break
        ############################################################
        #                            OUTPUT TIMESTEP               #
        ############################################################
        if dyn.tdat_output[t_output_ind] <= dyn.tdat[dyn.t_ind] + 0.5 * sim.dt:
            dyn.calculate_observables(sim)
            traj.add_observable_dict(t_output_ind, dyn.observables_t)
            t_output_ind += 1
        ############################################################
        #                         DYNAMICS TIMESTEP                #
        ############################################################
        dyn.propagate_classical_subsystem(sim)
        dyn.propagate_quantum_subsystem(sim)
        dyn.update_state(sim)
    traj.add_to_dic('t', dyn.tdat_output * sim.num_trajs)
    return traj


def dynamics_old(sim, traj=simulation.Trajectory()):
    sim = auxilliary.load_defaults(sim)
    # initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # store the number of states
    assert sim.num_states == len(psi_db)
    num_branches = sim.num_branches
    num_states = sim.num_states
    num_trajs = sim.num_trajs
    num_class_coords = sim.num_classical_coordinates
    # compute initial Hamiltonian
    h_q = np.copy(sim.h_q(sim.h_q_params))
    h_q_branch = np.repeat(h_q[np.newaxis, :, :], num_branches * num_trajs, axis=0)
    # initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    # initialize random numbers needed in each trajectory
    hopping_probs_rand_vals = np.zeros((num_trajs, len(tdat_bath)))
    stochastic_sh_rand_vals = np.zeros((num_trajs, num_branches))
    for nt in range(num_trajs):
        np.random.seed(traj.seed[nt])
        hopping_probs_rand_vals[nt, :] = np.random.rand(len(tdat_bath))
        stochastic_sh_rand_vals[nt, :] = np.random.rand(num_branches)
    # initialize classical coordinate in each branch
    z_branch = np.zeros((num_trajs, num_branches, num_class_coords), dtype=complex)
    for nt in range(num_trajs):
        for n in range(num_branches):
            z_branch[nt, n] = sim.init_classical(sim, traj.seed[nt])
    z_branch = z_branch.reshape((num_trajs * num_branches, num_class_coords))
    ############################################################
    #              MEAN FIELD SPECIFIC INITIALIZATION     #
    ############################################################
    if sim.dynamics_method == 'MF':
        assert num_branches == 1
        h_tot_branch = h_q_branch + sim.h_qc_branch(sim.h_qc_params, z_branch)
        psi_db_branch = np.zeros((num_trajs * num_branches, num_states), dtype=complex)
        psi_db_branch[:] = psi_db
    ############################################################
    #              SURFACE HOPPING SPECIFIC INITIALIZATION     #
    ############################################################
    if sim.dynamics_method == 'CFSSH' or sim.dynamics_method == 'FSSH':
        # initialize adiabatic wavefunctions
        psi_adb_branch = np.zeros((num_trajs * num_branches, num_states), dtype=complex)
        # initialize adiabatic density matrix in each trajectory
        rho_adb_0_traj = np.zeros((num_trajs, num_states, num_states), dtype=complex)
        # initialize branch eigenvalues and eigenvectors variables
        evals_branch = np.zeros((num_trajs * num_branches, num_states))
        evecs_branch = np.zeros((num_trajs * num_branches, num_states, num_states), dtype=complex)
        # initialize branch-pair eigenvalues and eigenvectors if needed
        if sim.dmat_const > 0:
            evecs_branch_pair = np.zeros((num_trajs, num_branches, num_branches, num_states, num_states), dtype=complex)
            evals_branch_pair = np.zeros((num_trajs, num_branches, num_branches, num_states))
        # initialize Hamiltonian in all branches and trajectories
        h_tot_branch = h_q_branch + sim.h_qc_branch(sim.h_qc_params, z_branch)
        # compute initial eigenvalues and eigenvectors in each branch
        for n in range(num_trajs):
            evals_0, evecs_0 = np.linalg.eigh(
                h_tot_branch[n * num_branches])  # first set of eigs corresponding to branches belonging to trajectory n
            z_branch_0 = z_branch[
                n * num_branches]  # first set of coordinates corresponding to branches belonging to trajectory n
            # compute initial gauge shift for real-valued derivative couplings
            dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z_branch_0, sim)
            # execute phase shift
            evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
            # recalculate phases and check that they are zero
            dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z_branch_0, sim)
            if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
                # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
                # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
                print('Warning: phase init',
                      np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
            # initialize eigenvalues and eigenvectors in each branch for each trajectory
            evals_branch[n * num_branches:(n + 1) * num_branches] = evals_0
            evecs_branch[n * num_branches:(n + 1) * num_branches] = evecs_0
            # initial wavefunction in branches for each trajectory
            psi_adb_n = auxilliary.psi_db_to_adb(psi_db, evecs_0)
            psi_adb_branch[n * num_branches:(n + 1) * num_branches] = psi_adb_n
            rho_adb_0_traj[n] = np.outer(psi_adb_n, np.conj(psi_adb_n))  # np.real(np.abs(psi_adb_n)**2)
            # store branch-pair eigenvalues and eigenvectors if needed
            if sim.dmat_const > 0:
                evecs_branch_pair[n, :, :] = evecs_0
                evals_branch_pair[n, :, :] = evals_0
        ############################################################
        #                   ACTIVE SURFACE INITIALIZATION          #
        ############################################################
        # Options for deterministic branch simulation, num_branches==num_states
        if sim.sh_deterministic:
            assert num_branches == num_states
            act_surf_ind_0 = np.zeros((num_trajs, num_branches)) + np.arange(num_branches, dtype=int)[np.newaxis, :]
        else:
            if sim.dynamics_method == 'CFSSH':
                assert num_branches > 1
            # determine initial active surfaces
            intervals = np.zeros(num_trajs, num_states)
            for nt in range(num_trajs):
                for n in range(num_states):
                    intervals[nt, n] = np.sum(np.diag(rho_adb_0_traj[nt])[0:n + 1])
            # initialize active surface index
            act_surf_ind_0 = np.zeros((num_trajs, num_branches), dtype=int)
            for nt in range(num_trajs):
                # np.random.seed(traj.seed[nt])
                # rand_val = np.random.rand(num_branches)
                for n in range(num_branches):
                    act_surf_ind_0[nt, n] = \
                        np.arange(num_states, dtype=int)[intervals[nt] > stochastic_sh_rand_vals[nt, n]][0]
                act_surf_ind_0[nt] = np.sort(act_surf_ind_0[nt])
        # initialize active surface and active surface index in each branch
        act_surf_ind_0 = act_surf_ind_0.reshape(num_trajs * num_branches).astype(int)
        act_surf_ind_branch = np.copy(act_surf_ind_0)
        act_surf_branch = np.zeros((num_trajs, num_branches, num_states), dtype=int)
        for nt in range(num_trajs):
            act_surf_branch[nt][
                np.arange(num_branches, dtype=int), act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches]] = 1
        act_surf_branch = act_surf_branch.reshape((num_trajs * num_branches, num_states)).astype(int)
        ############################################################
        #                    WAVEFUNCTION INITIALIZATION           #
        ############################################################
        # initialize wavefunction as a delta function in each branch
        psi_adb_delta_branch = np.zeros((num_trajs, num_branches, num_states), dtype=complex)
        for nt in range(num_trajs):
            psi_adb_delta_branch[nt][np.arange(num_branches, dtype=int), act_surf_ind_0[nt * num_branches:(
                                                                                                                  nt + 1) * num_branches]] = 1.0 + 0.j
        psi_adb_delta_branch = psi_adb_delta_branch.reshape((num_trajs * num_branches, num_states))
        # transform to diabatic basis
        psi_db_branch = auxilliary.psi_adb_to_db_branch(psi_adb_branch, evecs_branch)
        psi_db_delta_branch = auxilliary.psi_adb_to_db_branch(psi_adb_delta_branch, evecs_branch)

        ############################################################
        #         COHERENT SURFACE HOPPING SPECIFIC INITIALIZATION#
        ############################################################
        # store the phase of each branch
        phase_branch = np.zeros(num_trajs * num_branches)
        rho_db_fssh_out = np.zeros((len(tdat), num_states, num_states), dtype=complex)
        rho_db_cfssh_out = np.zeros((len(tdat), num_states, num_states), dtype=complex)
    ############################################################
    #                        TIME EVOLUTION                   #
    ############################################################
    t_ind = 0
    for t_bath_ind in np.arange(0, len(tdat_bath)):
        if t_ind == len(tdat):
            break
        ############################################################
        #                            OUTPUT TIMESTEP               #
        ############################################################
        if tdat[t_ind] <= tdat_bath[t_bath_ind] + 0.5 * sim.dt_bath:
            # First calculate density matrices
            ############################################################
            #                                 CFSSH                    #
            ############################################################
            if sim.calc_cfssh_obs:
                if sim.cfssh_branch_pair_update == 1 and sim.dmat_const == 1:  # update branch-pairs every output timestep
                    evecs_branch_pair_previous = np.copy(evecs_branch_pair)
                    for nt in range(num_trajs):
                        evals_branch_pair[nt], evecs_branch_pair[nt] = auxilliary.get_branch_pair_eigs(
                            z_branch[nt * num_branches:(nt + 1) * num_branches], evecs_branch_pair_previous[nt], sim)
                # calculate overlap matrix
                overlap = np.zeros((num_trajs, num_branches, num_branches))
                for nt in range(num_trajs):
                    overlap[nt] = auxilliary.get_classical_overlap(z_branch[nt * num_branches:(nt + 1) * num_branches],
                                                                   sim)
                if sim.dmat_const == 0:
                    # Inexpensive density matrix construction
                    rho_adb_cfssh_branch = np.zeros((num_trajs, num_branches, num_states, num_states), dtype=complex)
                    rho_adb_cfssh_coh = np.zeros((num_trajs, num_states, num_states), dtype=complex)
                    for nt in range(num_trajs):
                        phase_branch_nt = phase_branch[nt * num_branches:(nt + 1) * num_branches]
                        act_surf_ind_branch_nt = act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches]
                        act_surf_ind_0_nt = act_surf_ind_0[nt * num_branches:(nt + 1) * num_branches]
                        act_surf_branch_nt = act_surf_branch[nt * num_branches:(nt + 1) * num_branches]
                        for i in range(num_branches):
                            for j in range(i + 1, num_branches):
                                a_i = act_surf_ind_branch_nt[i]
                                a_j = act_surf_ind_branch_nt[j]
                                a_i_0 = act_surf_ind_0_nt[i]
                                a_j_0 = act_surf_ind_0_nt[j]
                                if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(
                                        rho_adb_0_traj[nt][a_i, a_j]) > 1e-12:
                                    if sim.sh_deterministic:
                                        prob_fac = 1
                                    else:
                                        prob_fac = 1 / (rho_adb_0_traj[nt][a_i, a_i] * rho_adb_0_traj[nt][a_j, a_j] * (
                                                num_branches - 1))
                                    rho_ij = prob_fac * rho_adb_0_traj[nt][a_i, a_j] * overlap[nt][i, j] * \
                                             np.exp(-1.0j * (phase_branch_nt[i] - phase_branch_nt[j]))
                                    rho_adb_cfssh_coh[nt][a_i, a_j] += rho_ij
                                    rho_adb_cfssh_coh[nt][a_j, a_i] += np.conj(rho_ij)
                        if sim.sh_deterministic:
                            # construct diagonal of adiaabtic density matrix
                            rho_adb_cfssh_branch_diag = np.diag(rho_adb_0_traj[nt]).reshape(
                                (-1, 1)) * act_surf_branch_nt
                            np.einsum('...jj->...j', rho_adb_cfssh_branch[nt])[...] = rho_adb_cfssh_branch_diag
                            rho_adb_cfssh_branch[nt] = rho_adb_cfssh_branch[nt] + rho_adb_cfssh_coh[nt] / num_branches
                        else:
                            for n in range(num_branches):
                                rho_adb_cfssh_branch[nt][n, act_surf_ind_branch[n], act_surf_ind_branch[n]] += 1
                            # add coherences averaged over branches
                            rho_adb_cfssh_branch[nt] = (rho_adb_cfssh_branch[nt] + rho_adb_cfssh_coh[
                                nt] / num_branches) / num_branches
                    rho_adb_cfssh_branch = rho_adb_cfssh_branch.reshape(num_trajs * num_branches, num_states,
                                                                        num_states)
                    rho_db_cfssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_cfssh_branch, evecs_branch)
                    rho_db_cfssh = np.sum(rho_db_cfssh_branch, axis=0)
                # expensive CFSSH density matrix construction
                if sim.dmat_const == 1:
                    evecs_branch_pair_previous = np.copy(evecs_branch_pair)
                    # assert sim.sh_deterministic == True
                    rho_adb_cfssh_branch = np.zeros((num_trajs, num_branches, num_states, num_states), dtype=complex)
                    rho_adb_cfssh_coh = np.zeros((num_trajs, num_states, num_states), dtype=complex)
                    rho_db_cfssh_coh = np.zeros((num_trajs, num_states, num_states), dtype=complex)
                    for nt in range(num_trajs):
                        rho_adb_cfssh_coh_ij = np.zeros((num_states, num_states), dtype=complex)
                        phase_branch_nt = phase_branch[nt * num_branches:(nt + 1) * num_branches]
                        act_surf_ind_branch_nt = act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches]
                        act_surf_branch_nt = act_surf_branch[nt * num_branches:(nt + 1) * num_branches]
                        act_surf_ind_0_nt = act_surf_ind_0[nt * num_branches:(nt + 1) * num_branches]
                        z_branch_nt = z_branch[nt * num_branches:(nt + 1) * num_branches]
                        for i in range(num_branches):
                            for j in range(i + 1, num_branches):
                                a_i = act_surf_ind_branch_nt[i]
                                a_j = act_surf_ind_branch_nt[j]
                                a_i_0 = act_surf_ind_0_nt[i]
                                a_j_0 = act_surf_ind_0_nt[j]
                                if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(
                                        rho_adb_0_traj[nt][a_i, a_j]) > 1e-12:
                                    if sim.cfssh_branch_pair_update == 0:
                                        z_branch_ij = np.array([(z_branch_nt[i] + z_branch_nt[j]) / 2])
                                        h_tot_branch_ij = h_q + sim.h_qc_branch(sim.h_qc_params, z_branch_ij)[0]
                                        evals_branch_pair[nt][i, j], evecs_branch_pair[nt][i, j] = np.linalg.eigh(
                                            h_tot_branch_ij)
                                        evecs_branch_pair_ij_tmp, _ = auxilliary.sign_adjust_branch(
                                            evecs_branch_pair[nt][i, j].reshape(1, num_states, num_states),
                                            evecs_branch_pair_previous[nt][i, j].reshape(1, num_states, num_states),
                                            evals_branch_pair[nt][i, j].reshape(1, num_states), z_branch_ij, sim)
                                        evecs_branch_pair[nt][i, j] = np.copy(evecs_branch_pair_ij_tmp[0])
                                    if sim.sh_deterministic:
                                        prob_fac = 1
                                    else:
                                        prob_fac = 1 / (rho_adb_0_traj[nt][a_i, a_i] * rho_adb_0_traj[nt][a_j, a_j] * (
                                                num_branches - 1))
                                    coh_ij_tmp = prob_fac * rho_adb_0_traj[nt][a_i, a_j] * overlap[nt][i, j] * np.exp(
                                        -1.0j * (phase_branch_nt[i] - phase_branch_nt[j]))
                                    rho_adb_cfssh_coh_ij[a_i, a_j] += coh_ij_tmp
                                    rho_adb_cfssh_coh_ij[a_j, a_i] += np.conj(coh_ij_tmp)
                                    # transform only the coherence to diabatic basis
                                    # rho_db_cfssh_coh_ij = auxilliary.rho_adb_to_db(rho_adb_cfssh_coh_ij, evecs_branch_pair[i,j])
                                    rho_db_cfssh_coh_ij = coh_ij_tmp * np.outer(evecs_branch_pair[nt][i, j][:, a_i],
                                                                                np.conj(evecs_branch_pair[nt][i, j][:,
                                                                                        a_j])) + \
                                                          np.conj(coh_ij_tmp) * np.outer(
                                        evecs_branch_pair[nt][i, j][:, a_j],
                                        np.conj(evecs_branch_pair[nt][i, j][:, a_i]))
                                    # accumulate coherences for each basis
                                    rho_db_cfssh_coh[nt] = rho_db_cfssh_coh[nt] + rho_db_cfssh_coh_ij
                                    rho_adb_cfssh_coh[nt] = rho_adb_cfssh_coh[nt] + rho_adb_cfssh_coh_ij
                                    # reset the matrix to store the individual adiabatic coherences
                                    rho_adb_cfssh_coh_ij = np.zeros((num_states, num_states), dtype=complex)
                        # place the active surface on the diagonal weighted by the initial populations
                        if sim.sh_deterministic:
                            rho_diag = np.diag(rho_adb_0_traj[nt]).reshape((-1, 1)) * act_surf_branch_nt
                            np.einsum('...jj->...j', rho_adb_cfssh_branch[nt])[...] = rho_diag
                        else:
                            for n in range(num_branches):
                                rho_adb_cfssh_branch[n, act_surf_ind_branch[n], act_surf_ind_branch[n]] += 1
                    rho_adb_cfssh_branch = rho_adb_cfssh_branch.reshape(
                        (num_trajs * num_branches, num_states, num_states))
                    rho_db_cfssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_cfssh_branch, evecs_branch).reshape(
                        (num_trajs, num_branches, num_states, num_states))
                    if sim.sh_deterministic:
                        rho_db_cfssh_branch = (
                                rho_db_cfssh_branch + (rho_db_cfssh_coh[:, np.newaxis, :, :] / num_branches))
                    else:
                        rho_db_cfssh_branch = (rho_db_cfssh_branch + (
                                rho_db_cfssh_coh[:, np.newaxis, :, :] / num_branches)) / num_branches
                    rho_db_cfssh_branch = rho_db_cfssh_branch.reshape(
                        (num_trajs * num_branches, num_states, num_states))
                    rho_db_cfssh = np.sum(rho_db_cfssh_branch, axis=0)
            ############################################################
            #                                 FSSH                     #
            ############################################################
            if sim.calc_fssh_obs:
                if sim.dmat_const == 0:
                    rho_adb_fssh = np.einsum('ni,nj->nij', psi_adb_branch, np.conj(psi_adb_branch))
                    for nt in range(num_trajs):
                        np.einsum('...jj->...j', rho_adb_fssh[nt * num_branches:(nt + 1) * num_branches])[
                            ...] = act_surf_branch[nt * num_branches:(nt + 1) * num_branches]
                    rho_db_fssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_fssh, evecs_branch)
                    if sim.sh_deterministic:
                        rho_db_fssh_branch = (
                                (np.einsum('njj->nj', rho_adb_0_traj).reshape((num_trajs, num_states)))[:, :,
                                np.newaxis, np.newaxis] * rho_db_fssh_branch.reshape(num_trajs, num_branches,
                                                                                     num_states, num_states))
                    else:
                        rho_db_fssh_branch = rho_db_fssh_branch / num_branches
                    rho_db_fssh_branch = rho_db_fssh_branch.reshape(num_trajs * num_branches, num_states, num_states)
                    rho_db_fssh = np.sum(rho_db_fssh_branch, axis=0)
            ############################################################
            #                                  MF                      #
            ############################################################
            if sim.calc_mf_obs:
                if sim.dmat_const == 0:
                    rho_db_mf_branch = np.einsum('ni,nk->nik', psi_db_branch, np.conj(psi_db_branch)) / (num_branches)
                    rho_db_mf = np.sum(rho_db_mf_branch, axis=0)
            if sim.calc_fssh_obs or sim.calc_cfssh_obs or sim.calc_mf_obs:
                # Evaluate the state variables to be used for the calculations of observables
                state_vars = {}
                for i in range(len(sim.state_vars_list)):
                    if sim.state_vars_list[i] in locals():
                        state_vars[sim.state_vars_list[i]] = eval(sim.state_vars_list[i])
            # calculate observables
            if sim.calc_cfssh_obs:
                cfssh_observables_t = sim.cfssh_observables(sim, state_vars)
                cfssh_observables_t['rho_db_cfssh'] = rho_db_cfssh
                eq = 0
                for n in range(len(act_surf_ind_branch)):
                    eq += evals_branch[n][act_surf_ind_branch[n]]
                cfssh_observables_t['e_q'] = eq / num_branches
                cfssh_observables_t['e_c'] = np.sum(sim.h_c_branch(sim.h_c_params, z_branch)) / num_branches
                if t_ind == 0 and t_bath_ind == 0:
                    for key in cfssh_observables_t.keys():
                        traj.new_observable(key, (len(tdat), *np.shape(cfssh_observables_t[key])),
                                            cfssh_observables_t[key].dtype)
                traj.add_observable_dict(t_ind, cfssh_observables_t)
            if sim.calc_fssh_obs:
                fssh_observables_t = sim.fssh_observables(sim, state_vars)
                fssh_observables_t['rho_db_fssh'] = rho_db_fssh
                eq = 0
                for n in range(len(act_surf_ind_branch)):
                    eq += evals_branch[n][act_surf_ind_branch[n]]
                fssh_observables_t['e_q'] = eq / num_branches
                fssh_observables_t['e_c'] = np.sum(sim.h_c_branch(sim.h_c_params, z_branch)) / num_branches
                if t_ind == 0 and t_bath_ind == 0:
                    for key in fssh_observables_t.keys():
                        traj.new_observable(key, (len(tdat), *np.shape(fssh_observables_t[key])),
                                            fssh_observables_t[key].dtype)
                traj.add_observable_dict(t_ind, fssh_observables_t)
            if sim.calc_mf_obs:
                mf_observables_t = sim.mf_observables(sim, state_vars)
                mf_observables_t['rho_db_mf'] = rho_db_mf
                mf_observables_t['e_q'] = np.real(
                    np.einsum('ni,nij,nj', np.conjugate(psi_db_branch), h_tot_branch, psi_db_branch)) / (num_branches)
                mf_observables_t['e_c'] = np.sum(sim.h_c_branch(sim.h_c_params, z_branch)) / (num_branches)
                if t_ind == 0 and t_bath_ind == 0:
                    for key in mf_observables_t.keys():
                        traj.new_observable(key, (len(tdat), *np.shape(mf_observables_t[key])),
                                            mf_observables_t[key].dtype)
                traj.add_observable_dict(t_ind, mf_observables_t)
            t_ind += 1
        ############################################################
        #                     CLASSICAL PROPAGATION                #
        ############################################################
        # calculate quantum force
        if sim.dynamics_method == 'MF':
            qfzc_branch = auxilliary.quantum_force_branch(psi_db_branch, None, z_branch, sim)
        if sim.dynamics_method == 'FSSH' or sim.dynamics_method == 'CFSSH':
            qfzc_branch = auxilliary.quantum_force_branch(evecs_branch, act_surf_ind_branch, z_branch, sim)
        # evolve classical coordinates
        z_branch = auxilliary.rk4_c(z_branch, qfzc_branch, sim.dt_bath, sim)
        # update Hamiltonian
        h_tot_branch = h_q_branch + sim.h_qc_branch(sim.h_qc_params, z_branch)
        ############################################################
        #                  MEAN-FIELD QUANTUM PROPAGATION         #
        ############################################################
        if sim.dynamics_method == 'MF':
            psi_db_branch = auxilliary.rk4_q_branch(h_tot_branch, psi_db_branch, sim.dt_bath)
        ############################################################
        #                SURFACE HOPPING QUANTUM PROPAGATION       #
        ############################################################
        if sim.dynamics_method == 'FSSH' or sim.dynamics_method == 'CFSSH':
            evecs_branch_previous = np.copy(evecs_branch)
            # obtain branch eigenvalues and eigenvectors
            evals_branch, evecs_branch = np.linalg.eigh(h_tot_branch)
            for nt in range(num_trajs):
                # adjust gauge of eigenvectors
                evecs_branch[nt * num_branches:(nt + 1) * num_branches], _ = auxilliary.sign_adjust_branch(
                    evecs_branch[nt * num_branches:(nt + 1) * num_branches], \
                    evecs_branch_previous[nt * num_branches:(nt + 1) * num_branches],
                    evals_branch[nt * num_branches:(nt + 1) * num_branches],
                    z_branch[nt * num_branches:(nt + 1) * num_branches], sim)
            # propagate phases
            phase_branch = phase_branch + sim.dt_bath * evals_branch[
                np.arange(num_branches * num_trajs, dtype=int), act_surf_ind_0]
            # construct eigenvalue exponential
            evals_exp_branch = np.exp(-1.0j * evals_branch * sim.dt_bath)
            # transform wavefunctions to adiabatic basis
            psi_adb_branch = np.copy(auxilliary.psi_db_to_adb_branch(psi_db_branch, evecs_branch))
            psi_adb_delta_branch = np.copy(auxilliary.psi_db_to_adb_branch(psi_db_delta_branch, evecs_branch))
            # multiply by propagator
            psi_adb_branch = np.copy(evals_exp_branch * psi_adb_branch)
            psi_adb_delta_branch = np.copy(evals_exp_branch * psi_adb_delta_branch)
            # transform back to diabatic basis
            psi_db_branch = auxilliary.psi_adb_to_db_branch(psi_adb_branch, evecs_branch)
            psi_db_delta_branch = auxilliary.psi_adb_to_db_branch(psi_adb_delta_branch, evecs_branch)
            # update branch-pairs if needed
            if sim.cfssh_branch_pair_update == 2 and sim.dmat_const == 1:  # update branch-pairs every bath timestep
                evecs_branch_pair_previous = np.copy(evecs_branch_pair)
                for nt in range(num_trajs):
                    evals_branch_pair[nt], evecs_branch_pair_previous[nt] = auxilliary.get_branch_pair_eigs(
                        z_branch[nt * num_branches:(nt + 1) * num_branches], evecs_branch_pair_previous[nt], sim)
            ############################################################
            #                         HOPPING PROCEDURE                #
            ############################################################
            for nt in range(num_trajs):
                rand = hopping_probs_rand_vals[nt, t_bath_ind]
                for i in range(num_branches):
                    # compute hopping probabilities
                    prod = np.matmul(np.conjugate(evecs_branch[nt * num_branches:(nt + 1) * num_branches][i][:,
                                                  act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]), \
                                     evecs_branch_previous[nt * num_branches:(nt + 1) * num_branches][i])
                    if sim.pab_cohere:
                        hop_prob = -2 * np.real(prod * (psi_adb_branch[nt * num_branches:(nt + 1) * num_branches][i] \
                                                        / psi_adb_branch[nt * num_branches:(nt + 1) * num_branches][i][
                                                            act_surf_ind_branch[
                                                            nt * num_branches:(nt + 1) * num_branches][i]]))
                    if not sim.pab_cohere:
                        hop_prob = -2 * np.real(
                            prod * (psi_adb_delta_branch[nt * num_branches:(nt + 1) * num_branches][i] \
                                    / psi_adb_delta_branch[nt * num_branches:(nt + 1) * num_branches][i][
                                        act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]))
                    hop_prob[act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]] = 0
                    bin_edge = 0
                    # hop if possible
                    for k in range(len(hop_prob)):
                        print(hop_prob)
                        hop_prob[k] = auxilliary.nan_num(hop_prob[k])
                        bin_edge = bin_edge + hop_prob[k]
                        if rand < bin_edge:
                            # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                            evec_k = evecs_branch[nt * num_branches:(nt + 1) * num_branches][i][:,
                                     act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]
                            evec_j = evecs_branch[nt * num_branches:(nt + 1) * num_branches][i][:, k]
                            eval_k = evals_branch[nt * num_branches:(nt + 1) * num_branches][i][
                                act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]
                            eval_j = evals_branch[nt * num_branches:(nt + 1) * num_branches][i][k]
                            ev_diff = eval_j - eval_k
                            # dkj_q is wrt q dkj_p is wrt p.
                            dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff,
                                                               z_branch[nt * num_branches:(nt + 1) * num_branches][i],
                                                               sim)
                            ## check that nonadiabatic couplings are real-valued
                            dkj_q = np.sqrt(sim.h * sim.m / 2) * (dkj_z + dkj_zc)
                            dkj_p = np.sqrt(1 / (2 * sim.h * sim.m)) * 1.0j * (dkj_z - dkj_zc)
                            if np.abs(np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2 or \
                                    np.abs(np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                                raise Exception('Nonadiabatic coupling is complex, needs gauge fixing!')
                            delta_z = dkj_zc
                            z_branch[nt * num_branches:(nt + 1) * num_branches][i], hopped = \
                                sim.hop(sim, z_branch[nt * num_branches:(nt + 1) * num_branches][i], delta_z, ev_diff)
                            if hopped:  # adjust active surfaces if a hop has occured
                                act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i] = k
                                act_surf_branch[nt * num_branches:(nt + 1) * num_branches][i] = np.zeros_like(
                                    act_surf_branch[nt * num_branches:(nt + 1) * num_branches][i])
                                act_surf_branch[nt * num_branches:(nt + 1) * num_branches][i][
                                    act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]] = 1
                            break
    traj.add_to_dic('t', tdat * num_trajs)
    return traj

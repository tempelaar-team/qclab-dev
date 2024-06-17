import simulation
import numpy as np
import auxilliary


def dynamics(sim,traj=simulation.Trajectory(None)):
    np.random.seed(traj.seed)
    # initial wavefunction in diabatic basis
    psi_db = sim.psi_db_0
    # store the number of states
    num_states = len(psi_db)
    sim.num_states = num_states
    if sim.num_branches is None:
        sim.num_branches = num_states
    num_branches = sim.num_branches
    # compute initial Hamiltonian
    h_q = np.copy(sim.h_q())
    h_q_branch = np.repeat(h_q[np.newaxis,:,:],num_branches,axis=0)
    # initialize outputs
    tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
    tdat_bath = np.arange(0, sim.tmax + sim.dt_bath, sim.dt_bath)
    ############################################################
    #              MEAN FIELD SPECIFIC INITIALIZATION     #
    ############################################################
    if sim.dynamics_method == 'MF':
        # initialize classical coordinate in each branch
        for n in range(num_branches):
            if n == 0:
                z_branch = sim.init_classical(np.random.randint(1,10000000))
                z_0 = np.copy(z_branch)
            else:
                z_branch = np.vstack((z_branch,sim.init_classical(np.random.randint(1,10000000))))
        z_branch = z_branch.reshape((num_branches, len(z_0)))
        h_tot_branch = h_q_branch + sim.h_qc_branch(z_branch)#np.transpose(np.apply_along_axis(sim.h_qc, 1, z_branch), axes=(0,1,2))
        psi_db_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_db_branch[:] = psi_db
        rho_db_mf_out = np.zeros((len(tdat), num_states, num_states), dtype=complex)
    ############################################################
    #              SURFACE HOPPING SPECIFIC INITIALIZATION     #
    ############################################################
    if sim.dynamics_method == 'CFSSH' or sim.dynamics_method == 'FSSH':
        # initialize classical coordinate in each branch
        for n in range(num_branches):
            if n == 0:
                z_branch = sim.init_classical(traj.seed)
                z_0 = np.copy(z_branch)
            else:
                z_branch = np.vstack((z_branch,z_0))
        z_branch = z_branch.reshape((num_branches, len(z_0)))
        h_tot_branch = h_q_branch + sim.h_qc_branch(z_branch)#sim.h_qc(z_branch[0])[np.newaxis, :, :]
        # compute initial eigenvalues and eigenvectors
        evals_0, evecs_0 = np.linalg.eigh(h_tot_branch[0])
        # compute initial gauge shift for real-valued derivative couplings
        dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z_branch[0], sim)
        # execute phase shift
        evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
        # recalculate phases and check that they are zero
        dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z_branch[0], sim)
        if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
            # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
            # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
            print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
        # determine initial adiabatic wavefunction in fixed gauge
        psi_adb = auxilliary.psi_db_to_adb(psi_db, evecs_0)
        # determine initial adiabatic density matrix
        rho_adb_0 = np.outer(psi_adb, np.conj(psi_adb))
        # initial wavefunction in branches
        psi_adb_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_adb_branch[:] = psi_adb
        # initialize eigenvalues and eigenvectors in each branch
        evals_branch = np.zeros((num_branches, num_states))
        evecs_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
        evals_branch[:] = evals_0
        evecs_branch[:] = evecs_0
        # initialize branch-pair eigenvalues and eigenvectors
        if sim.dmat_const > 0:
            evecs_branch_pair = np.zeros((num_branches, num_branches, num_states, num_states), dtype=complex)
            evals_branch_pair = np.zeros((num_branches, num_branches, num_states))
            evecs_branch_pair[:, :] = evecs_0
            evals_branch_pair[:, :] = evals_0
        ############################################################
        #                   ACTIVE SURFACE INITIALIZATION          #
        ############################################################
        # Options for deterministic branch simulation, num_branches==num_states
        if sim.sh_deterministic:
            assert num_branches == num_states
            act_surf_ind_0 = np.arange(num_branches,dtype=int)
        else:
            if sim.dynamics_method == 'CFSSH':
                assert num_branches > 1
            # determine initial active surfaces
            intervals = np.zeros(num_states)
            for n in range(num_states):
                intervals[n] = np.sum(np.real(np.abs(psi_adb) ** 2)[0:n + 1])
            rand_val = np.random.rand(num_branches)
            # initialize active surface index
            act_surf_ind_0 = np.zeros((num_branches), dtype=int)
            for n in range(num_branches):
                act_surf_ind_0[n] = np.arange(num_states)[intervals > rand_val[n]][0]
            act_surf_ind_0 = np.sort(act_surf_ind_0)
        # initialize active surface and active surface index in each branch
        act_surf_ind_branch = np.copy(act_surf_ind_0)
        act_surf_branch = np.zeros((num_branches, num_states), dtype=int)
        act_surf_branch[np.arange(num_branches, dtype=int), act_surf_ind_branch] = 1

        ############################################################
        #                    WAVEFUNCTION INITIALIZATION           #
        ############################################################
        # initialize wavefunction as a delta function in each branch
        psi_adb_delta_branch = np.zeros((num_branches, num_states), dtype=complex)
        psi_adb_delta_branch[np.arange(num_branches, dtype=int), act_surf_ind_0] = 1.0 + 0.j
        # transform to diabatic basis
        psi_db_branch = auxilliary.psi_adb_to_db_branch(psi_adb_branch, evecs_branch)
        psi_db_delta_branch = auxilliary.psi_adb_to_db_branch(psi_adb_delta_branch, evecs_branch)

        ############################################################
        #         COHERENT SURFACE HOPPING SPECIFIC INITIALIZATION#
        ############################################################
        # store the phase of each branch
        phase_branch = np.zeros(num_branches)
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
                    evals_branch_pair, evecs_branch_pair = auxilliary.get_branch_pair_eigs(z_branch, evecs_branch_pair_previous, sim)
                # calculate overlap matrix
                overlap = auxilliary.get_classical_overlap(z_branch, sim)
                if sim.dmat_const == 0:
                    # Inexpensive density matrix construction
                    rho_adb_cfssh_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
                    rho_adb_cfssh_coh = np.zeros((num_states, num_states), dtype=complex)
                    for i in range(num_branches):
                        for j in range(i+1, num_branches):
                            a_i = act_surf_ind_branch[i]
                            a_j = act_surf_ind_branch[j]
                            a_i_0 = act_surf_ind_0[i]
                            a_j_0 = act_surf_ind_0[j]
                            if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(rho_adb_0[a_i,a_j]) > 1e-12:
                                if sim.sh_deterministic:
                                    prob_fac = 1
                                else:
                                    prob_fac = 1/(rho_adb_0[a_i,a_i]*rho_adb_0[a_j,a_j]*(num_branches-1))
                                rho_ij = prob_fac * rho_adb_0[a_i,a_j] * overlap[i, j] * \
                                    np.exp(-1.0j*(phase_branch[i] - phase_branch[j]))
                                rho_adb_cfssh_coh[a_i, a_j] += rho_ij
                                rho_adb_cfssh_coh[a_j, a_i] += np.conj(rho_ij)
                    if sim.sh_deterministic:
                        # construct diagonal of adiaabtic density matrix
                        rho_adb_cfssh_branch_diag = np.diag(rho_adb_0).reshape((-1, 1)) * act_surf_branch
                        np.einsum('...jj->...j', rho_adb_cfssh_branch)[...] = rho_adb_cfssh_branch_diag
                        rho_adb_cfssh_branch = rho_adb_cfssh_branch + rho_adb_cfssh_coh / num_branches
                    else:
                        for n in range(num_branches):
                            rho_adb_cfssh_branch[n, act_surf_ind_branch[n], act_surf_ind_branch[n]] += 1
                        # add coherences averaged over branches
                        rho_adb_cfssh_branch = (rho_adb_cfssh_branch + rho_adb_cfssh_coh / num_branches) / num_branches
                    rho_db_cfssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_cfssh_branch, evecs_branch)
                    rho_db_cfssh = np.sum(rho_db_cfssh_branch, axis=0)
                # expensive CFSSH density matrix construction
                if sim.dmat_const == 1:
                    evecs_branch_pair_previous = np.copy(evecs_branch_pair)
                    #assert sim.sh_deterministic == True
                    rho_adb_cfssh_branch = np.zeros((num_branches, num_states, num_states), dtype=complex)
                    rho_adb_cfssh_coh_ij = np.zeros((num_states, num_states), dtype=complex)
                    rho_adb_cfssh_coh = np.zeros((num_states, num_states), dtype=complex)
                    rho_db_cfssh_coh = np.zeros((num_states, num_states), dtype=complex)
                    for i in range(num_branches):
                        for j in range(i + 1, num_branches):
                            a_i = act_surf_ind_branch[i]
                            a_j = act_surf_ind_branch[j]
                            a_i_0 = act_surf_ind_0[i]
                            a_j_0 = act_surf_ind_0[j]
                            if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(rho_adb_0[a_i,a_j]) > 1e-12:
                                if sim.cfssh_branch_pair_update == 0:
                                    
                                    z_branch_ij = np.array([(z_branch[i] + z_branch[j])/2])
                                    h_tot_branch_ij = h_q + sim.h_qc_branch(z_branch_ij)[0]#sim.h_qc(z_branch_ij[0])#
                                    evals_branch_pair[i,j], evecs_branch_pair[i,j] = np.linalg.eigh(h_tot_branch_ij)
                                    evecs_branch_pair_ij_tmp,_ = auxilliary.sign_adjust_branch(evecs_branch_pair[i,j].reshape(1,num_states,num_states),
                                                                                    evecs_branch_pair_previous[i,j].reshape(1,num_states,num_states),
                                                                                    evals_branch_pair[i,j].reshape(1,num_states), z_branch_ij, sim)
                                    evecs_branch_pair[i,j] = np.copy(evecs_branch_pair_ij_tmp[0])
                                if sim.sh_deterministic:
                                    prob_fac = 1
                                else:
                                    prob_fac = 1/(rho_adb_0[a_i,a_i]*rho_adb_0[a_j,a_j]*(num_branches-1))
                                coh_ij_tmp = prob_fac*rho_adb_0[a_i,a_j]*overlap[i,j]*np.exp(-1.0j*(phase_branch[i]-phase_branch[j]))
                                rho_adb_cfssh_coh_ij[a_i,a_j] += coh_ij_tmp
                                rho_adb_cfssh_coh_ij[a_j,a_i] += np.conj(coh_ij_tmp)
                                # transform only the coherence to diabatic basis
                                #rho_db_cfssh_coh_ij = auxilliary.rho_adb_to_db(rho_adb_cfssh_coh_ij, evecs_branch_pair[i,j])
                                rho_db_cfssh_coh_ij = coh_ij_tmp*np.outer(evecs_branch_pair[i,j][:,a_i],np.conj(evecs_branch_pair[i,j][:,a_j])) + \
                                    np.conj(coh_ij_tmp)*np.outer(evecs_branch_pair[i,j][:,a_j],np.conj(evecs_branch_pair[i,j][:,a_i]))
                                # accumulate coherences for each basis
                                rho_db_cfssh_coh = rho_db_cfssh_coh + rho_db_cfssh_coh_ij
                                rho_adb_cfssh_coh = rho_adb_cfssh_coh + rho_adb_cfssh_coh_ij
                                # reset the matrix to store the individual adiabatic coherences
                                rho_adb_cfssh_coh_ij = np.zeros((num_states, num_states), dtype=complex)

                    # place the active surface on the diagonal weighted by the initial populations
                    if sim.sh_deterministic:
                        rho_diag = np.diag(rho_adb_0).reshape((-1,1)) * act_surf_branch
                        np.einsum('...jj->...j', rho_adb_cfssh_branch)[...] = rho_diag
                        rho_db_cfssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_cfssh_branch, evecs_branch)
                        rho_db_cfssh_branch = (rho_db_cfssh_branch + (rho_db_cfssh_coh/num_branches))
                    else:
                        for n in range(num_branches):
                            rho_adb_cfssh_branch[n, act_surf_ind_branch[n], act_surf_ind_branch[n]] += 1
                        rho_db_cfssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_cfssh_branch, evecs_branch)
                        rho_db_cfssh_branch = (rho_db_cfssh_branch + (rho_db_cfssh_coh/num_branches))/num_branches
                    rho_db_cfssh = np.sum(rho_db_cfssh_branch, axis=0)
            ############################################################
            #                                 FSSH                     #
            ############################################################
            if sim.calc_fssh_obs:
                if sim.dmat_const == 0:
                    rho_adb_fssh = np.einsum('ni,nj->nij', psi_adb_branch, np.conj(psi_adb_branch))
                    np.einsum('...jj->...j', rho_adb_fssh)[...] = act_surf_branch
                    rho_db_fssh_branch = auxilliary.rho_adb_to_db_branch(rho_adb_fssh, evecs_branch)
                    if sim.sh_deterministic:
                        rho_db_fssh_branch = np.diag(rho_adb_0)[:,np.newaxis,np.newaxis]*rho_db_fssh_branch
                    else:
                        rho_db_fssh_branch = rho_db_fssh_branch/num_branches
                    rho_db_fssh = np.sum(rho_db_fssh_branch, axis=0)
            ############################################################
            #                                  MF                      #
            ############################################################
            if sim.calc_mf_obs:
                if sim.dmat_const == 0:
                    rho_db_mf_branch = np.einsum('ni,nk->nik', psi_db_branch, np.conj(psi_db_branch))
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
                cfssh_observables_t['e_q'] = eq/num_branches
                cfssh_observables_t['e_c'] = np.sum(sim.h_c_branch(z_branch))/num_branches
                if t_ind == 0 and t_bath_ind == 0:
                    for key in cfssh_observables_t.keys():
                        traj.new_observable(key, (len(tdat), *np.shape(cfssh_observables_t[key])), cfssh_observables_t[key].dtype)
                traj.add_observable_dict(t_ind, cfssh_observables_t)
            if sim.calc_fssh_obs:
                fssh_observables_t = sim.fssh_observables(sim, state_vars)
                fssh_observables_t['rho_db_fssh'] = rho_db_fssh
                eq = 0
                for n in range(len(act_surf_ind_branch)):
                    eq += evals_branch[n][act_surf_ind_branch[n]]
                fssh_observables_t['e_q'] = eq/num_branches
                fssh_observables_t['e_c'] = np.sum(sim.h_c_branch(z_branch))/num_branches
                if t_ind == 0 and t_bath_ind == 0:
                    for key in fssh_observables_t.keys():
                        traj.new_observable(key, (len(tdat), *np.shape(fssh_observables_t[key])), fssh_observables_t[key].dtype)
                traj.add_observable_dict(t_ind, fssh_observables_t)
            if sim.calc_mf_obs:
                mf_observables_t = sim.mf_observables(sim, state_vars)
                mf_observables_t['rho_db_mf'] = rho_db_mf
                mf_observables_t['e_q'] = np.real(np.einsum('ni,nij,nj', np.conjugate(psi_db_branch), h_tot_branch, psi_db_branch))/num_branches
                mf_observables_t['e_c'] = np.sum(sim.h_c_branch(z_branch))/num_branches
                if t_ind == 0 and t_bath_ind == 0:
                    for key in mf_observables_t.keys():
                        traj.new_observable(key, (len(tdat), *np.shape(mf_observables_t[key])), mf_observables_t[key].dtype)
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
        h_tot_branch = h_q_branch + sim.h_qc_branch(z_branch)
        #h_tot_branch = h_q_branch + np.transpose(np.apply_along_axis(sim.h_qc, 1, z_branch), axes=(0,1,2))
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
            # adjust gauge of eigenvectors
            evecs_branch,_ = auxilliary.sign_adjust_branch(evecs_branch, evecs_branch_previous, evals_branch, z_branch, sim)
            # propagate phases
            phase_branch = phase_branch + sim.dt_bath * evals_branch[np.arange(num_branches,dtype=int),act_surf_ind_0]
            # construct eigenvalue exponential
            evals_exp_branch = np.exp(-1.0j * evals_branch * sim.dt_bath)
            # evolve wavefunction
            psi_adb_branch = np.copy(auxilliary.psi_db_to_adb_branch(psi_db_branch, evecs_branch))
            psi_adb_delta_branch = np.copy(auxilliary.psi_db_to_adb_branch(psi_db_delta_branch, evecs_branch))
            # multiply by propagator
            psi_adb_branch = np.copy(evals_exp_branch*psi_adb_branch)
            psi_adb_delta_branch = np.copy(evals_exp_branch*psi_adb_delta_branch)
            # transform back to diabatic basis
            psi_db_branch = auxilliary.psi_adb_to_db_branch(psi_adb_branch, evecs_branch)
            psi_db_delta_branch = auxilliary.psi_adb_to_db_branch(psi_adb_delta_branch, evecs_branch)
            # update branch-pairs if needed
            if sim.cfssh_branch_pair_update == 2 and sim.dmat_const == 1:  # update branch-pairs every bath timestep
                evecs_branch_pair_previous = np.copy(evecs_branch_pair)
                evals_branch_pair, evecs_branch_pair_previous = auxilliary.get_branch_pair_eigs(z_branch, evecs_branch_pair_previous, sim)
            ############################################################
            #                         HOPPING PROCEDURE                #
            ############################################################
            # draw a random number (same for all branches)
            rand = np.random.rand()
            for i in range(num_branches):
                # compute hopping probabilities
                prod = np.matmul(np.conjugate(evecs_branch[i][:, act_surf_ind_branch[i]]), evecs_branch_previous[i])
                if sim.pab_cohere:
                    hop_prob = -2 * np.real(prod * (psi_adb_branch[i] / psi_adb_branch[i][act_surf_ind_branch[i]]))
                if not sim.pab_cohere:
                    hop_prob = -2 * np.real(
                        prod * (psi_adb_delta_branch[i] / psi_adb_delta_branch[i][act_surf_ind_branch[i]]))
                hop_prob[act_surf_ind_branch[i]] = 0
                bin_edge = 0
                # hop if possible
                for k in range(len(hop_prob)):
                    hop_prob[k] = auxilliary.nan_num(hop_prob[k])
                    bin_edge = bin_edge + hop_prob[k]
                    if rand < bin_edge:
                        # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                        evec_k = evecs_branch[i][:, act_surf_ind_branch[i]]
                        evec_j = evecs_branch[i][:, k]
                        eval_k = evals_branch[i][act_surf_ind_branch[i]]
                        eval_j = evals_branch[i][k]
                        ev_diff = eval_j - eval_k
                        # dkj_q is wrt q dkj_p is wrt p.
                        dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff, z_branch[i], sim)
                        ## check that nonadiabatic couplings are real-valued
                        dkj_q = np.sqrt(sim.h * sim.m / 2) * (dkj_z + dkj_zc)
                        dkj_p = np.sqrt(1 / (2 * sim.h * sim.m)) * 1.0j * (dkj_z - dkj_zc)
                        if np.abs(np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2 or \
                            np.abs(np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                            raise Exception('Nonadiabatic coupling is complex, needs gauge fixing!')
                        delta_z = dkj_zc
                        z_branch[i], hopped = sim.hop(z_branch[i], delta_z, ev_diff)
                        if hopped: # adjust active surfaces if a hop has occured
                            act_surf_ind_branch[i] = k
                            act_surf_branch[i] = np.zeros_like(act_surf_branch[i])
                            act_surf_branch[i][act_surf_ind_branch[i]] = 1
                        break
    traj.add_to_dic('t', tdat)
    return traj

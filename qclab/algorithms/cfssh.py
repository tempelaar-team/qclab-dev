import numpy as np
import qclab.auxilliary as auxilliary

class CoherentFewestSwitchesSurfaceHoppingDynamics:
    def __init__(self, sim):
        var_names = list(sim.__dict__.keys())
        defaults = {
            'hop': auxilliary.harmonic_oscillator_hop,
            'init_classical': auxilliary.harmonic_oscillator_bolztmann_init_classical,
            'h_c_branch': auxilliary.harmonic_oscillator_h_c_branch,
            'dh_c_dz_branch': auxilliary.harmonic_oscillator_dh_c_dz_branch,
            'dh_c_dzc_branch': auxilliary.harmonic_oscillator_dh_c_dzc_branch,
            'h_c_params' : (sim.h),
            'h_qc_params' : None,
            'h_q_params' : None,
            'tmax': 10,
            'dt_output': 0.1,
            'dt': 0.01,
            'temp':1,
            'num_states':2,
            'num_branches':2,
            'sh_deterministic':True,
            'gauge_fix':0,
            'pab_cohere':False,
            'dmat_const':0,
            'observables':auxilliary.no_observables,
            'num_classical_coordinates':None
            }
        for name in defaults.keys():
            if not(name in list(var_names)):
                sim.__dict__[name] = defaults[name]
        if sim.sh_deterministic:
            assert sim.num_branches == sim.num_states
        else:
            assert sim.num_branches > 1
        
        return
    def initialize_dynamics(self, sim):
        num_trajs = sim.num_trajs
        num_states = sim.num_states
        num_branches = sim.num_branches
        # initialize time axes 
        self.tdat_output = np.arange(0, sim.tmax + sim.dt_output, sim.dt_output)
        self.tdat = np.arange(0, sim.tmax + sim.dt, sim.dt)
        # initialize random numbers needed in each trajectory
        self.hopping_probs_rand_vals = np.zeros((num_trajs, len(self.tdat)))
        self.stochastic_sh_rand_vals = np.zeros((num_trajs, num_branches))
        for nt in range(num_trajs):
            np.random.seed(sim.seeds[nt])
            self.hopping_probs_rand_vals[nt,:] = np.random.rand(len(self.tdat))
            self.stochastic_sh_rand_vals[nt,:] = np.random.rand(num_branches)
        # initialize variables describing the state of the system
        self.z_coord = np.zeros((num_trajs, num_branches, sim.num_classical_coordinates), dtype=complex)
        # load initial values of the z coordinate 
        for traj_n in range(num_trajs):
            self.z_coord[traj_n, :, :] = sim.init_classical(sim, sim.seeds[traj_n]) # init_classical could arguablty be in init_state
        self.z_coord = self.z_coord.reshape(num_trajs*num_branches, sim.num_classical_coordinates)
        # load initial values of the wavefunction
        self.wf_db = np.zeros((num_trajs*num_branches, num_states), dtype=complex) + sim.wf_db[np.newaxis, :]
        # initialize adiabatic wavefunctions
        self.wf_adb = np.zeros((num_trajs*num_branches,num_states), dtype=complex)
        # initialize adiabatic density matrix in each trajectory
        self.dm_adb_0_traj = np.zeros((num_trajs, num_states, num_states), dtype=complex)
        # initialize branch eigenvalues and eigenvectors variables
        self.evals_branch = np.zeros((num_trajs*num_branches, num_states))
        self.evecs_branch = np.zeros((num_trajs*num_branches, num_states, num_states), dtype=complex)
        # initialize branch-pair eigenvalues and eigenvectors if needed
        if sim.dmat_const > 0:
            self.evecs_branch_pair = np.zeros((num_trajs, num_branches, num_branches, num_states, num_states), dtype=complex)
            self.evals_branch_pair = np.zeros((num_trajs, num_branches, num_branches, num_states))
        # initialize variables for the gradients (Hamiltonian and quantum forces)
        self.h_q = sim.h_q(sim.h_q_params)
        self.h_tot = self.h_q + sim.h_qc_branch(sim.h_qc_params, self.z_coord)
        # compute initial eigenvalues and eigenvectors in each branch
        for traj_n in range(num_trajs):
            evals_0, evecs_0 = np.linalg.eigh(self.h_tot[traj_n*num_branches]) # first set of eigs corresponding to branches belonging to trajectory n
            z_coord_0 = self.z_coord[traj_n*num_branches] # first set of coordinates corresponding to branches belonging to trajectory n
            # compute initial gauge shift for real-valued derivative couplings
            dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z_coord_0, sim)
            # execute phase shift
            evecs_0 = np.matmul(evecs_0, np.diag(np.conjugate(dab_q_phase)))
            # recalculate phases and check that they are zero
            dab_q_phase, dab_p_phase = auxilliary.get_dab_phase(evals_0, evecs_0, z_coord_0, sim)
            if np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2) > 1e-10:
                # this error will indicate that symmetries of the Hamiltonian have been broken by the representation
                # and/or that the Hamiltonian is not suitable for SH methods without additional gauge fixing.
                print('Warning: phase init', np.sum(np.abs(np.imag(dab_q_phase)) ** 2 + np.abs(np.imag(dab_p_phase)) ** 2))
            # initialize eigenvalues and eigenvectors in each branch for each trajectory
            self.evals_branch[traj_n*num_branches:(traj_n+1)*num_branches] = evals_0
            self.evecs_branch[traj_n*num_branches:(traj_n+1)*num_branches] = evecs_0
            # initial wavefunction in branches for each trajectory
            wf_adb_n = auxilliary.psi_db_to_adb(self.wf_db[traj_n*num_branches], evecs_0)
            self.wf_adb[traj_n*num_branches:(traj_n+1)*num_branches] = wf_adb_n
            self.dm_adb_0_traj[traj_n] = np.outer(wf_adb_n, np.conj(wf_adb_n))
            # store branch-pair eigenvalues and eigenvectors if needed
            if sim.dmat_const > 0:
                self.evecs_branch_pair[traj_n, :, :] = evecs_0
                self.evals_branch_pair[traj_n, :, :] = evals_0
        ############################################################
        #                   ACTIVE SURFACE INITIALIZATION          #
        ############################################################
        # Options for deterministic branch simulation, num_branches==num_states
        if sim.sh_deterministic:
            assert num_branches == num_states
            act_surf_ind_0 = np.zeros((num_trajs, num_branches)) + np.arange(num_branches,dtype=int)[np.newaxis, :]
        else:
            if sim.dynamics_method == 'CFSSH':
                assert num_branches > 1
            # determine initial active surfaces
            intervals = np.zeros(num_trajs, num_states)
            for traj_n in range(num_trajs):
                for state_n in range(num_states):
                    intervals[traj_n, state_n] = np.sum(np.diag(self.dm_adb_0_traj[traj_n])[0:state_n + 1])
            # initialize active surface index
            act_surf_ind_0 = np.zeros((num_trajs, num_branches), dtype=int)
            for traj_n in range(num_trajs):
                for branch_n in range(num_branches):
                    act_surf_ind_0[traj_n, branch_n] = np.arange(num_states,dtype=int)[intervals[traj_n] > self.stochastic_sh_rand_vals[traj_n, branch_n]][0]
                act_surf_ind_0[traj_n] = np.sort(act_surf_ind_0[traj_n])
        # initialize active surface and active surface index in each branch
        self.act_surf_ind_0 = act_surf_ind_0.reshape(num_trajs*num_branches).astype(int)
        self.act_surf_ind_branch = np.copy(self.act_surf_ind_0).astype(int)
        act_surf_branch = np.zeros((num_trajs, num_branches, num_states), dtype=int)
        for nt in range(num_trajs):
            act_surf_branch[nt][np.arange(num_branches, dtype=int), self.act_surf_ind_branch[nt*num_branches:(nt+1)*num_branches]] = 1
        self.act_surf_branch = act_surf_branch.reshape((num_trajs*num_branches, num_states)).astype(int)
        ############################################################
        #                    WAVEFUNCTION INITIALIZATION           #
        ############################################################
        # initialize wavefunction as a delta function in each branch
        wf_adb_delta = np.zeros((num_trajs, num_branches, num_states), dtype=complex)
        for nt in range(num_trajs):
            wf_adb_delta[nt][np.arange(num_branches, dtype=int), self.act_surf_ind_0[nt*num_branches:(nt+1)*num_branches]] = 1.0 + 0.j
        self.wf_adb_delta = wf_adb_delta.reshape((num_trajs*num_branches,num_states))
        # transform to diabatic basis
        self.wf_db = auxilliary.psi_adb_to_db_branch(self.wf_adb, self.evecs_branch)
        self.wf_db_delta = auxilliary.psi_adb_to_db_branch(self.wf_adb_delta, self.evecs_branch)
        # initialize quantum force
        self.qfzc = auxilliary.quantum_force_branch(self.evecs_branch, self.act_surf_ind_branch, self.z_coord, sim)
        ############################################################
        #         COHERENT SURFACE HOPPING SPECIFIC INITIALIZATION#
        ############################################################
        # store the phase of each branch
        self.phase_branch = np.zeros(num_trajs * num_branches)
        return
    
    def propagate_classical_subsystem(self, sim):
        self.z_coord = auxilliary.rk4_c(self.z_coord, self.qfzc, sim.dt, sim)
        return
    
    def propagate_quantum_subsystem(self, sim):
        # propagate phases
        self.phase_branch = self.phase_branch + sim.dt * self.evals_branch[
                    np.arange(sim.num_branches * sim.num_trajs, dtype=int), self.act_surf_ind_0]
        # construct eigenvalue exponential
        evals_exp_branch = np.exp(-1.0j * self.evals_branch * sim.dt)
        # transform wavefunctions to adiabatic basis
        self.wf_adb = np.copy(auxilliary.psi_db_to_adb_branch(self.wf_db, self.evecs_branch))
        self.wf_adb_delta = np.copy(auxilliary.psi_db_to_adb_branch(self.wf_db_delta, self.evecs_branch))
        # multiply by propagator
        self.wf_adb = np.copy(evals_exp_branch * self.wf_adb)
        self.wf_adb_delta = np.copy(evals_exp_branch * self.wf_adb_delta)
        # transform back to diabatic basis
        self.wf_db = np.copy(auxilliary.psi_adb_to_db_branch(self.wf_adb, self.evecs_branch))
        self.wf_db_delta = np.copy(auxilliary.psi_adb_to_db_branch(self.wf_adb_delta, self.evecs_branch))
        
        return
    
    def update_state(self, sim):
        num_trajs = sim.num_trajs
        num_states = sim.num_states
        num_branches = sim.num_branches
        self.evecs_branch_previous = np.copy(self.evecs_branch)
        # update the Hamiltonian
        self.h_tot = sim.h_q(sim.h_q_params)[np.newaxis, :, :] + sim.h_qc_branch(sim.h_qc_params, self.z_coord)
        # obtain branch eigenvalues and eigenvectors
        self.evals_branch, self.evecs_branch = np.linalg.eigh(self.h_tot)
        for nt in range(num_trajs):
            # adjust gauge of eigenvectors
            self.evecs_branch[nt * num_branches:(nt + 1) * num_branches], _ = auxilliary.sign_adjust_branch(
                self.evecs_branch[nt * num_branches:(nt + 1) * num_branches], \
                self.evecs_branch_previous[nt * num_branches:(nt + 1) * num_branches],
                self.evals_branch[nt * num_branches:(nt + 1) * num_branches],
                self.z_coord[nt * num_branches:(nt + 1) * num_branches], sim)
        # update branch-pairs if needed
        if sim.cfssh_branch_pair_update == 2 and sim.dmat_const == 1:  # update branch-pairs every bath timestep
            self.evecs_branch_pair_previous = np.copy(self.evecs_branch_pair)
            for nt in range(num_trajs):
                self.evals_branch_pair[nt], self.evecs_branch_pair_previous[nt] = auxilliary.get_branch_pair_eigs(
                    self.z_coord[nt * num_branches:(nt + 1) * num_branches], self.evecs_branch_pair_previous[nt], sim)
        ############################################################
        #                         HOPPING PROCEDURE                #
        ############################################################
        for nt in range(num_trajs):
            rand = self.hopping_probs_rand_vals[nt, self.t_ind]
            for i in range(num_branches):
                # compute hopping probabilities
                prod = np.matmul(np.conjugate(self.evecs_branch[nt * num_branches:(nt + 1) * num_branches][i][:,
                                                self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]), \
                                    self.evecs_branch_previous[nt * num_branches:(nt + 1) * num_branches][i])
                if sim.pab_cohere:
                    hop_prob = -2 * np.real(prod * (self.wf_adb[nt * num_branches:(nt + 1) * num_branches][i] \
                                                    / self.wf_adb[nt * num_branches:(nt + 1) * num_branches][i][
                                                        self.act_surf_ind_branch[
                                                        nt * num_branches:(nt + 1) * num_branches][i]]))
                if not sim.pab_cohere:
                    hop_prob = -2 * np.real(
                        prod * (self.wf_adb_delta[nt * num_branches:(nt + 1) * num_branches][i] \
                                / self.wf_adb_delta[nt * num_branches:(nt + 1) * num_branches][i][
                                    self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]))
                hop_prob[self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]] = 0
                bin_edge = 0
                # hop if possible
                for k in range(len(hop_prob)):
                    hop_prob[k] = auxilliary.nan_num(hop_prob[k])
                    bin_edge = bin_edge + hop_prob[k]
                    if rand < bin_edge:
                        # compute nonadiabatic coupling d_{kj}= <k|\nabla H|j>/(e_{j} - e_{k})
                        evec_k = self.evecs_branch[nt * num_branches:(nt + 1) * num_branches][i][:,
                                    self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]
                        evec_j = self.evecs_branch[nt * num_branches:(nt + 1) * num_branches][i][:, k]
                        eval_k = self.evals_branch[nt * num_branches:(nt + 1) * num_branches][i][
                            self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]]
                        eval_j = self.evals_branch[nt * num_branches:(nt + 1) * num_branches][i][k]
                        ev_diff = eval_j - eval_k
                        # dkj_q is wrt q dkj_p is wrt p.
                        dkj_z, dkj_zc = auxilliary.get_dab(evec_k, evec_j, ev_diff,
                                                            self.z_coord[nt * num_branches:(nt + 1) * num_branches][i],
                                                            sim)
                        ## check that nonadiabatic couplings are real-valued
                        dkj_q = np.sqrt(sim.h * sim.m / 2) * (dkj_z + dkj_zc)
                        dkj_p = np.sqrt(1 / (2 * sim.h * sim.m)) * 1.0j * (dkj_z - dkj_zc)
                        if np.abs(np.sin(np.angle(dkj_q[np.argmax(np.abs(dkj_q))]))) > 1e-2 or \
                                np.abs(np.sin(np.angle(dkj_p[np.argmax(np.abs(dkj_p))]))) > 1e-2:
                            raise Exception('Nonadiabatic coupling is complex, needs gauge fixing!')
                        delta_z = dkj_zc
                        self.z_coord[nt * num_branches:(nt + 1) * num_branches][i], hopped = \
                            sim.hop(sim, self.z_coord[nt * num_branches:(nt + 1) * num_branches][i], delta_z, ev_diff)
                        if hopped:  # adjust active surfaces if a hop has occured
                            self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i] = k
                            self.act_surf_branch[nt * num_branches:(nt + 1) * num_branches][i] = np.zeros_like(
                                self.act_surf_branch[nt * num_branches:(nt + 1) * num_branches][i])
                            self.act_surf_branch[nt * num_branches:(nt + 1) * num_branches][i][
                                self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches][i]] = 1
                        break
        self.qfzc = auxilliary.quantum_force_branch(self.evecs_branch, self.act_surf_ind_branch, self.z_coord, sim)
        return
    
    def calculate_observables(self, sim):
        num_trajs = sim.num_trajs
        num_states = sim.num_states
        num_branches = sim.num_branches

        if sim.cfssh_branch_pair_update == 1 and sim.dmat_const == 1:  # update branch-pairs every output timestep
            self.evecs_branch_pair_previous = np.copy(self.evecs_branch_pair)
            for nt in range(num_trajs):
                self.evals_branch_pair[nt], self.evecs_branch_pair[nt] = auxilliary.get_branch_pair_eigs(
                    self.z_coord[nt * num_branches:(nt + 1) * num_branches], self.evecs_branch_pair_previous[nt], sim)
        # calculate overlap matrix
        overlap = np.zeros((num_trajs, num_branches, num_branches))
        for nt in range(num_trajs):
            overlap[nt] = auxilliary.get_classical_overlap(self.z_coord[nt * num_branches:(nt + 1) * num_branches], sim)
        if sim.dmat_const == 0:
            # Inexpensive density matrix construction
            self.dm_adb = np.zeros((num_trajs, num_branches, num_states, num_states), dtype=complex)
            dm_adb_coh = np.zeros((num_trajs, num_states, num_states), dtype=complex)
            for nt in range(num_trajs):
                phase_branch_nt = self.phase_branch[nt * num_branches:(nt + 1) * num_branches]
                act_surf_ind_branch_nt = self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches]
                act_surf_ind_0_nt = self.act_surf_ind_0[nt * num_branches:(nt + 1) * num_branches]
                act_surf_branch_nt = self.act_surf_branch[nt * num_branches:(nt + 1) * num_branches]
                for i in range(num_branches):
                    for j in range(i + 1, num_branches):
                        a_i = act_surf_ind_branch_nt[i]
                        a_j = act_surf_ind_branch_nt[j]
                        a_i_0 = act_surf_ind_0_nt[i]
                        a_j_0 = act_surf_ind_0_nt[j]
                        if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(
                                self.dm_adb_0_traj[nt][a_i, a_j]) > 1e-12:
                            if sim.sh_deterministic:
                                prob_fac = 1
                            else:
                                prob_fac = 1 / (self.dm_adb_0_traj[nt][a_i, a_i] * self.dm_adb_0_traj[nt][a_j, a_j] * (
                                        num_branches - 1))
                            rho_ij = prob_fac * self.dm_adb_0_traj[nt][a_i, a_j] * overlap[nt][i, j] * \
                                        np.exp(-1.0j * (phase_branch_nt[i] - phase_branch_nt[j]))
                            dm_adb_coh[nt][a_i, a_j] += rho_ij
                            dm_adb_coh[nt][a_j, a_i] += np.conj(rho_ij)
                if sim.sh_deterministic:
                    # construct diagonal of adiaabtic density matrix
                    dm_adb_diag = np.diag(self.dm_adb_0_traj[nt]).reshape(
                        (-1, 1)) * act_surf_branch_nt
                    np.einsum('...jj->...j', self.dm_adb[nt])[...] = dm_adb_diag
                    self.dm_adb[nt] = self.dm_adb[nt] + dm_adb_coh[nt] / num_branches
                else:
                    for n in range(num_branches):
                        self.dm_adb[nt][n, self.act_surf_ind_branch[n], self.act_surf_ind_branch[n]] += 1
                    # add coherences averaged over branches
                    self.dm_adb[nt] = (self.dm_adb[nt] + self.dm_adb_coh[nt] / num_branches) / num_branches
            self.dm_adb = self.dm_adb.reshape(num_trajs * num_branches, num_states,
                                                                num_states)
            self.dm_db_branch = auxilliary.rho_adb_to_db_branch(self.dm_adb, self.evecs_branch)
            self.dm_db = np.sum(self.dm_db_branch, axis=0)
        # expensive CFSSH density matrix construction
        if sim.dmat_const == 1:
            self.evecs_branch_pair_previous = np.copy(self.evecs_branch_pair)
            # assert sim.sh_deterministic == True
            self.dm_adb = np.zeros((num_trajs, num_branches, num_states, num_states), dtype=complex)
            dm_adb_coh = np.zeros((num_trajs, num_states, num_states), dtype=complex)
            dm_db_coh = np.zeros((num_trajs, num_states, num_states), dtype=complex)
            for nt in range(num_trajs):
                dm_adb_coh_ij = np.zeros((num_states, num_states), dtype=complex)
                phase_branch_nt = self.phase_branch[nt * num_branches:(nt + 1) * num_branches]
                act_surf_ind_branch_nt = self.act_surf_ind_branch[nt * num_branches:(nt + 1) * num_branches]
                act_surf_branch_nt = self.act_surf_branch[nt * num_branches:(nt + 1) * num_branches]
                act_surf_ind_0_nt = self.act_surf_ind_0[nt * num_branches:(nt + 1) * num_branches]
                z_branch_nt = self.z_coord[nt * num_branches:(nt + 1) * num_branches]
                for i in range(num_branches):
                    for j in range(i + 1, num_branches):
                        a_i = act_surf_ind_branch_nt[i]
                        a_j = act_surf_ind_branch_nt[j]
                        a_i_0 = act_surf_ind_0_nt[i]
                        a_j_0 = act_surf_ind_0_nt[j]
                        if a_i != a_j and a_i != a_j and a_i == a_i_0 and a_j == a_j_0 and np.abs(
                                self.dm_adb_0_traj[nt][a_i, a_j]) > 1e-12:
                            if sim.cfssh_branch_pair_update == 0:
                                z_branch_ij = np.array([(z_branch_nt[i] + z_branch_nt[j]) / 2])
                                h_tot_branch_ij = self.h_q + sim.h_qc_branch(sim.h_qc_params, z_branch_ij)[0]
                                self.evals_branch_pair[nt][i, j], self.evecs_branch_pair[nt][i, j] = np.linalg.eigh(h_tot_branch_ij)
                                evecs_branch_pair_ij_tmp, _ = auxilliary.sign_adjust_branch(
                                    self.evecs_branch_pair[nt][i, j].reshape(1, num_states, num_states),
                                    self.evecs_branch_pair_previous[nt][i, j].reshape(1, num_states, num_states),
                                    self.evals_branch_pair[nt][i, j].reshape(1, num_states), z_branch_ij, sim)
                                self.evecs_branch_pair[nt][i, j] = np.copy(evecs_branch_pair_ij_tmp[0])
                            if sim.sh_deterministic:
                                prob_fac = 1
                            else:
                                prob_fac = 1 / (self.dm_adb_0_traj[nt][a_i, a_i] * self.dm_adb_0_traj[nt][a_j, a_j] * (
                                        num_branches - 1))
                            coh_ij_tmp = prob_fac * self.dm_adb_0_traj[nt][a_i, a_j] * overlap[nt][i, j] * np.exp(
                                -1.0j * (phase_branch_nt[i] - phase_branch_nt[j]))
                            dm_adb_coh_ij[a_i, a_j] += coh_ij_tmp
                            dm_adb_coh_ij[a_j, a_i] += np.conj(coh_ij_tmp)
                            # transform only the coherence to diabatic basis
                            # rho_db_cfssh_coh_ij = auxilliary.rho_adb_to_db(rho_adb_cfssh_coh_ij, evecs_branch_pair[i,j])
                            dm_db_coh_ij = coh_ij_tmp * np.outer(self.evecs_branch_pair[nt][i, j][:, a_i],
                                                                        np.conj(self.evecs_branch_pair[nt][i, j][:,a_j])) + \
                                                    np.conj(coh_ij_tmp) * np.outer(self.evecs_branch_pair[nt][i, j][:, a_j],
                                                            np.conj(self.evecs_branch_pair[nt][i, j][:, a_i]))
                            # accumulate coherences for each basis
                            dm_db_coh[nt] = dm_db_coh[nt] + dm_db_coh_ij
                            dm_adb_coh[nt] = dm_adb_coh[nt] + dm_adb_coh_ij
                            # reset the matrix to store the individual adiabatic coherences
                            dm_adb_coh_ij = np.zeros((num_states, num_states), dtype=complex)
                # place the active surface on the diagonal weighted by the initial populations
                if sim.sh_deterministic:
                    rho_diag = np.diag(self.dm_adb_0_traj[nt]).reshape((-1, 1)) * act_surf_branch_nt
                    np.einsum('...jj->...j', self.dm_adb[nt])[...] = rho_diag
                else:
                    for n in range(num_branches):
                        self.dm_adb[n, self.act_surf_ind_branch[n], self.act_surf_ind_branch[n]] += 1
            self.dm_adb = self.dm_adb.reshape((num_trajs * num_branches, num_states, num_states))
            self.dm_db_branch = auxilliary.rho_adb_to_db_branch(self.dm_adb, self.evecs_branch).reshape(
                (num_trajs, num_branches, num_states, num_states))
            if sim.sh_deterministic:
                self.dm_db_branch = (self.dm_db_branch + (dm_db_coh[:, np.newaxis, :, :] / num_branches))
            else:
                self.dm_db_branch = (self.dm_db_branch + (dm_db_coh[:, np.newaxis, :, :] / num_branches)) / num_branches
            self.dm_db_branch = self.dm_db_branch.reshape((num_trajs * num_branches, num_states, num_states))
            self.dm_db = np.sum(self.dm_db_branch, axis=0)
        self.observables_t = sim.observables(sim, self)
        eq = 0
        for n in range(len(self.act_surf_ind_branch)):
            eq += self.evals_branch[n][self.act_surf_ind_branch[n]]
        self.observables_t['e_q'] = eq / num_branches
        self.observables_t['e_c'] = np.sum(sim.h_c_branch(sim.h_c_params, self.z_coord))/(sim.num_branches)
        self.observables_t['dm_db'] = self.dm_db
        return 

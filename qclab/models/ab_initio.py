import numpy as np
import pyscf
from pyscf import gto, scf, ci, grad, nac
from functools import reduce
import qclab.auxiliary as auxiliary

class AbInitioModel:
    def __init__(self, input_params):
        self.pyscf_mol = input_params['pyscf_mol']
        self.init_momentum = input_params['init_momentum']
        self.num_states = input_params['num_states']
        self.method = input_params['method']
        self.use_x2c = False#input_params['use_x2c']

        #init_pos = np.array(self.pyscf_mol.atom_coords())#np.array([[0,0,n*self.init_disp] for n in range(self.num_atoms)])
        #init_mom = 0.1*np.array([[0,0,-.1*n*self.init_disp] for n in range(self.num_atoms)])
        self.init_position = np.array(self.pyscf_mol.atom_coords(unit='Angstrom'))
        self.num_atoms = len(self.init_position)
        self.pq_weight = np.ones(3*self.num_atoms)
        self.mass = (np.ones((self.num_atoms, 3))*self.pyscf_mol.atom_mass_list()[:,np.newaxis]).flatten()
        print(self.mass)
        print(self.init_position)
        self.num_classical_coordinates = self.num_atoms * 3

        def dh_qc_dz_UCISD(state, model, params, z_coord, psi_a, psi_b):
            dz_out = np.zeros((params.batch_size, params.num_branches, model.num_classical_coordinates), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    if model.num_states > 1:
                        for n in range(model.num_states):
                            dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * \
                                (myci.Gradients().grad_elec(civec=myci.ci[state.order[traj, branch][n]]).flatten()/0.529177249)*(np.abs(psi_b[traj, branch, n])**2)
                    else:
                        dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * \
                                (myci.Gradients().grad_elec(civec=myci.ci).flatten()/0.529177249)*(np.abs(psi_b[traj, branch, 0])**2)
            return dz_out
        
        def dh_qc_dzc_UCISD(state, model, params, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz_UCISD(state, model, params, z_coord, psi_a, psi_b))

        def dh_qc_dz_CISD(state, model, params, z_coord, psi_a, psi_b):
            dz_out = np.zeros((params.batch_size, params.num_branches, model.num_classical_coordinates), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    if model.num_states > 1:
                        for n in range(model.num_states):
                            dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * \
                                (myci.Gradients().grad_elec(civec=myci.ci[state.order[traj, branch][n]]).flatten()/0.529177249)*(np.abs(psi_b[traj, branch, n])**2)
                    else:
                        dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * \
                                (myci.Gradients().grad_elec(civec=myci.ci).flatten()/0.529177249)*(np.abs(psi_b[traj, branch, 0])**2)
                    #dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * myci.Gradients().grad_nuc().flatten()
            return dz_out# np.zeros((params.num_trajs, params.num_branches, model.num_classical_coordinates), dtype=complex)

        def dh_qc_dzc_CISD(state, model, params, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz_CISD(state, model, params, z_coord, psi_a, psi_b))
        
        def dh_qc_dz_FCI(state, model, params, z_coord, psi_a, psi_b):
            dz_out = np.zeros((params.batch_size, params.num_branches, model.num_classical_coordinates), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    if model.num_states > 1:
                        for n in range(model.num_states):
                            dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * \
                                myci.Gradients().grad_elec(civec=myci.ci[n]).flatten()*(np.abs(psi_b[traj, branch,n])**2)
                    else:
                        dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * \
                                myci.Gradients().grad_elec(civec=myci.ci).flatten()*(np.abs(psi_b[traj, branch, 0])**2)
                    #dz_out[traj, branch] += np.sqrt(1/(2*model.mass * model.pq_weight)) * myci.Gradients().grad_nuc().flatten()
            return dz_out# np.zeros((params.num_trajs, params.num_branches, model.num_classical_coordinates), dtype=complex)

        def dh_qc_dzc_FCI(state, model, params, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz_FCI(state, model, params, z_coord, psi_a, psi_b))


        def h_q(state, model, params):
            out = np.identity(model.num_states)*0
            return out[np.newaxis, np.newaxis, :, :]
        
        def h_qc_UCISD(state, model, params, z_coord):
            out_mat = np.zeros((params.batch_size, params.num_branches, model.num_states, model.num_states), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    myci_prev = state.ab_initio_hams_posthf_prev[traj][branch]
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf_prev = state.ab_initio_hams_mf_prev[traj][branch]
                    s12 = pyscf.gto.intor_cross('cint1e_ovlp_sph', mf.mol, mf_prev.mol)
                    s12 = (reduce(np.dot, (mf.mo_coeff[0].T, s12, mf_prev.mo_coeff[0])),reduce(np.dot, (mf.mo_coeff[1].T, s12, mf_prev.mo_coeff[1])))
                    nmo = mf_prev.mo_energy.size // 2
                    nocc = mf.mol.nelectron // 2
                    if model.num_states > 1:
                        diag_e = myci.e_tot - mf.energy_nuc() + state.energy_offset[traj, branch]
                        for n in range(model.num_states):
                            for m in range(model.num_states):
                                if n == m:
                                    out_mat[traj, branch ,m, n] += diag_e[n]
                                if n != m:
                                    out_mat[traj, branch, m, n] += (-1.0j/params.dt)*(0 - pyscf.ci.ucisd.overlap(myci.ci[state.order[traj, branch][m]], myci_prev.ci[n], (nmo, nmo), (nocc, nocc), s12)) 
                    else:
                        diag_e = myci.e_tot - mf.energy_nuc() + state.energy_offset[traj, branch]
                        out_mat[traj, branch, 0, 0] = diag_e
        
            return (out_mat + np.einsum('ijkl->ijlk',np.conj(out_mat), optimize='greedy'))/2

        def h_qc_CISD(state, model, params, z_coord):
            out_mat = np.zeros((params.batch_size, params.num_branches, model.num_states, model.num_states), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    myci_prev = state.ab_initio_hams_posthf_prev[traj][branch]
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf_prev = state.ab_initio_hams_mf_prev[traj][branch]
                    s12 = pyscf.gto.intor_cross('cint1e_ovlp_sph', mf.mol, mf_prev.mol)
                    s12 = reduce(np.dot, (mf.mo_coeff.T, s12, mf_prev.mo_coeff))
                    nmo = mf_prev.mo_energy.size 
                    nocc = mf.mol.nelectron // 2
                    if model.num_states > 1:
                        diag_e = myci.e_tot - mf.energy_nuc() + state.energy_offset[traj, branch]
                        for n in range(model.num_states):
                            for m in range(model.num_states):
                                if n == m:
                                    out_mat[traj, branch ,m, n] += diag_e[state.order[traj, branch][m]]
                                if n != m:
                                    out_mat[traj, branch, m, n] += (-1.0j/params.dt)*(0 - pyscf.ci.cisd.overlap(myci.ci[state.order[traj, branch][m]], myci_prev.ci[state.order_prev[traj, branch][n]], nmo, nocc, s12)) 
                    else:
                        diag_e = myci.e_tot - mf.energy_nuc() + state.energy_offset[traj, branch]
                        out_mat[traj, branch, 0, 0] = diag_e
            return (out_mat + np.einsum('ijkl->ijlk',np.conj(out_mat), optimize='greedy'))/2
        
        def h_qc_FCI(state, model, params, z_coord):
            out_mat = np.zeros((params.batch_size, params.num_branches, model.num_states, model.num_states), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    myci_prev = state.ab_initio_hams_posthf_prev[traj][branch]
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf_prev = state.ab_initio_hams_mf_prev[traj][branch]
                    eris = pyscf.ao2mo.full(mf.mol, mf.mo_coeff)
                    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
                    norb = mf.mo_energy.size
                    nelec = mf.mol.nelectron
                    #H_fci = pyscf.fci.direct_spin1.pspace(h1, eris, norb, nelec, np=model.num_states)[1]
                    s12 = pyscf.gto.intor_cross('int1e_ovlp', mf.mol, mf_prev.mol)
                    s12 = reduce(np.dot, (mf.mo_coeff.T, s12, mf_prev.mo_coeff))
                    #s12 = mf.mo_coeff.T @ mf.mol.intor('int1e_ovlp') @ mf_prev.mo_coeff
                    #nmo = mf_prev.mo_energy.size 
                    #nocc = mf.mol.nelectron // 2
                    diag_energies = myci.e_tot - mf.energy_nuc()
                    if model.num_states > 1:
                        #overlap_mat = np.zeros((model.num_states, model.num_states), dtype=complex)
                        for n in range(model.num_states):
                            for m in range(model.num_states):
                                #overlap_mat[m, n] = pyscf.ci.cisd.overlap(myci.ci[m], myci_prev.ci[n], nmo, nocc, s12)
                                #overlap_mat[m, n] = pyscf.fci.addons.overlap(myci.ci[m], myci_prev.ci[n], norb, nelec, s12)
                                if n == m:
                                    out_mat[traj, branch ,m, n] += diag_energies[n]
                                #if n == m:
                                #    out_mat[traj, branch, m, n] += (-1.0j/params.dt)*(1 - pyscf.ci.cisd.overlap(myci.ci[m], myci_prev.ci[n], nmo, nocc, s12))
                                if n != m:
                                    out_mat[traj, branch, m, n] += (-1.0j/params.dt)*(0 - pyscf.fci.addons.overlap(myci.ci[m], myci_prev.ci[n], norb, nelec, s12))
                        
                    else:
                        #out_mat[traj, branch, 0, 0] = np.dot(np.conj(myci.ci), myci.contract(myci.ci, eris=eris)) + (-1.0j/params.dt)*(1 - pyscf.fci.addons.overlap(myci.ci[m], myci_prev.ci[n], norb, nelec, s12))
                        out_mat[traj, branch, 0, 0] = diag_energies# + (-1.0j/params.dt)*(1 - pyscf.fci.addons.overlap(myci.ci, myci_prev.ci, norb, nelec, s12))
                            
        
            return (out_mat + np.einsum('ijkl->ijlk',np.conj(out_mat)))/2
        
        def init_classical(model, seed=None):
            q = self.init_position.flatten()
            p = self.init_momentum.flatten()
            z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
            return z
        


        def dh_c_dz(state, model, params, z_coord):
            dz_out = np.zeros((params.batch_size, params.num_branches, model.num_classical_coordinates), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    dz_out[traj, branch] = (np.sqrt(1/(2*model.mass * model.pq_weight))*myci.Gradients().grad_nuc().flatten()/0.529177249) + \
                    (model.pq_weight/2)*(np.conj(z_coord[traj,branch]) - z_coord[traj,branch])
            return dz_out


        def dh_c_dzc(state, model, params, z_coord):
            return np.conj(dh_c_dz(state, model, params, z_coord))

        def h_c(state, model, params, z_coord):
            out = np.zeros((params.batch_size, params.num_branches))
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    mf = state.ab_initio_hams_mf[traj][branch]
                    out[traj, branch] += mf.mol.energy_nuc()*0 # obtian nuclear potential energy
            # add nuclear kinetic energy
            out += np.real(np.sum((model.pq_weight[...,:]/4)*(2*np.conj(z_coord)*z_coord - z_coord*z_coord - np.conj(z_coord)*np.conj(z_coord)),axis=-1))
            return out

        if self.method == 'UCISD':
            self.dh_qc_dz = dh_qc_dz_UCISD
            self.dh_qc_dzc = dh_qc_dzc_UCISD
            self.h_qc = h_qc_UCISD
        if self.method == 'CISD':
            self.dh_qc_dz = dh_qc_dz_CISD
            self.dh_qc_dzc = dh_qc_dzc_CISD
            self.h_qc = h_qc_CISD
        if self.method == 'FCI':
            self.dh_qc_dz = dh_qc_dz_FCI
            self.dh_qc_dzc = dh_qc_dzc_FCI 
            self.h_qc = h_qc_FCI
        #if self.method == 'UCISD':
        #    self.
        self.h_q = h_q
        self.dh_c_dz = dh_c_dz
        self.dh_c_dzc = dh_c_dzc
        self.h_c = h_c
        self.init_classical= init_classical

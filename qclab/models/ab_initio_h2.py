import numpy as np
import pyscf
from pyscf import gto, scf, ci, grad, nac
from functools import reduce
import qclab.auxiliary as auxiliary

class AbInitioH2Model:
    def __init__(self, input_params):
        self.init_disp = input_params['init_disp']
        self.basis = input_params['basis']
        self.num_atoms = input_params['num_atoms']
        self.temp = input_params['temp']  # temperature
        self.num_states = input_params['num_states']

        init_pos = np.array([[0,0,n*self.init_disp] for n in range(self.num_atoms)])
        init_mom = np.array([[0,0,-2*n*self.init_disp] for n in range(self.num_atoms)])

        self.pq_weight = np.ones(3*self.num_atoms)
        self.mass = np.ones(3*self.num_atoms)
        self.num_classical_coordinates = self.num_atoms * 3
        self.w = np.zeros(self.num_classical_coordinates)

        def dh_qc_dz(state, model, params, z_coord, psi_a, psi_b):
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

        def dh_qc_dzc(state, model, params, z_coord, psi_a, psi_b):
            return np.conj(dh_qc_dz(state, model, params, z_coord, psi_a, psi_b))


        def h_q(state, model, params):
            out = np.identity(model.num_states)
            return out[np.newaxis, np.newaxis, :, :]
        def h_qc(state, model, params, z_coord):
            out_mat = np.zeros((params.batch_size, params.num_branches, model.num_states, model.num_states), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    myci_prev = state.ab_initio_hams_posthf_prev[traj][branch]
                    eris = myci.ao2mo()
                    #eris = state.ab_initio_hams_mf[traj][branch].ao2mo()
                    mf = state.ab_initio_hams_mf[traj][branch]
                    #eris = pyscf.ao2mo.full(mf.mol, mf.mo_coeff)
                    mf = state.ab_initio_hams_mf[traj][branch]
                    mf_prev = state.ab_initio_hams_mf_prev[traj][branch]
                    s12 = pyscf.gto.intor_cross('cint1e_ovlp_sph', mf.mol, mf_prev.mol)
                    #print(np.shape(s12), np.shape(mf.mo_coeff), np.shape(mf_prev.mo_coeff))
                    s12 = reduce(np.dot, (mf.mo_coeff.T, s12, mf_prev.mo_coeff))
                    #s12 = np.einsum('nlj,ji,nik->lk',mf.mo_coeff, s12, mf_prev.mo_coeff, optimize='greedy')
                    nmo = mf_prev.mo_energy.size 
                    nocc = mf.mol.nelectron // 2
                    if model.num_states > 1:
                        overlap_mat = np.zeros((model.num_states, model.num_states), dtype=complex)
                        for n in range(model.num_states):
                            for m in range(model.num_states):
                                overlap_mat[m, n] = pyscf.ci.cisd.overlap(myci.ci[m], myci_prev.ci[n], nmo, nocc, s12)
                        #print(np.argmax(np.abs(overlap_mat),axis=1))
                        #for n in range(model.num_states):
                        #    for m in range(model.num_states):
                                out_mat[traj, branch ,m, n] = np.dot(np.conj(myci.ci[m]), myci.contract(myci.ci[n], eris=eris))
                                #if n == m:
                                #    out_mat[traj, branch, m, n] += (-1.0j/params.dt)*(1 - pyscf.ci.cisd.overlap(myci.ci[m], myci_prev.ci[n], nmo, nocc, s12))
                                if n != m:
                                    out_mat[traj, branch, m, n] += (-1.0j/params.dt)*(0 - pyscf.ci.cisd.overlap(myci.ci[m], myci_prev.ci[n], nmo, nocc, s12))
                        #order = np.argmax(np.abs(overlap_mat), axis=1)
                        #print(np.round(np.abs(overlap_mat)**2,1))
                        #print(np.round(np.diag(out_mat[traj,branch]),2))
                        #print(np.round(np.abs(overlap_mat)**2,1)[:,order][order,:])
                        #out_mat[traj,branch] = out_mat[traj, branch][:,order][order,:]
                        #print(np.round(np.diag(out_mat[traj,branch]),2))
                        
                    else:
                        out_mat[traj, branch, 0, 0] = np.dot(np.conj(myci.ci), myci.contract(myci.ci, eris=eris)) + (-1.0j/params.dt)*(1 - pyscf.ci.cisd.overlap(myci.ci, myci_prev.ci, nmo, nocc, s12))
                            
        
            return (out_mat + np.einsum('ijkl->ijlk',np.conj(out_mat)))/2
        
        def init_classical(model, seed=None):
            q = init_pos.flatten()
            p = init_mom.flatten()
            z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
            return z
        


        def dh_c_dz(state, model, params, z_coord):
            dz_out = np.zeros((params.batch_size, params.num_branches, model.num_classical_coordinates), dtype=complex)
            for traj in range(params.batch_size):
                for branch in range(params.num_branches):
                    myci = state.ab_initio_hams_posthf[traj][branch]
                    dz_out[traj, branch] = np.sqrt(1/(2*model.mass * model.pq_weight)) * myci.Gradients().grad_nuc().flatten() + \
                    (model.pq_weight/2)*(np.conj(z_coord[traj,branch]) - z_coord[traj,branch])
            return dz_out


        def dh_c_dzc(state, model, params, z_coord):

            return np.conj(dh_c_dz(state, model, params, z_coord))

        def h_c(state, model, params, z_coord):
            return (model.pq_weight[...,:]/4)*(2*np.conj(z_coord)*z_coord - z_coord*z_coord - np.conj(z_coord)*np.conj(z_coord))



        self.dh_qc_dz = dh_qc_dz
        self.dh_qc_dzc = dh_qc_dzc
        self.h_qc = h_qc
        self.h_q = h_q
        self.dh_c_dz = dh_c_dz
        self.dh_c_dzc = dh_c_dzc
        self.h_c = h_c
        self.init_classical= init_classical

from numba import jit
import numpy as np

@jit(nopython=True)
def rk4_c(q, p, qf, w, dt):
    fq, fp = qf
    k1 = dt * (p + fp)
    l1 = -dt * (w ** 2 * q + fq)  # [wn2] is w_alpha ^ 2
    k2 = dt * ((p + 0.5 * l1) + fp)
    l2 = -dt * (w ** 2 * (q + 0.5 * k1) + fq)
    k3 = dt * ((p + 0.5 * l2) + fp)
    l3 = -dt * (w ** 2 * (q + 0.5 * k2) + fq)
    k4 = dt * ((p + l3) + fp)
    l4 = -dt * (w ** 2 * (q + k3) + fq)
    q = q + 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = p + 0.166667 * (l1 + 2 * l2 + 2 * l3 + l4)
    return q, p



@jit(nopython=True)
def rk4_q(h, psi, dt):
    k1 = (-1j * h.dot(psi))
    k2 = (-1j * h.dot(psi + 0.5 * dt * k1))
    k3 = (-1j * h.dot(psi + 0.5 * dt * k2))
    k4 = (-1j * h.dot(psi + dt * k3))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi

def vec_adb_to_db(psi_adb, eigvec):
    psi_db = np.matmul(eigvec, psi_adb)
    return psi_db


def vec_db_to_adb(psi_db, eigvec):
    psi_adb = np.matmul(np.conjugate(np.transpose(eigvec)),psi_db)
    return psi_adb

@jit(nopython=True)
def rho_0_adb_to_db(rho_0_adb, eigvec): # transforms density matrix from adb to db representation
    rho_0_db = np.dot(np.dot(eigvec, rho_0_adb + 0.0j), np.conj(eigvec).transpose())
    return rho_0_db


@jit(nopython=True)
def rho_0_db_to_adb(rho_0_db, eigvec): # transforms density matrix from db to adb representation
    rho_0_db = np.dot(np.dot(np.conj(eigvec).transpose(), rho_0_db + 0.0j), eigvec)
    return rho_0_db


def get_dab_phase(evals, evecs, dq_vars):
    dabq_phase = np.ones(len(evals), dtype=complex)
    dabp_phase = np.ones(len(evals), dtype=complex)
    for i in range(len(evals)-1):
        j = i + 1
        evec_i = evecs[:, i]
        evec_j = evecs[:, j]
        eval_i = evals[i]
        eval_j = evals[j]
        ev_diff = eval_j - eval_i
        plus = 0
        if np.abs(ev_diff) < 1e-14:
            plus = 1
            print('Warning: Degenerate eigenvalues')
        dkkq, dkkp = get_dab(evec_i, evec_j, ev_diff + plus, dq_vars)
        dkkq_angle = np.angle(dkkq[np.argmax(np.abs(dkkq))])
        dkkp_angle = np.angle(dkkp[np.argmax(np.abs(dkkp))])
        if np.max(np.abs(dkkq)) < 1e-14:
            dkkq_angle = 0
        if np.max(np.abs(dkkp)) < 1e-14:
            dkkp_angle = 0
        dabq_phase[i+1:] = np.exp(1.0j * dkkq_angle) * dabq_phase[i+1:]
        dabp_phase[i+1:] = np.exp(1.0j * dkkp_angle) * dabp_phase[i+1:]
    return dabq_phase, dabp_phase

def h_qc_branch(q_branch, p_branch, h_qc_func , num_branches, num_states):
    out = np.zeros((num_branches, num_states),dtype=complex)
    for i in range(num_branches):
        out[i] = h_qc_func(q_branch, p_branch)
    return out

def get_branch_eigs(q_branch, p_branch, u_ij_previous,h_q_mat, h_qc_func):
    u_ij = np.zeros_like(u_ij_previous)
    num_branches = np.shape(u_ij_previous)[0]
    num_states = np.shape(u_ij_previous)[-1]
    e_ij = np.zeros((num_branches, num_branches, num_states))
    for i in range(num_branches):
        for j in range(i, num_branches):
            branch_mat = h_q_mat + h_qc_func((q_branch[i] + q_branch[j])/2, (p_branch[i] + p_branch[j])/2)
            e_ij[i, j], u_ij[i,j] = np.linalg.eigh(branch_mat)
            e_ij[j,i] = e_ij[i,j]
            u_ij[i,j], _ = sign_adjust(u_ij[i,j], u_ij_previous[i,j], e_ij[i,j])
            u_ij[j,i], _ = u_ij[i, j]
    return e_ij, u_ij


def sign_adjust(evecs, evecs_previous, evals, sim):
    phase_out = np.ones((len(evals)), dtype=complex)
    if sim.gauge_fix >= 1:
        phases = np.exp(-1.0j * np.angle(np.einsum('jk,jk->k', np.conjugate(evecs_previous), evecs)))
        evecs = np.einsum('jk,k->jk', evecs, phases)
        phase_out *= phases
    if sim.gauge_fix >= 2:
        dabQ_phase_list, dabP_phase_list = get_dab_phase(evecs, evals, sim.dq_vars)
        dab_phase_list = np.conjugate(dabQ_phase_list)
        phase_out *= dab_phase_list
        evecs = np.einsum('jk,k->jk', evecs, dab_phase_list)
    if sim.gauge_fix >= 0:
        signs = np.sign(np.einsum('jk,jk->k', np.conjugate(evecs_previous), evecs))
        evecs = np.einsum('jk,k->jk', evecs, signs)
        phase_out *= signs
    return evecs, phase_out

@jit(nopython=True)
def matprod_sparse(shape, ind, mels, vec1, vec2): # calculates <1|mat|2>
    i_ind, j_ind, k_ind = ind
    prod = np.conj(vec1)[j_ind]*mels*vec2[k_ind]
    out_mat = np.zeros((shape[0])) + 0.0j
    for i in range(len(i_ind)):
        out_mat[i_ind[i]] += prod[i]
    return out_mat
def quantum_force(psi,dq_vars): # computes <\psi|\nabla H|\psi> using sparse methods
    (dq_shape, dq_ind, dq_mels, dp_shape, dp_ind, dp_mels) = dq_vars
    fq = matprod_sparse(dq_shape, dq_ind, dq_mels, psi, psi)
    fp = matprod_sparse(dp_shape, dp_ind, dp_mels, psi, psi)
    return fq, fp

def get_dab(evec_a, evec_b, ev_diff, dq_vars):  # computes d_{ab} using sparse methods
    (dq_shape, dq_ind, dq_mels, dp_shape, dp_ind, dp_mels) = dq_vars
    dkkq = matprod_sparse(dq_shape, dq_ind, dq_mels, evec_a, evec_b)/ev_diff
    dkkp = matprod_sparse(dp_shape, dp_ind, dp_mels, evec_a, evec_b)/ev_diff
    return dkkq, dkkp

@jit(nopython=True)
def nan_num(num):
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num

nan_num_vec = np.vectorize(nan_num)
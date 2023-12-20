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
def rk4_c(h, psi, dt):
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

def get_dkk(evec_i, evec_j, ev_diff, sim):  # computes dkk_{ij} using sparse methods

def get_dab_phase(evals, evecs, sim):
    dabq_phase = np.ones(len(evals))
    dabp_phase = np.ones(len(evals))
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
        dkkq, dkkp = get_dkk(evec_i, evec_j, ev_diff + plus, sim)
        dkkq_angle = np.angle(dkkq[np.argmax(np.abs(dkkq))])
        dkkp_angle = np.angle(dkkp[np.argmax(np.abs(dkkp))])
        if np.max(np.abs(dkkq)) < 1e-14:
            dkkq_angle = 0
        if np.max(np.abs(dkkp)) < 1e-14:
            dkkp_angle = 0
        dabq_phase[i+1:] = np.exp(1.0j * dkkq_angle) * dabq_phase[i+1:]
        dabp_phase[i+1:] = np.exp(1.0j * dkkp_angle) * dabp_phase[i+1:]
    return dabq_phase, dabp_phase
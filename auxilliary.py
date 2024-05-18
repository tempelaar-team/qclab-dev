from numba import jit
import numpy as np


def rk4_c(z, qfzc, dt, sim):
    k1 = -1.0j*(sim.dh_c_dzc(z,sim) + qfzc)
    k2 = -1.0j*(sim.dh_c_dzc(z + 0.5*dt*k1, sim) + qfzc)
    k3 = -1.0j*(sim.dh_c_dzc(z + 0.5*dt*k2, sim) + qfzc)
    k4 = -1.0j*(sim.dh_c_dzc(z + dt*k3, sim) + qfzc)
    z = z + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z



#@jit(nopython=True)
def rk4_q(h, psi, dt):
    """
    4-th order Runge-Kutta for quantum wavefunction, works with branch wavefunctions
    :param h: Hamiltonian h(t)
    :param psi: wavefunction psi(t)
    :param dt: timestep dt
    :return: psi(t+dt)
    """
    k1 = (-1j * np.matmul(h, psi))
    k2 = (-1j * np.matmul(h, psi + 0.5 * dt * k1))
    k3 = (-1j * np.matmul(h, psi + 0.5 * dt * k2))
    k4 = (-1j * np.matmul(h, psi + dt * k3))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi


def vec_0_adb_to_db(psi_adb, eigvec):
    """
    Transforms a vector in adiabatic basis to diabatic basis
    psi_{db} = V psi_{adb}
    :param psi_adb: adiabatic vector psi_{adb}
    :param eigvec: eigenvectors V
    :return: diabatic vector psi_{db}
    """
    psi_db = np.matmul(eigvec, psi_adb)
    return psi_db


def vec_0_db_to_adb(psi_db, eigvec):
    """
    Transforms a vector in diabatic basis to adiabatic basis
    psi_{adb} = V^{dagger}psi_{db}
    :param psi_db: diabatic vector psi_{db}
    :param eigvec: eigenvectors V
    :return: adiabatic vector psi_{adb}
    """
    psi_adb = np.matmul(np.conjugate(np.transpose(eigvec)), psi_db)
    return psi_adb

def vec_db_to_adb(psi_db_branch, eigvec_branch):
    """
    Transforms a vector in diabatic basis to adiabatic basis
    psi_{adb} = V^{dagger}psi_{db}
    :param psi_db: diabatic vector psi_{db}
    :param eigvec: eigenvectors V
    :return: adiabatic vector psi_{adb}
    """
    #psi_adb = np.matmul(np.conjugate(np.transpose(eigvec)), psi_db)
    psi_adb_branch = np.einsum('nba,nb->na',np.conjugate(eigvec_branch), psi_db_branch, optimize='greedy')
    return psi_adb_branch

def vec_adb_to_db(psi_adb_branch, eigvec_branch):
    psi_db_branch = np.einsum('nab,nb->na', eigvec_branch, psi_adb_branch, optimize='greedy')
    return psi_db_branch

def rho_adb_to_db(rho_0_adb, eigvec):
    return np.einsum('nij,njk,nlk->nil', eigvec, rho_0_adb, np.conj(eigvec), optimize='greedy')

def rho_db_to_adb(rho_0_db, eigvec):
    return np.einsum('nji,njk,nkl->nil', np.conj(eigvec), rho_0_db, eigvec, optimize='greedy')

@jit(nopython=True)
def rho_0_adb_to_db(rho_0_adb, eigvec):  # transforms density matrix from adb to db representation
    """
    Transforms a density matrix rho_{adb} from adiabatic to diabatic basis:
    rho_{db} = Vrho_{adb}V^{dagger}
    :param rho_0_adb: adiabatic density matrix rho_{adb}
    :param eigvec: eigenvectors V
    :return: diabatic density matrix rho_{db}
    """
    rho_0_db = np.dot(np.dot(eigvec, rho_0_adb + 0.0j), np.conj(eigvec).transpose())
    return rho_0_db


@jit(nopython=True)
def rho_0_db_to_adb(rho_0_db, eigvec):  # transforms density matrix from db to adb representation
    """
    Transforms a density matrix rho_{db} from diabatic to adiabatic basis:
    rho_{adb} = V^{dagger}rho_{db}V
    :param rho_0_db: diabatic density matrix rho_{db}
    :param eigvec: eigenvectors V
    :return:  adiabatic density matrix rho_{adb}
    """
    rho_0_db = np.dot(np.dot(np.conj(eigvec).transpose(), rho_0_db + 0.0j), eigvec)
    return rho_0_db


def get_dab_phase(evals, evecs, z, sim):
    """
    Computes the diagonal gauge transformation G such that (VG)^{dagger}\nabla(VG) is real-valued. :param evals:
    eigenvalues :param evecs: eigenvectors (V) :param diff_vars: sparse matrix variables of \nabla_{z} H and \nabla_{
    zc} H (stored in sim.dq_vars) :return: dabq_phase (diag(G^{dagger}) calculated using d_{ab}^{q}), dabp_phase (
    diag(G^{dagger}) calculated using d_{ab}^{p})
    """
    dabq_phase = np.ones(len(evals), dtype=complex)
    dabp_phase = np.ones(len(evals), dtype=complex)
    for i in range(len(evals) - 1):
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
        dkk_z, dkk_zc = get_dab(evec_i, evec_j, ev_diff + plus, z, sim)
        # convert to q/p nonadiabatic couplings
        dkkq = np.sqrt(sim.h * sim.m / 2) * (dkk_z + dkk_zc)
        dkkp = np.sqrt(1 / (2*sim.h*sim.m)) * 1.0j * (dkk_z - dkk_zc)
        dkkq_angle = np.angle(dkkq[np.argmax(np.abs(dkkq))])
        dkkp_angle = np.angle(dkkp[np.argmax(np.abs(dkkp))])
        if np.max(np.abs(dkkq)) < 1e-14:
            dkkq_angle = 0
        if np.max(np.abs(dkkp)) < 1e-14:
            dkkp_angle = 0
        dabq_phase[i + 1:] = np.exp(1.0j * dkkq_angle) * dabq_phase[i + 1:]
        dabp_phase[i + 1:] = np.exp(1.0j * dkkp_angle) * dabp_phase[i + 1:]
    return dabq_phase, dabp_phase

def h_qc_branch(z_branch, sim):
    """
    evaluates h_qc over each branch
    :param z_branch:
    :param sim:
    :return:
    """
    out = np.zeros((sim.num_branches, sim.num_states, sim.num_states), dtype=complex)
    for i in range(sim.num_branches):
        out[i] += sim.h_qc(z_branch[i], sim)
    return out



def get_branch_pair_eigs(z_branch, u_ij_previous, h_q_mat, sim):
    u_ij = np.zeros_like(u_ij_previous)
    num_branches = np.shape(u_ij_previous)[0]
    num_states = np.shape(u_ij_previous)[-1]
    e_ij = np.zeros((num_branches, num_branches, num_states))
    for i in range(num_branches):
        for j in range(i, num_branches):
            branch_mat = h_q_mat + sim.h_qc((z_branch[i] + z_branch[j]) / 2, sim)
            e_ij[i, j], u_ij[i, j] = np.linalg.eigh(branch_mat)
            e_ij[j, i] = e_ij[i, j]
            u_ij[i, j], _ = sign_adjust(u_ij[i, j], u_ij_previous[i, j], e_ij[i, j], sim)
            u_ij[j, i], _ = u_ij[i, j]
    return e_ij, u_ij


def get_classical_overlap(z_branch, sim):
    out_mat = np.zeros((len(z_branch), len(z_branch)))
    zc_branch = np.conjugate(z_branch)
    q_branch = (1/np.sqrt(2*sim.m*sim.h))*(z_branch + zc_branch)
    p_branch = -1.0j*np.sqrt(sim.h*sim.m/2)*(z_branch - zc_branch)
    for i in range(len(z_branch)):
        for j in range(len(z_branch)):
            out_mat[i,j] = np.exp(-(1/2)*np.sum(np.abs((p_branch[i] - p_branch[j]) * (q_branch[i] - q_branch[j]))))
    return out_mat


def sign_adjust_branch(evecs_branch, evecs_branch_previous, evals_branch, z_branch, sim):
    phase_out = np.ones((len(evecs_branch), len(evecs_branch[0])), dtype=complex)
    if sim.gauge_fix >= 1:
        phases = np.exp(-1.0j*np.angle(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch)))
        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,phases)
        phase_out *= phases
    if sim.gauge_fix >= 2:
        dab_phase_mat = np.ones((len(evecs_branch),len(evecs_branch)),dtype=complex)
        for i in range(len(evecs_branch)):
            dabQ_phase_list, dabP_phase_list = get_dab_phase(evecs_branch[i], evals_branch[i], z_branch[i], sim)
            dab_phase_list = np.conjugate(dabQ_phase_list)
            dab_phase_mat[i] = dab_phase_list
            phase_out[i] *= dab_phase_list
        #    evecs_branch[i] = np.einsum('jk,k->jk',evecs_branch[i],dab_phase_list)
        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,dab_phase_mat)
    if sim.gauge_fix >= 0:
        signs = np.sign(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch))

        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,signs)
        phase_out *= signs
    return evecs_branch, phase_out

@jit(nopython=True)
def matprod_sparse(shape, ind, mels, vec1, vec2):  # calculates <1|mat|2>
    """
    Computes the expectation value f_{i} = <1|H^{i}_{jk}|2>
    where H^{i}_{jk} is a tensor (like \nabla_{i} H_{jk} )
    :param shape: shape of H tensor
    :param ind: list of i,j,k coordinates where H tensor is nonzero
    :param mels: list of nonzero values of H tensor
    :param vec1: |1>
    :param vec2: |2>
    :return: f_{i}
    """
    i_ind, j_ind, k_ind = ind
    prod = np.conj(vec1)[j_ind] * mels * vec2[k_ind]
    out_mat = np.zeros((shape[0])) + 0.0j
    for i in range(len(i_ind)):
        out_mat[i_ind[i]] += prod[i]
    return out_mat


def quantum_force(psi, z, sim):  # computes <\psi|\nabla H|\psi> using sparse methods
    """
    Computes the Hellman-Feynmann force using the formula
    f_{q(p)} = <psi| \nabla_{q(p)} H |psi>
    :param psi: |psi>
    :param diff_vars: sparse matrix variables of \nabla_{z} H and \nabla_{zc} H (stored in sim.diff_vars)
    :return: f_{z} and f_{zc}
    """
    #(dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels) = diff_vars
    #fz = sim.dh_qc_dz(psi, psi, z)#matprod_sparse(dz_shape, dz_ind, dz_mels, psi, psi)
    fzc = sim.dh_qc_dzc(psi, psi, z, sim)#matprod_sparse(dzc_shape, dzc_ind, dzc_mels, psi, psi)
    return fzc

def quantum_force_branch(evecs_branch, act_surf_ind_branch, z_branch, sim):
    fzc_branch = np.zeros(np.shape(z_branch), dtype=complex)
    if act_surf_ind_branch is None:
        for i in range(len(evecs_branch)):
            fzc_branch[i] = quantum_force(evecs_branch[i], z_branch[i], sim)
    else:
        for i in range(len(evecs_branch)):
            fzc_branch[i] = quantum_force(evecs_branch[i][:,act_surf_ind_branch[i]], z_branch[i], sim)
    return fzc_branch


def get_dab(evec_a, evec_b, ev_diff, z, sim):  # computes d_{ab} using sparse methods
    """
    Computes the nonadiabatic coupling using the formula
    d_{ab}^{z(zc)} = <a|\nabla_{z(zc)} H|b>/(e_{b} - e_{a})
    :param evec_a: |a>
    :param evec_b: |b>
    :param ev_diff: e_{b} - e_{a}
    :param diff_vars: sparse matrix variables of \nabla_{z} H and \nabla_{zc} H (stored in sim.diff_vars)
    :return: d_{ab}^{z} and d_{ab}^{zc}
    """
    #(dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels) = diff_vars
    dab_z = sim.dh_qc_dz(evec_a, evec_b, z, sim) / ev_diff#matprod_sparse(dz_shape, dz_ind, dz_mels, evec_a, evec_b) / ev_diff
    dab_zc = sim.dh_qc_dzc(evec_a, evec_b, z, sim) / ev_diff#matprod_sparse(dzc_shape, dzc_ind, dzc_mels, evec_a, evec_b) / ev_diff
    return dab_z, dab_zc


@jit(nopython=True)
def nan_num(num):
    """
    converts nan to a large or small number using numba acceleration
    """
    if np.isnan(num):
        return 0.0
    if num == np.inf:
        return 100e100
    if num == -np.inf:
        return -100e100
    else:
        return num


# vectorized form of nan_num
nan_num_vec = np.vectorize(nan_num)


def add_dictionary(dict_1, dict_2):
    # adds entries of dict_2 to dict_1
    keys_1 = dict_1.keys()
    keys_2 = dict_2.keys()
    for nk in len(keys_2):
        if keys_2[nk] in keys_1:
            dict_1[keys_2[nk]] += dict_2[keys_2[nk]]
        else:
            dict_1[keys_2[nk]] = dict_2[keys_2[nk]]
    return dict_1
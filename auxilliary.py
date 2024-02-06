from numba import jit
import numpy as np


def rk4_c(z, zc, qf, dt, sim):
    fz, fzc = qf
    # k values evolve z
    # l values evolve zc
    k1 = -1.0j*(sim.dh_c_dzc(z,zc) + fzc)
    l1 = +1.0j*(sim.dh_c_dz(z,zc)  + fz )
    k2 = -1.0j*(sim.dh_c_dzc(z + 0.5*dt*k1, zc + 0.5*dt*l1) + fzc)
    l2 = +1.0j*(sim.dh_c_dz(z + 0.5*dt*k1, zc + 0.5*dt*l1) + fz)
    k3 = -1.0j*(sim.dh_c_dzc(z + 0.5*dt*k2, zc + 0.5*dt*l2) + fzc)
    l3 = +1.0j*(sim.dh_c_dz(z + 0.5*dt*k2, zc + 0.5*dt*l2) + fz)
    k4 = -1.0j*(sim.dh_c_dzc(z + dt*k3, zc + dt*l3) + fzc)
    l4 = +1.0j*(sim.dh_c_dz(z + dt*k3, zc + dt*l3) + fz)
    z = z + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    zc = zc + dt * 0.166667 * (l1 + 2 * l2 + 2 * l3 + l4)
    return z, zc


@jit(nopython=True)
def rk4_c_old(z, zc, qf, m, h, dt):
    """
    4-th order Runge-Kutta for classical coordinates
    :param z: complex coordinate z
    :param zc: conjugate coordinate zc
    :param qf: tuple of quantum forces qf = (fz, fzc)
    :param m: mass of each coordinate
    :param h: auxilliary frequency of each coordinate
    :param dt: timestep dt
    :return: z(t+dt), zc(t+dt)
    """
    fz, fzc = qf
    # convert fz and fzc to fq and fp
    fq = np.real(np.sqrt((m*h) / 2) * (fz + fzc))
    fp = np.real(1j * np.sqrt(1 / (2*(m*h))) * (fz - fzc))
    # fq and fp are derivatives of h_qc wrt q and p respectively

    q = np.real((z + zc) / np.sqrt(2*m*h))
    p = np.real(-1.0j * (z - zc) * np.sqrt(m*h / 2))
    k1 = dt * (p + fp)
    l1 = -dt * (w**2 * q + fq)  # [wn2] is w_alpha ^ 2
    k2 = dt * ((p + 0.5 * l1) + fp)
    l2 = -dt * (w**2 * (q + 0.5 * k1) + fq)
    k3 = dt * ((p + 0.5 * l2) + fp)
    l3 = -dt * (w**2 * (q + 0.5 * k2) + fq)
    k4 = dt * ((p + l3) + fp)
    l4 = -dt * (w**2 * (q + k3) + fq)
    q = q + 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    p = p + 0.166667 * (l1 + 2 * l2 + 2 * l3 + l4)
    return np.sqrt(w / 2) * (q + 1.0j * (p/w)), np.sqrt(w / 2) * (q - 1.0j * (p/w))


@jit(nopython=True)
def rk4_q(h, psi, dt):
    """
    4-th order Runge-Kutta for quantum wavefunction
    :param h: Hamiltonian h(t)
    :param psi: wavefunction psi(t)
    :param dt: timestep dt
    :return: psi(t+dt)
    """
    k1 = (-1j * h.dot(psi))
    k2 = (-1j * h.dot(psi + 0.5 * dt * k1))
    k3 = (-1j * h.dot(psi + 0.5 * dt * k2))
    k4 = (-1j * h.dot(psi + dt * k3))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi


def vec_adb_to_db(psi_adb, eigvec):
    """
    Transforms a vector in adiabatic basis to diabatic basis
    psi_{db} = V psi_{adb}
    :param psi_adb: adiabatic vector psi_{adb}
    :param eigvec: eigenvectors V
    :return: diabatic vector psi_{db}
    """
    psi_db = np.matmul(eigvec, psi_adb)
    return psi_db


def vec_db_to_adb(psi_db, eigvec):
    """
    Transforms a vector in diabatic basis to adiabatic basis
    psi_{adb} = V^{dagger}psi_{db}
    :param psi_db: diabatic vector psi_{db}
    :param eigvec: eigenvectors V
    :return: adiabatic vector psi_{adb}
    """
    psi_adb = np.matmul(np.conjugate(np.transpose(eigvec)), psi_db)
    return psi_adb


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


def get_dab_phase(evals, evecs, sim):
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
        dkk_z, dkk_zc = get_dab(evec_i, evec_j, ev_diff + plus, sim.diff_vars)
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


def h_tot_branch(z_branch, zc_branch, h_q, h_qc_func, num_branches, num_states):
    """
    evaluates h_qc_func over each branch
    """
    out = np.zeros((num_branches, num_states, num_states), dtype=complex)
    for i in range(num_branches):
        out[i] = h_q + h_qc_func(z_branch[i], zc_branch[i])
    return out


def get_branch_pair_eigs(z_branch, zc_branch, u_ij_previous, h_q_mat, sim):
    u_ij = np.zeros_like(u_ij_previous)
    num_branches = np.shape(u_ij_previous)[0]
    num_states = np.shape(u_ij_previous)[-1]
    e_ij = np.zeros((num_branches, num_branches, num_states))
    for i in range(num_branches):
        for j in range(i, num_branches):
            branch_mat = h_q_mat + sim.h_qc((z_branch[i] + z_branch[j]) / 2, (zc_branch[i] + zc_branch[j]) / 2)
            e_ij[i, j], u_ij[i, j] = np.linalg.eigh(branch_mat)
            e_ij[j, i] = e_ij[i, j]
            u_ij[i, j], _ = sign_adjust(u_ij[i, j], u_ij_previous[i, j], e_ij[i, j], sim)
            u_ij[j, i], _ = u_ij[i, j]
    return e_ij, u_ij


def get_branch_eigs(z_branch, zc_branch, evecs_previous, h_q_mat, sim):
    num_branches = np.shape(evecs_previous)[0]
    num_states = np.shape(evecs_previous)[1]
    evals_branch, evecs_branch = np.linalg.eigh(h_tot_branch(z_branch, zc_branch, h_q_mat, sim.h_qc,num_branches,num_states))
    evecs_branch, evecs_phases = sign_adjust_branch(evecs_branch, evecs_previous, evals_branch, sim)
    return evals_branch, evecs_branch, evecs_phases

def get_classical_overlap(z_branch, zc_branch, sim):
    out_mat = np.zeros((len(z_branch), len(z_branch)))
    q_branch = (1/np.sqrt(2*sim.m*sim.h))*(z_branch + zc_branch)
    p_branch = -1.0j*np.sqrt(sim.h*sim.m/2)*(z_branch - zc_branch)
    for i in range(len(z_branch)):
        for j in range(len(z_branch)):
            out_mat[i,j] = np.exp(-(1/2))*np.sum(np.abs((p_branch[i] - p_branch[j]) * (q_branch[i] - q_branch[j])))
    return out_mat


def sign_adjust(evecs, evecs_previous, evals, sim):
    """
    Adjusts the gauge of eigenvectors at a t=t to enforce parallel transport with respect to t=t-dt
    using different levels of accuracy.
    gauge_fix == 0 -- adjust the overall sign so that <a(t-dt)|a(t)> is positive
    gauge_fix == 1 -- adjust the phase so that <a(t-dt)|a(t)> is real-valued
    gauge_fix == 2 -- computes the gauge transformation that ensures d_{ab}(t) is real-valued
    :param evecs: eigenvectors at t=t
    :param evecs_previous: eigenvectors at t=t-dt
    :param evals: eigenvalues at t=t (only for gauge_fix==2)
    :param sim: simulation object
    :return: eigenvectors at time t=t satisfying parallel transport
    """
    phase_out = np.ones((len(evals)), dtype=complex)
    if sim.gauge_fix >= 1:
        phases = np.exp(-1.0j * np.angle(np.einsum('jk,jk->k', np.conjugate(evecs_previous), evecs)))
        evecs = np.einsum('jk,k->jk', evecs, phases)
        phase_out *= phases
    if sim.gauge_fix >= 2:
        dab_q_phase_list, dab_p_phase_list = get_dab_phase(evecs, evals, sim)
        dab_phase_list = np.conjugate(dab_q_phase_list)
        phase_out *= dab_phase_list
        evecs = np.einsum('jk,k->jk', evecs, dab_phase_list)
    if sim.gauge_fix >= 0:
        signs = np.sign(np.einsum('jk,jk->k', np.conjugate(evecs_previous), evecs))
        evecs = np.einsum('jk,k->jk', evecs, signs)
        phase_out *= signs
    return evecs, phase_out

def sign_adjust_branch(evecs_branch, evecs_branch_previous, evals_branch, sim):
    phase_out = np.ones((len(evecs_branch), len(evecs_branch)), dtype=complex)
    if sim.gauge_fix >= 1:
        phases = np.exp(-1.0j*np.angle(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch)))
        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,phases)
        phase_out *= phases
    if sim.gauge_fix >= 2:
        dab_phase_mat = np.ones((len(evecs_branch),len(evecs_branch)),dtype=complex)
        for i in range(len(evecs_branch)):
            dabQ_phase_list, dabP_phase_list = get_dab_phase(evecs_branch[i], evals_branch[i], sim)
            dab_phase_list = np.conjugate(dabQ_phase_list)
            dab_phase_mat[i] = dab_phase_list
            phase_out[i] *= dab_phase_list
        #    evecs_branch[i] = np.einsum('jk,k->jk',evecs_branch[i],dab_phase_list)
        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,dab_phase_mat)
    if sim.gauge_fix >= 0:
        signs = np.sign(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch))

        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,signs)
        phase_out *= signs
    #for i in range(nbranches):
    #    evecs_branch[i] = operators.sign_adjust(evecs_branch[i],evecs_branch_previous[i], gauge_ind)
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


def quantum_force(psi, z, zc, sim):  # computes <\psi|\nabla H|\psi> using sparse methods
    """
    Computes the Hellman-Feynmann force using the formula
    f_{q(p)} = <psi| \nabla_{q(p)} H |psi>
    :param psi: |psi>
    :param diff_vars: sparse matrix variables of \nabla_{z} H and \nabla_{zc} H (stored in sim.diff_vars)
    :return: f_{z} and f_{zc}
    """
    #(dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels) = diff_vars
    fz = sim.dh_qc_dz(psi, psi, z, zc)#matprod_sparse(dz_shape, dz_ind, dz_mels, psi, psi)
    fzc = sim.dh_qc_dzc(psi, psi, z, zc)#matprod_sparse(dzc_shape, dzc_ind, dzc_mels, psi, psi)
    return fz, fzc

def quantum_force_branch(evecs_branch, act_surf_ind_branch, z, zc, sim):
    #(dz_shape, dz_ind, dz_mels, dzc_shape, dzc_ind, dzc_mels) = diff_vars
    fz_branch = np.zeros((len(evecs_branch), len(z)), dtype=complex)
    fzc_branch = np.zeros((len(evecs_branch), len(z)), dtype=complex)
    for i in range(len(evecs_branch)):
        fz_branch[i], fzc_branch[i] = quantum_force(evecs_branch[i][:,act_surf_ind_branch[i]], z, zc, sim)
    return fz_branch, fzc_branch


def get_dab(evec_a, evec_b, ev_diff, z, zc, sim):  # computes d_{ab} using sparse methods
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
    dab_z = sim.dh_qc_dz(evec_a, evec_b, z, zc) / ev_diff#matprod_sparse(dz_shape, dz_ind, dz_mels, evec_a, evec_b) / ev_diff
    dab_zc = sim.dh_qc_dzc(evec_a, evec_b, z, zc) / ev_diff#matprod_sparse(dzc_shape, dzc_ind, dzc_mels, evec_a, evec_b) / ev_diff
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

def prop_phase(phase_branch, evals_branch, dt):
    """
    Propagates the complex-phase of each branch
    :param phase_branch: list of phases(t)
    :param evals_branch: eigenvalues of each branch
    :return: phases(t+dt)
    """
    phase_out = phase_branch + dt*np.diag(evals_branch)
    return phase_out
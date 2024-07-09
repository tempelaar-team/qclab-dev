from numba import njit
import numpy as np


############################################################
#                       RUNGE-KUTTA                       #
############################################################


def rk4_c(z_branch, qfzc_branch, dt, sim):
    """ 4-th order Runge-Kutta integrator for the z_branch coordinate with force qfzc_branch"""
    k1 = -1.0j*(sim.dh_c_dzc_branch(sim.h_c_params, z_branch) + qfzc_branch)
    k2 = -1.0j*(sim.dh_c_dzc_branch(sim.h_c_params, z_branch + 0.5*dt*k1) + qfzc_branch)
    k3 = -1.0j*(sim.dh_c_dzc_branch(sim.h_c_params, z_branch + 0.5*dt*k2) + qfzc_branch)
    k4 = -1.0j*(sim.dh_c_dzc_branch(sim.h_c_params, z_branch + dt*k3) + qfzc_branch)
    z_branch = z_branch + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    #k1 = -1.0j*(np.apply_along_axis(sim.dh_c_dzc, 1, z_branch) + qfzc_branch)
    #k2 = -1.0j*(np.apply_along_axis(sim.dh_c_dzc, 1, z_branch + 0.5*dt*k1) + qfzc_branch)
    #k3 = -1.0j*(np.apply_along_axis(sim.dh_c_dzc, 1, z_branch + 0.5*dt*k2) + qfzc_branch)
    #k4 = -1.0j*(np.apply_along_axis(sim.dh_c_dzc, 1, z_branch + dt*k3) + qfzc_branch)
    #z_branch = z_branch + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)


    return z_branch

@njit
def rk4_q_branch(h_branch, psi_branch, dt):
    psi_branch_out = np.zeros(np.shape(psi_branch)) + 0.0j
    for n in range(len(psi_branch)):
        psi_branch_out[n] = rk4_q(h_branch[n], psi_branch[n], dt)
    return psi_branch_out


@njit
def rk4_q(h, psi, dt, path=None):
    """
    4-th order Runge-Kutta for quantum wavefunction, works with branch wavefunctions
    :param h: Hamiltonian h(t)
    :param psi: wavefunction psi(t)
    :param dt: timestep dt
    :return: psi(t+dt)
    """
    k1 = (-1j * np.dot(h, psi))
    k2 = (-1j * np.dot(h, psi + 0.5 * dt * k1))
    k3 = (-1j * np.dot(h, psi + 0.5 * dt * k2))
    k4 = (-1j * np.dot(h, psi + dt * k3))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi



############################################################
#                   BASIS TRANSFORMATIONS                  #
############################################################

def psi_adb_to_db(psi_adb, eigvec):
    """
    Transforms a vector in adiabatic basis to diabatic basis
    psi_{db} = V psi_{adb}
    :param psi_adb: adiabatic vector psi_{adb}
    :param eigvec: eigenvectors V
    :return: diabatic vector psi_{db}
    """
    psi_db = np.matmul(eigvec, psi_adb)
    return psi_db

def psi_db_to_adb(psi_db, eigvec):
    """
    Transforms a vector in diabatic basis to adiabatic basis
    psi_{adb} = V^{dagger}psi_{db}
    :param psi_db: diabatic vector psi_{db}
    :param eigvec: eigenvectors V
    :return: adiabatic vector psi_{adb}
    """
    psi_adb = np.matmul(np.conjugate(np.transpose(eigvec)), psi_db)
    return psi_adb

@njit
def psi_adb_to_db_branch(psi_adb_branch, eigvec_branch): # transforms branch adibatic to diabatic basis
    psi_db_branch = np.ascontiguousarray(np.zeros(np.shape(psi_adb_branch))) + 0.0j
    for i in range(len(eigvec_branch)):
        psi_db_branch[i] = np.dot(eigvec_branch[i], psi_adb_branch[i] + 0.0j)
    return psi_db_branch

@njit
def psi_db_to_adb_branch(psi_db_branch, eigvec_branch): # transforms branch adibatic to diabatic basis
    psi_adb_branch = np.ascontiguousarray(np.zeros(np.shape(psi_db_branch))) + 0.0j
    for i in range(len(eigvec_branch)):
        psi_adb_branch[i] = np.dot(np.conj(eigvec_branch[i]).transpose(), psi_db_branch[i] + 0.0j)
    return psi_adb_branch
@njit
def rho_adb_to_db(rho_adb, eigvec):
    return np.dot(np.dot(eigvec, rho_adb + 0.0j), np.conj(eigvec).transpose())

@njit
def rho_db_to_adb(rho_db, eigvec):
    return np.dot(np.dot(np.conj(eigvec).transpose(), rho_db + 0.0j), eigvec)

@njit
def rho_adb_to_db_branch(rho_adb_branch, eigvec_branch): # transforms branch adibatic to diabatic basis
    rho_db_branch = np.ascontiguousarray(np.zeros(np.shape(eigvec_branch))) + 0.0j
    for i in range(len(eigvec_branch)):
        rho_db_branch[i] = np.dot(np.dot(eigvec_branch[i], rho_adb_branch[i] + 0.0j), np.conj(eigvec_branch[i]).transpose())
    return rho_db_branch

@njit
def rho_db_to_adb_branch(rho_db_branch, eigvec_branch): # transforms branch adibatic to diabatic basis
    rho_adb_branch = np.ascontiguousarray(np.zeros(np.shape(eigvec_branch))) + 0.0j
    for i in range(len(eigvec_branch)):
        rho_adb_branch[i] = np.dot(np.dot(np.conj(eigvec_branch[i]).transpose(), rho_db_branch[i] + 0.0j), eigvec_branch[i])
    return rho_adb_branch


############################################################
#                       QUANTUM FORCE                      #
############################################################


def quantum_force_branch(evecs_branch, act_surf_ind_branch, z_branch, sim):
    if act_surf_ind_branch is None:
        fzc_branch = sim.dh_qc_dzc_branch(sim.h_qc_params, evecs_branch, evecs_branch, z_branch)
    else:
        fzc_branch = sim.dh_qc_dzc_branch(sim.h_qc_params, evecs_branch[range(sim.num_trajs*sim.num_branches),:,act_surf_ind_branch], evecs_branch[range(sim.num_trajs*sim.num_branches),:,act_surf_ind_branch], z_branch)
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
    dab_z = sim.dh_qc_dz_branch(sim.h_qc_params, evec_a[np.newaxis,:], evec_b[np.newaxis,:], z[np.newaxis,:])[0] / ev_diff
    dab_zc = sim.dh_qc_dzc_branch(sim.h_qc_params, evec_a[np.newaxis,:], evec_b[np.newaxis,:], z[np.newaxis,:])[0] / ev_diff
    return dab_z, dab_zc

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

def get_classical_overlap(z_branch, sim):
    out_mat = np.zeros((len(z_branch), len(z_branch)))
    zc_branch = np.conjugate(z_branch)
    q_branch = (1/np.sqrt(2*sim.m*sim.h))*(z_branch + zc_branch)
    p_branch = -1.0j*np.sqrt(sim.h*sim.m/2)*(z_branch - zc_branch)
    for i in range(len(z_branch)):
        for j in range(len(z_branch)):
            out_mat[i,j] = np.exp(-(1/2)*np.sum(np.abs((p_branch[i] - p_branch[j]) * (q_branch[i] - q_branch[j]))))
    return out_mat

@njit
def sign_adjust_branch_0(evecs_branch, evecs_branch_previous, phase_out):
    #signs = np.sign(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch))
    signs = np.sign(np.sum(np.conjugate(evecs_branch_previous)*evecs_branch, axis=1))
    #evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,signs)
    evecs_branch = evecs_branch*(signs.reshape(len(signs),1,len(signs[0])))
    phase_out = phase_out*signs
    return evecs_branch, phase_out

@njit
def sign_adjust_branch_1(evecs_branch, evecs_branch_previous, phase_out):
    #phases = np.exp(-1.0j*np.angle(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch)))
    phases = np.exp(-1.0j*np.angle(np.sum(np.conjugate(evecs_branch_previous)*evecs_branch, axis=1)))
    #evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,phases)
    evecs_branch = evecs_branch*phases.reshape(len(phases),1,len(phases[0]))
    phase_out = phase_out*phases
    return evecs_branch, phase_out

def sign_adjust_branch(evecs_branch, evecs_branch_previous, evals_branch, z_branch, sim):
    # commented out einsum terms found to be slower
    phase_out = np.ones((sim.num_branches, sim.num_states), dtype=complex)
    if sim.gauge_fix >= 1:
        evecs_branch, phase_out = sign_adjust_branch_1(evecs_branch, evecs_branch_previous, phase_out)
    if sim.gauge_fix >= 2:
        dab_phase_mat = np.ones((len(evecs_branch),len(evecs_branch)),dtype=complex)
        for i in range(len(evecs_branch)):
            dabQ_phase_list, dabP_phase_list = get_dab_phase(evals_branch[i], evecs_branch[i], z_branch[i], sim)
            dab_phase_list = np.conjugate(dabQ_phase_list)
            dab_phase_mat[i] = dab_phase_list
            phase_out[i] *= dab_phase_list
        #    evecs_branch[i] = np.einsum('jk,k->jk',evecs_branch[i],dab_phase_list)
        evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,dab_phase_mat,optimize='greedy')
    if sim.gauge_fix >= 0:
        evecs_branch, phase_out = sign_adjust_branch_0(evecs_branch, evecs_branch_previous, phase_out)
    return evecs_branch, phase_out


@njit
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


@njit
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


def get_branch_pair_eigs(z_branch, evecs_branch_pair_previous, sim):
    evals_branch_pair = np.zeros((sim.num_branches, sim.num_branches, sim.num_states))
    evecs_branch_pair = np.zeros((sim.num_branches, sim.num_branches, sim.num_states, sim.num_states), dtype=complex)
    h_q = sim.h_q()
    for i in range(sim.num_branches):
        for j in range(i + 1, sim.num_branches):
            z_branch_ij = np.array([(z_branch[i] + z_branch[j])/2])
            evals_branch_pair[i,j], evecs_branch_pair[i,j] = np.linalg.eigh(h_q + sim.h_qc_branch(sim.h_qc_params, z_branch_ij)[0])
            evecs_branch_pair[i,j], _ = sign_adjust_branch(evecs_branch_pair[i,j][np.newaxis,:,:], evecs_branch_pair_previous[i,j][np.newaxis,:,:], 
                                                           evals_branch_pair[i,j][np.newaxis,:], z_branch_ij[np.newaxis,:], sim)
            evals_branch_pair[j,i] = evals_branch_pair[i,j]
            evecs_branch_pair[j,i] = evecs_branch_pair[j,i]
    return evals_branch_pair, evecs_branch_pair

def no_observables(sim, state_vars):
    return {}


def harmonic_oscillator_hop(sim, z, delta_z, ev_diff):
    """
    Carries out the hopping procedure for a harmonic oscillator Hamiltonian, defined on a single branch only. 
    :param z: z coordinate
    :param delta_z: rescaling direction
    :param ev_diff: change in quantum energy following a hop: e_{final} - e_{initial}
    :param sim: simulation object
    :return z, hopped: updated z coordinate and boolean indicating if a hop has or has not occured
    """
    hopped = False
    delta_zc = np.conj(delta_z)
    zc = np.conj(z)
    akj_z = np.real(np.sum(sim.h * delta_zc * delta_z))
    bkj_z = np.real(np.sum(1j * sim.h * (zc * delta_z - z * delta_zc)))
    ckj_z = ev_diff
    disc = bkj_z ** 2 - 4 * akj_z * ckj_z
    if disc >= 0:
        if bkj_z < 0:
            gamma = bkj_z + np.sqrt(disc)
        else:
            gamma = bkj_z - np.sqrt(disc)
        if akj_z == 0:
            gamma = 0
        else:
            gamma = gamma / (2 * akj_z)
        # adjust classical coordinate
        z = z - 1.0j * np.real(gamma) * delta_z
        hopped = True
    return z, hopped

def harmonic_oscillator_bolztmann_init_classical(sim, seed=None):
    """
    Initialize classical coordiantes according to Boltzmann statistics
    :param sim: simulation object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(w*h/2)*(q + i*(p/((w*h))), z* = sqrt(w*h/2)*(q - i*(p/((w*h)))
    """
    np.random.seed(seed)
    q = np.random.normal(loc=0, scale=np.sqrt(sim.temp / (sim.m * (sim.h ** 2))), size=sim.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=np.sqrt(sim.temp), size=sim.num_classical_coordinates)
    z = np.sqrt(sim.h * sim.m / 2) * (q + 1.0j * (p / (sim.h * sim.m)))
    return z


def harmonic_oscillator_h_c_branch(h_c_params, z_branch):
        h = h_c_params
        return np.real(np.sum(h[np.newaxis,:]*np.conj(z_branch)*z_branch, axis=1))
def harmonic_oscillator_dh_c_dz_branch(h_c_params, z_branch):
    """
    Gradient of harmonic oscillator hamiltonian wrt z_branch
    :param z_branch: z coordinate in each branch
    :param sim: simulation object
    :return:
    """
    h = h_c_params
    return h[np.newaxis, :] * np.conj(z_branch)

def harmonic_oscillator_dh_c_dzc_branch(h_c_params, z_branch):
    """
    Gradient of harmonic oscillator hamiltonian wrt zc_branch
    :param z_branch: z coordinate in each branch
    :param sim: simulation object
    :return:
    """
    h = h_c_params
    return h[np.newaxis, :] * z_branch

def load_defaults(sim):
    var_names = list(sim.__dict__.keys())
    defaults = {
        'hop': harmonic_oscillator_hop,
        'init_classical': harmonic_oscillator_bolztmann_init_classical,
        'h_c_branch': harmonic_oscillator_h_c_branch,
        'dh_c_dz_branch': harmonic_oscillator_dh_c_dz_branch,
        'dh_c_dzc_branch': harmonic_oscillator_dh_c_dzc_branch,
        'h_c_params' : (sim.h),
        'h_qc_params' : None,
        'h_q_params' : None,
        'dynamics_method': None,
        'tmax': None,
        'dt': None,
        'dt_bath': None,
        'calc_mf_obs':None,
        'calc_fssh_obs':None,
        'calc_cfssh_obs':None,
        'mf_observables':None,
        'fssh_observables':None,
        'cfssh_observables':None,
        'gauge_fix':None,
        'dmat_const':None,
        'cfssh_branch_pair_update':None,
        'num_class_coords':None,
        'state_vars_list':[],
        'mf_observables':no_observables,
        'fssh_observables':no_observables,
        'cfssh_observables':no_observables,
        'observables':no_observables,
        'temp':None
        }
    for name in defaults.keys():
        if not(name in list(var_names)):
            sim.__dict__[name] = defaults[name]
    return sim

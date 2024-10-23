from numba import njit
import numpy as np
import dill as pickle


############################################################
#                       RUNGE-KUTTA                       #
############################################################


def rk4_c(state, z_coord, qfzc, dt):
    """ 4-th order Runge-Kutta integrator for the z_coord coordinate with force qfzc"""
    k1 = -1.0j * (state.model.dh_c_dzc(state, z_coord) + qfzc)
    k2 = -1.0j * (state.model.dh_c_dzc(state, z_coord + 0.5 * dt * k1) + qfzc)
    k3 = -1.0j * (state.model.dh_c_dzc(state, z_coord + 0.5 * dt * k2) + qfzc)
    k4 = -1.0j * (state.model.dh_c_dzc(state, z_coord + dt * k3) + qfzc)
    z_coord = z_coord + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z_coord


def rk4_q(h, psi, dt):
    # h and psi have to have shapes ...ij,...j
    k1 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi), optimize='greedy'))
    k2 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi) + 0.5 * dt * k1, optimize='greedy'))
    k3 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi) + 0.5 * dt * k2, optimize='greedy'))
    k4 = (-1j * np.einsum('...ij,...j->...i', h, np.ascontiguousarray(psi) + dt * k3, optimize='greedy'))
    psi = psi + dt * 0.166667 * (k1 + 2 * k2 + 2 * k3 + k4)
    return psi


############################################################
#                   BASIS TRANSFORMATIONS                  #
############################################################

def vec_adb_to_db(vec_adb, eigvecs):
    return np.einsum('...ij,...j->...i', eigvecs, vec_adb, optimize='greedy')


def vec_db_to_adb(vec_db, eigvecs):
    return np.einsum('...ji,...j->...i', np.conj(eigvecs), vec_db, optimize='greedy')


def mat_adb_to_db(mat_adb, eigvecs):
    return np.einsum('...ni,...ij,...mj->...nm', eigvecs, mat_adb, np.conj(eigvecs), optimize='greedy')


def mat_db_to_adb(mat_db, eigvecs):
    return np.einsum('...ni,...nm,...mj->...ij', np.conj(eigvecs), mat_db, eigvecs, optimize='greedy')


############################################################
#                     QUANTUM EVOLUTION                    #
############################################################


def evolve_wf_eigs(wf_db, eigvals, eigvecs, dt):
    # construct eigenvalue exponential
    evals_exp_branch = np.exp(-1.0j * eigvals * dt)
    # transform wavefunctions to adiabatic basis
    # wf_adb = np.copy(psi_db_to_adb_branch(wf_db, eigvecs)) # TODO remove this comment
    wf_adb = np.copy(vec_db_to_adb(wf_db, eigvecs))
    # multiply by propagator
    wf_adb = np.copy(evals_exp_branch * wf_adb)
    # transform back to diabatic basis
    # wf_db = np.copy(psi_adb_to_db_branch(wf_adb, eigvecs)) # TODO remove this comment
    wf_db = np.copy(vec_adb_to_db(wf_adb, eigvecs))
    return wf_db, wf_adb


############################################################
#                  NONADIABATIC COUPLINGS                 #
############################################################


def get_der_couple(model, state, evec_a, evec_b, ev_diff):
    """
    Computes the nonadiabatic coupling using the formula
    d_{ab}^{z(zc)} = <a|\nabla_{z(zc)} H|b>/(e_{b} - e_{a})
    :param evec_a: |a>
    :param evec_b: |b>
    :param ev_diff: e_{b} - e_{a}
    :return: d_{ab}^{z} and d_{ab}^{zc}
    """
    der_couple_z = model.dh_qc_dz(model, state, evec_a, evec_b) / ev_diff
    der_couple_zc = model.dh_qc_dzc(model, state, evec_a, evec_b) / ev_diff
    return der_couple_z, der_couple_zc


############################################################
#                       GAUGE FIXING                      #
############################################################


def get_der_couple_phase(model, state, evals, evecs):
    """
    Computes the diagonal gauge transformation G such that (VG)^{dagger}\nabla(VG) is real-valued. :param evals:
    eigenvalues :param evecs: eigenvectors (V) :param diff_vars: sparse matrix variables of \nabla_{z} H and \nabla_{
    zc} H (stored in model.dq_vars) :return: der_couple_q_phase (diag(G^{dagger}) calculated using d_{ab}^{q}),
    der_couple_p_phase ( diag(G^{dagger}) calculated using d_{ab}^{p})
    """
    der_couple_q_phase = np.ones(len(evals), dtype=complex)
    der_couple_p_phase = np.ones(len(evals), dtype=complex)
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
        der_couple_z, der_couple_zc = get_der_couple(model, state, evec_i, evec_j,
                                       ev_diff + plus)  # (evec_i, evec_j, ev_diff + plus, z, model)
        # convert to q/p nonadiabatic couplings
        der_couple_q = np.sqrt(model.pq_weight * model.mass / 2) * (der_couple_z + der_couple_zc)
        der_couple_p = np.sqrt(1 / (2 * model.pq_weight * model.mass)) * 1.0j * (der_couple_z - der_couple_zc)
        der_couple_q_angle = np.angle(der_couple_q[np.argmax(np.abs(der_couple_q))])
        der_couple_p_angle = np.angle(der_couple_p[np.argmax(np.abs(der_couple_p))])
        if np.max(np.abs(der_couple_q)) < 1e-14:
            der_couple_q_angle = 0
        if np.max(np.abs(der_couple_p)) < 1e-14:
            der_couple_p_angle = 0
        der_couple_q_phase[i + 1:] = np.exp(1.0j * der_couple_q_angle) * der_couple_q_phase[i + 1:]
        der_couple_p_phase[i + 1:] = np.exp(1.0j * der_couple_p_angle) * der_couple_p_phase[i + 1:]
    return der_couple_q_phase, der_couple_p_phase


@njit
def sign_adjust_branch_0(evecs_branch, evecs_branch_previous, phase_out):
    # signs = np.sign(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch))
    signs = np.sign(np.sum(np.conjugate(evecs_branch_previous) * evecs_branch, axis=1))
    # evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,signs)
    evecs_branch = evecs_branch * (signs.reshape(len(signs), 1, len(signs[0])))
    phase_out = phase_out * signs
    return evecs_branch, phase_out


@njit
def sign_adjust_branch_1(evecs_branch, evecs_branch_previous, phase_out):  # make name more descriptive
    # phases = np.exp(-1.0j*np.angle(np.einsum('ijk,ijk->ik',np.conjugate(evecs_branch_previous),evecs_branch)))
    phases = np.exp(-1.0j * np.angle(np.sum(np.conjugate(evecs_branch_previous) * evecs_branch, axis=1)))
    # evecs_branch = np.einsum('ijk,ik->ijk',evecs_branch,phases)
    evecs_branch = evecs_branch * phases.reshape(len(phases), 1, len(phases[0]))
    phase_out = phase_out * phases
    return evecs_branch, phase_out


def sign_adjust_branch(evecs_branch, evecs_branch_previous, evals_branch, z_coord, model):
    # commented out einsum terms found to be slower
    phase_out = np.ones((model.num_branches, model.num_states), dtype=complex)
    if model.gauge_fix >= 1:
        evecs_branch, phase_out = sign_adjust_branch_1(evecs_branch, evecs_branch_previous, phase_out)
    if model.gauge_fix >= 2:
        der_couple_phase_mat = np.ones((len(evecs_branch), len(evecs_branch)), dtype=complex)
        for i in range(len(evecs_branch)):
            der_couple_q_phase_list, der_couple_p_phase_list = get_der_couple_phase(evals_branch[i], evecs_branch[i],
                                                                                    z_coord[i], model)
            der_couple_phase_list = np.conjugate(der_couple_q_phase_list)
            der_couple_phase_mat[i] = der_couple_phase_list
            phase_out[i] *= der_couple_phase_list
        #    evecs_branch[i] = np.einsum('jk,k->jk',evecs_branch[i],der_couple_phase_list)
        evecs_branch = np.einsum('ijk,ik->ijk', evecs_branch, der_couple_phase_mat, optimize='greedy')
    if model.gauge_fix >= 0:
        evecs_branch, phase_out = sign_adjust_branch_0(evecs_branch, evecs_branch_previous, phase_out)
    return evecs_branch, phase_out


def sign_adjust_branch_pair_eigs(z_coord, eigvecs_branch_pair, eigvals_branch_pair, eigvecs_branch_pair_previous,
                                 model):
    for i in range(model.num_branches):
        for j in range(i + 1, model.num_branches):
            z_coord_ij = np.array([(z_coord[i] + z_coord[j]) / 2])
            eigvecs_branch_pair[i, j], _ = sign_adjust_branch(eigvecs_branch_pair[i, j][np.newaxis, :, :],
                                                              eigvecs_branch_pair_previous[i, j][np.newaxis, :, :],
                                                              eigvals_branch_pair[i, j][np.newaxis, :],
                                                              z_coord_ij[np.newaxis, :], model)
            eigvecs_branch_pair[j, i] = eigvecs_branch_pair[i, j]
    return eigvals_branch_pair, eigvecs_branch_pair


############################################################
#                       MISCELLANEOUS                      #
############################################################


def get_branch_pair_eigs(z_coord, model):
    eigvals_branch_pair = np.zeros((model.num_branches, model.num_branches, model.num_states))
    eigvecs_branch_pair = np.zeros((model.num_branches, model.num_branches, model.num_states, model.num_states),
                                   dtype=complex)
    h_q = model.h_q(model.h_q_params)
    for i in range(model.num_branches):
        for j in range(i + 1, model.num_branches):
            z_coord_ij = np.array([(z_coord[i] + z_coord[j]) / 2])
            eigvals_branch_pair[i, j], eigvecs_branch_pair[i, j] = np.linalg.eigh(
                h_q + model.h_qc(model.h_qc_params, z_coord_ij)[0])
            eigvals_branch_pair[j, i] = eigvals_branch_pair[i, j]
            eigvecs_branch_pair[j, i] = eigvecs_branch_pair[i, j]
    return eigvals_branch_pair, eigvecs_branch_pair


def get_classical_overlap(z_coord, model):
    out_mat = np.zeros((len(z_coord), len(z_coord)))
    zc_branch = np.conjugate(z_coord)
    q_branch = (1 / np.sqrt(2 * model.mass * model.pq_weight)) * (z_coord + zc_branch)
    p_branch = -1.0j * np.sqrt(model.pq_weight * model.mass / 2) * (z_coord - zc_branch)
    for i in range(len(z_coord)):
        for j in range(len(z_coord)):
            out_mat[i, j] = np.exp(-(1 / 2) * np.sum(np.abs((p_branch[i] - p_branch[j]) * (q_branch[i] - q_branch[j]))))
    return out_mat


# @njit
def matprod_sparse(shape, ind, mels, vec1, vec2):  # calculates <1|mat|2> TODO make this jit compatible?
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
   # i_ind, j_ind, k_ind = ind
    prod = np.conj(vec1)[..., ind[-2]] * mels[...] * vec2[..., ind[-1]]
    #out_mat = np.zeros((*np.shape(vec1)[:-1], shape[0])) + 0.0j
    #for i in range(len(ind[0])):
    #    out_mat[..., i_ind[i]] += prod[..., i]
    #out_mat = np.zeros((*np.shape(vec1)[:-1], shape[-3])) + 0.0j
    #np.put(out_mat, )
    print(np.shape(ind))
    print(np.shape(out_mat), np.shape(ind[:-2]), np.shape(prod))
    print(ind[:-2])
    out_mat = np.zeros(len(ind), dtype=complex)
    np.add.at(out_mat, ind, prod.flatten())
    return out_mat.reshape(shape[:-2])


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


############################################################
#        DEFAULT FUNCTIONS FOR HARMONIC OSCILLATOR        #
############################################################


def harmonic_oscillator_hop(model, z, delta_z, ev_diff):
    """
    Carries out the hopping procedure for a harmonic oscillator Hamiltonian, defined on a single branch only. 
    :param z: z coordinate
    :param delta_z: rescaling direction
    :param ev_diff: change in quantum energy following a hop: e_{final} - e_{initial}
    :param model: model object
    :return z, hopped: updated z coordinate and boolean indicating if a hop has or has not occured
    """
    hopped = False
    delta_zc = np.conj(delta_z)
    zc = np.conj(z)
    akj_z = np.real(np.sum(model.pq_weight * delta_zc * delta_z))
    bkj_z = np.real(np.sum(1j * model.pq_weight * (zc * delta_z - z * delta_zc)))
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


def harmonic_oscillator_boltzmann_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to Boltzmann statistics
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed(seed)
    q = np.random.normal(loc=0, scale=np.sqrt(model.temp / (model.mass * (model.pq_weight ** 2))),
                         size=model.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=np.sqrt(model.temp), size=model.num_classical_coordinates)
    z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
    return z


def harmonic_oscillator_wigner_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to Wigner distribution of the ground state of a harmonic oscillator
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed(seed)
    q = np.random.normal(loc=0, scale=np.sqrt(1 / (2 * model.pq_weight * model.mass)),
                         size=model.num_classical_coordinates)
    p = np.random.normal(loc=0, scale=np.sqrt((model.mass * model.pq_weight) / 2), size=model.num_classical_coordinates)
    z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
    return z


def harmonic_oscillator_focused_init_classical(model, seed=None):
    """
    Initialize classical coordiantes according to focused sampling of the 
    Wigner distribution of the ground state of a harmonic oscillator
    :param model: model object with temperature, harmonic oscillator mass and frequency
    :return: z = sqrt(m*pq_weight/2)*(q + i*(p/((m*pq_weight))), z* = sqrt(m*pq_weight/2)*(q - i*(p/((m*pq_weight)))
    """
    np.random.seed()
    phase = np.random.rand(model.num_classical_coordinates) * 2 * np.pi
    q = np.sqrt(1 / (model.pq_weight * model.mass)) * np.cos(phase)
    p = np.sqrt((model.mass * model.pq_weight)) * np.sin(phase)
    z = np.sqrt(model.pq_weight * model.mass / 2) * (q + 1.0j * (p / (model.pq_weight * model.mass)))
    return z


def harmonic_oscillator_h_c(state, z_coord):
    return np.real(np.sum(state.model.pq_weight[..., :] * np.conj(state.z_coord) * state.z_coord, axis=(0,1)))


def harmonic_oscillator_dh_c_dz(state, z_coord):
    """
    Gradient of harmonic oscillator hamiltonian wrt z_coord
    :param z_coord: z coordinate in each branch
    :return:
    """
    return state.model.pq_weight[np.newaxis, np.newaxis, :] * np.conj(z_coord)


def harmonic_oscillator_dh_c_dzc(state, z_coord):
    """
    Gradient of harmonic oscillator hamiltonian wrt zc_branch
    :param z_coord: z coordinate in each branch
    :return:
    """
    return state.model.pq_weight[np.newaxis, np.newaxis, :] * z_coord


############################################################
#                      FILE MANAGEMENT                     #
############################################################


def save_pickle(obj, filename):
    file = open(filename, 'wb')
    pickle.dump(obj, file)
    file.close()
    return


def load_pickle(filename):
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


############################################################
#                    DYNAMICS FUNCTIONS                   #
############################################################


def initialize_timesteps(model):
    model.tmax_n = np.round(model.tmax / model.dt, 1).astype(int)
    model.dt_output_n = np.round(model.dt_output / model.dt, 1).astype(int)
    model.tdat = np.arange(0, model.tmax_n + 1, 1) * model.dt
    model.tdat_n = np.arange(0, model.tmax_n + 1, 1)
    model.tdat_output = np.arange(0, model.tmax_n + 1, model.dt_output_n) * model.dt
    model.tdat_ouput_n = np.arange(0, model.tmax_n + 1, model.dt_output_n)
    return model


def evaluate_observables_t(recipe):
    observables_dic = dict()
    state_dic = vars(recipe.state)
    for key in recipe.output_names:
        observables_dic[key] = state_dic[key]
    return observables_dic

